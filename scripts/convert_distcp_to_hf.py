#!/usr/bin/env python3
"""Convert DTensor distributed checkpoint (TP=8) to HuggingFace format.

This script converts PyTorch DTensor distributed checkpoints that were saved
with tensor_parallel_size=8 into a single HuggingFace-compatible model.
Optionally uploads the converted model to HuggingFace Hub.

Usage:
    # Local conversion only
    uv run python scripts/convert_distcp_to_hf.py \
        --checkpoint-dir ~/Downloads/sft_cot \
        --output-path ./models/sft_cot

    # Convert and upload to HuggingFace Hub (public)
    uv run python scripts/convert_distcp_to_hf.py \
        --checkpoint-dir ~/Downloads/sft_cot \
        --output-path ./models/sft_cot \
        --repo-id ft-llm-team-mkj/sft-cot-model

    # Convert and upload to HuggingFace Hub (private)
    uv run python scripts/convert_distcp_to_hf.py \
        --checkpoint-dir ~/Downloads/sft_cot \
        --output-path ./models/sft_cot \
        --repo-id ft-llm-team-mkj/sft-cot-model \
        --private
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_ORG = "ft-llm-team-mkj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DTensor distributed checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to the checkpoint directory containing policy/weights/*.distcp files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for the HuggingFace model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="ft-llm-team-mkj/baseline-model-instruct-8k-yarn",
        help="Base model name for loading config (default: from checkpoint config)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"HuggingFace Hub repository ID (e.g., '{DEFAULT_ORG}/model-name'). "
        f"If organization is omitted, defaults to '{DEFAULT_ORG}'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model converted from DTensor checkpoint",
        help="Commit message for HuggingFace Hub upload",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Don't create repository if it doesn't exist",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def load_distcp_weights(weights_dir: Path) -> dict[str, torch.Tensor]:
    """Load DTensor distributed checkpoint and merge shards."""
    print(f"Loading distributed checkpoint from {weights_dir}")

    # Load using dcp.load with no_dist mode for single-process loading
    storage_reader = dcp.FileSystemReader(weights_dir)

    # Read metadata to understand the checkpoint structure
    metadata = storage_reader.read_metadata()

    print("Reading checkpoint metadata...")
    print(f"Found {len(metadata.state_dict_metadata)} tensors")

    # Create placeholder tensors based on metadata
    # dcp.load() modifies state_dict in-place, doesn't return it
    state_dict = {}
    for key, tensor_meta in metadata.state_dict_metadata.items():
        # Get the full tensor shape from metadata
        shape = tensor_meta.size
        # Use the properties from metadata if available
        dtype = getattr(tensor_meta.properties, 'dtype', torch.bfloat16) if hasattr(tensor_meta, 'properties') else torch.bfloat16
        state_dict[key] = torch.empty(shape, dtype=dtype)

    # Load the checkpoint - this populates state_dict in-place
    dcp.load(
        state_dict=state_dict,
        storage_reader=storage_reader,
    )

    return state_dict


def merge_tp_weights(
    state_dict: dict[str, torch.Tensor], tp_size: int = 8
) -> dict[str, torch.Tensor]:
    """Merge tensor parallel shards into full tensors.

    For Llama-style models with TP, typical sharding patterns are:
    - Column parallel (q_proj, k_proj, v_proj, gate_proj, up_proj): concat on dim=0
    - Row parallel (o_proj, down_proj): concat on dim=1
    - Embeddings: concat on dim=0 (vocab dimension)
    - LM head: concat on dim=0 (vocab dimension)
    """
    merged = {}

    # Define sharding patterns for Llama architecture
    # Column parallel layers: split output dimension (dim=0)
    col_parallel_patterns = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    ]

    # Row parallel layers: split input dimension (dim=1)
    row_parallel_patterns = [
        "o_proj.weight",
        "down_proj.weight",
    ]

    for key, tensor in state_dict.items():
        # Check if this is a sharded tensor (DTensor)
        if hasattr(tensor, "full_tensor"):
            # Already a DTensor, get the full tensor
            merged[key] = tensor.full_tensor()
        else:
            # Regular tensor, keep as is
            merged[key] = tensor

    return merged


def convert_to_hf_format(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert state dict keys to HuggingFace format if needed."""
    hf_state_dict = {}

    for key, tensor in state_dict.items():
        # Remove 'model.' prefix if present
        new_key = key
        if key.startswith("model."):
            new_key = key[6:]  # Remove 'model.' prefix

        hf_state_dict[new_key] = tensor

    return hf_state_dict


def save_hf_model(
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    base_model: str,
    tokenizer_path: Path,
    dtype: torch.dtype,
) -> None:
    """Save the model in HuggingFace format."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert tensors to target dtype
    print(f"Converting tensors to {dtype}")
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            converted_state_dict[key] = tensor.to(dtype)
        else:
            converted_state_dict[key] = tensor

    # Save as safetensors
    print("Saving model weights as safetensors...")
    save_file(converted_state_dict, output_path / "model.safetensors")

    # Copy config from base model
    print(f"Loading config from {base_model}")
    config = AutoConfig.from_pretrained(base_model)
    config.save_pretrained(output_path)

    # Copy tokenizer files
    print(f"Copying tokenizer from {tokenizer_path}")
    for file in tokenizer_path.iterdir():
        if file.is_file():
            shutil.copy(file, output_path / file.name)

    # Also try to save tokenizer properly for full compatibility
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer via AutoTokenizer: {e}")
        print("Tokenizer files were copied manually.")

    # Create model index for safetensors
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in converted_state_dict.values())},
        "weight_map": {key: "model.safetensors" for key in converted_state_dict.keys()},
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Model saved to {output_path}")


def upload_to_hub(
    output_path: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    create_repo: bool,
) -> str:
    """Upload the converted model to HuggingFace Hub.

    Args:
        output_path: Path to the local model directory
        repo_id: HuggingFace Hub repository ID
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        create_repo: Whether to create the repository if it doesn't exist

    Returns:
        URL of the uploaded model on HuggingFace Hub
    """
    # Add default organization if not specified
    if "/" not in repo_id:
        repo_id = f"{DEFAULT_ORG}/{repo_id}"
        print(f"Using default organization: {repo_id}")

    api = HfApi()

    # Create repository if needed
    if create_repo:
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            visibility = "private" if private else "public"
            print(f"Repository '{repo_id}' ready ({visibility})")
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                raise RuntimeError(
                    "Authentication failed. Please run 'huggingface-cli login' "
                    "or set the HF_TOKEN environment variable."
                ) from e
            raise

    # Upload the model
    print(f"Uploading model to {repo_id}...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    hub_url = f"https://huggingface.co/{repo_id}"
    print(f"Model uploaded successfully: {hub_url}")
    return hub_url


def main() -> None:
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    dtype = get_dtype(args.dtype)

    # Validate checkpoint directory structure
    weights_dir = checkpoint_dir / "policy" / "weights"
    tokenizer_dir = checkpoint_dir / "policy" / "tokenizer"

    if not weights_dir.exists():
        raise ValueError(f"Weights directory not found: {weights_dir}")
    if not tokenizer_dir.exists():
        raise ValueError(f"Tokenizer directory not found: {tokenizer_dir}")

    # Check for .distcp files
    distcp_files = list(weights_dir.glob("*.distcp"))
    if not distcp_files:
        raise ValueError(f"No .distcp files found in {weights_dir}")
    print(f"Found {len(distcp_files)} .distcp shards")

    # Try to read base model from config.yaml
    base_model = args.base_model
    config_file = checkpoint_dir / "config.yaml"
    if config_file.exists():
        try:
            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f)
            if "policy" in config and "model_name" in config["policy"]:
                base_model = config["policy"]["model_name"]
                print(f"Using base model from config: {base_model}")
        except Exception as e:
            print(f"Warning: Could not read config.yaml: {e}")

    # Load and convert weights
    print("\n=== Loading distributed checkpoint ===")
    state_dict = load_distcp_weights(weights_dir)

    print(f"\nLoaded {len(state_dict)} tensors")
    if state_dict:
        # Print some sample keys
        sample_keys = list(state_dict.keys())[:5]
        print("Sample keys:", sample_keys)

    print("\n=== Converting to HuggingFace format ===")
    hf_state_dict = convert_to_hf_format(state_dict)

    print("\n=== Saving HuggingFace model ===")
    save_hf_model(
        hf_state_dict,
        output_path,
        base_model,
        tokenizer_dir,
        dtype,
    )

    print("\n=== Conversion complete ===")
    print(f"Output: {output_path}")

    # Upload to HuggingFace Hub if requested
    if args.repo_id:
        print("\n=== Uploading to HuggingFace Hub ===")
        hub_url = upload_to_hub(
            output_path=output_path,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            create_repo=not args.no_create_repo,
        )
        print(f"\nModel available at: {hub_url}")
        print("\nTo load the model from Hub:")
        print(f"  python -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{args.repo_id}')\"")
    else:
        print("\nTo verify the model:")
        print(f"  ls {output_path}/")
        print(f"  python -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{output_path}')\"")


if __name__ == "__main__":
    main()
