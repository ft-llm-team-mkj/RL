#!/usr/bin/env python3
"""
HuggingFace データセットを SFT用 JSONL に変換するスクリプト

使用例:
    python tools/convert_hf_to_jsonl.py \
        --dataset ft-llm-team-mkj/tir-merged \
        --output_dir data/tir-merged \
        --val_ratio 0.1 \
        --seed 42
"""

import argparse
import json
import os
from datasets import load_dataset


def convert_dataset(
    dataset_name: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    messages_key: str = "messages",
):
    """HuggingFaceデータセットをJSONLに変換"""

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # データセット情報を表示
    print(f"\n=== Dataset Info ===")
    print(ds)
    print(f"\nColumns: {ds['train'].column_names}")

    train_data = ds['train']

    # valスプリットがあるか確認
    if 'validation' in ds:
        print("Using existing validation split")
        train_split = train_data
        val_split = ds['validation']
    elif 'test' in ds:
        print("Using existing test split as validation")
        train_split = train_data
        val_split = ds['test']
    else:
        # Shuffle and split
        print(f"Creating validation split with ratio {val_ratio}")
        train_data = train_data.shuffle(seed=seed)
        split_idx = int(len(train_data) * (1 - val_ratio))
        train_split = train_data.select(range(split_idx))
        val_split = train_data.select(range(split_idx, len(train_data)))

    print(f"\nTrain: {len(train_split)}, Val: {len(val_split)}")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # JSONL保存
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    def save_jsonl(data, path, messages_key):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                # messagesキーのみを抽出
                output = {"messages": item[messages_key]}
                json.dump(output, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved: {path}")

    save_jsonl(train_split, train_path, messages_key)
    save_jsonl(val_split, val_path, messages_key)

    # サンプル表示
    print(f"\n=== Sample from {train_path} ===")
    with open(train_path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        print(f"Messages count: {len(sample['messages'])}")
        for msg in sample['messages']:
            content = msg['content']
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  {msg['role']}: {content}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to JSONL for SFT"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., ft-llm-team-mkj/tir-merged)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--messages_key",
        type=str,
        default="messages",
        help="Key for messages in dataset (default: messages)"
    )

    args = parser.parse_args()

    convert_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        messages_key=args.messages_key,
    )


if __name__ == "__main__":
    main()
