#!/usr/bin/env python3
"""Prepare ReTool SFT JSONL for verl's MultiTurnSFTDataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL with {'messages': [...]} rows.")
    parser.add_argument("--model", required=True, help="Tokenizer/model path.")
    parser.add_argument("--out-dir", required=True, help="Directory for train.parquet and val.parquet.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_messages(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj.get("messages") if isinstance(obj, dict) else obj
            if not isinstance(messages, list):
                raise ValueError(f"line {line_no}: missing messages list")
            roles = [m.get("role") for m in messages if isinstance(m, dict)]
            if roles.count("user") < 1 or roles.count("assistant") < 1:
                raise ValueError(f"line {line_no}: expected user and assistant messages")
            rows.append({"messages": messages})
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_messages(input_path)

    kept: list[dict] = []
    dropped = 0
    lengths: list[int] = []
    for row in rows:
        length = len(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=True))
        if length > args.max_length:
            dropped += 1
            continue
        row["token_length"] = length
        kept.append(row)
        lengths.append(length)

    if len(kept) <= args.val_size:
        raise ValueError(f"not enough rows after filtering: kept={len(kept)} val_size={args.val_size}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    val_rows = kept[: args.val_size]
    train_rows = kept[args.val_size :]

    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)

    meta = {
        "input": str(input_path),
        "model": args.model,
        "max_length": args.max_length,
        "total_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_too_long": dropped,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "min_length": min(lengths),
        "max_length_observed": max(lengths),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
