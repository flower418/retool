#!/usr/bin/env python3
"""Prepare EleutherAI Hendrycks MATH levels for veRL/DAPO RL training."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

SUBJECTS = (
    "prealgebra",
    "algebra",
    "number_theory",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "precalculus",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_DATA_DIR = "data/rl/math_l1_l3"
DEFAULT_PROMPT_TEMPLATE_PATH = "data/rl/math_l1_l3/prompt.txt"

LEVEL_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_RL_DATA_DIR,
        help="Output directory for train.parquet, val.parquet, and meta.json.",
    )
    parser.add_argument("--dataset", default="EleutherAI/hendrycks_math")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=10, help="Repeat train rows after split.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE_PATH,
        help="Prompt template file. Must contain {problem}.",
    )
    return parser.parse_args()


def load_prompt_template(path: str) -> str:
    template = resolve_repo_path(path).read_text(encoding="utf-8").strip()
    if "{problem}" not in template:
        raise ValueError(f"prompt template must contain {{problem}}: {path}")
    return template


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def level_to_int(level: str) -> int | None:
    match = LEVEL_RE.search(str(level))
    return int(match.group(1)) if match else None


def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return text[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(text: str) -> str:
    if "\\boxed " in text:
        return text[len("\\boxed ") :].strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text[len("\\boxed{") : -1].strip()
    if text.startswith("\\fbox{") and text.endswith("}"):
        return text[len("\\fbox{") : -1].strip()
    return text.strip()


def extract_answer(solution: str) -> str | None:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def make_row(row: dict[str, Any], subject: str, index: int, prompt_template: str) -> dict[str, Any] | None:
    level = level_to_int(row.get("level", ""))
    if level is None:
        return None
    answer = extract_answer(str(row.get("solution", "")))
    if not answer:
        return None

    problem = str(row["problem"])
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": prompt_template.format(problem=problem)}],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {
            "index": f"hendrycks_math:{subject}:train:{index}",
            "source": "EleutherAI/hendrycks_math",
            "subject": subject,
            "level": level,
            "type": row.get("type"),
        },
    }


def main() -> None:
    args = parse_args()
    import pandas as pd
    from datasets import load_dataset

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = load_prompt_template(args.prompt_template)

    rows: list[dict[str, Any]] = []
    counts_by_subject: dict[str, int] = {}
    counts_by_level: dict[int, int] = {}
    dropped_no_answer = 0

    for subject in args.subjects:
        dataset = load_dataset(args.dataset, subject, split="train")
        kept_for_subject = 0
        for index, item in enumerate(dataset):
            level = level_to_int(item.get("level", ""))
            if level is None or not (args.min_level <= level <= args.max_level):
                continue
            converted = make_row(item, subject, index, prompt_template)
            if converted is None:
                dropped_no_answer += 1
                continue
            rows.append(converted)
            kept_for_subject += 1
            counts_by_level[level] = counts_by_level.get(level, 0) + 1
        counts_by_subject[subject] = kept_for_subject

    if len(rows) <= args.val_size:
        raise ValueError(f"not enough rows: rows={len(rows)} val_size={args.val_size}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size :]
    if args.repeat > 1:
        train_rows = train_rows * args.repeat
        rng.shuffle(train_rows)

    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)

    meta = {
        "dataset": args.dataset,
        "subjects": args.subjects,
        "min_level": args.min_level,
        "max_level": args.max_level,
        "seed": args.seed,
        "prompt_template": args.prompt_template,
        "repeat": args.repeat,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unique_rows_before_split": len(rows),
        "dropped_no_answer": dropped_no_answer,
        "counts_by_subject": counts_by_subject,
        "counts_by_level": counts_by_level,
        "train_file": str(out_dir / "train.parquet"),
        "val_file": str(out_dir / "val.parquet"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
