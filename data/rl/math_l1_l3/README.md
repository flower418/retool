# MATH Level 1-3 RL Data

This directory defines the RL prompt dataset used by the DAPO/GRPO-style ReTool
run. The committed files are the prompt template and metadata. Generated
`train.parquet` and `val.parquet` files are intentionally ignored.

Generate the parquet files with:

```bash
python scripts/prepare_math_level_data.py \
  --out-dir data/rl/math_l1_l3 \
  --min-level 1 \
  --max-level 3 \
  --repeat 10 \
  --val-size 128 \
  --seed 2026
```

The output schema is veRL-compatible: each row has `prompt`,
`reward_model.ground_truth`, and `extra_info`.
