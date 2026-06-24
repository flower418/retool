# ReTool Training Pipeline

This repository contains the ReTool workflow for code-assisted math training:

```text
question data -> verified SFT rows -> veRL SFT
-> sandboxed DAPO/GRPO-style RL -> merged HF checkpoint
-> sandbox inference and evaluation
```

All trained responses use one final-answer contract:

```text
Answer: <final answer>
```

The sandbox and reward code are built around that line. Correct final answers
score as correct; wrong, malformed, or missing final answers do not.

## Layout

```text
gen_data.py                         # question -> verified SFT message rows
prompts/solve_with_code.txt         # SFT generation prompt template
data/sft/train.jsonl                # SFT train set, JoeYing/ReTool-SFT + local rows
data/rl/math_l1_l3/                 # RL MATH level 1-3 prompt and metadata
retool_sandbox/                     # async Python sandbox, agent loop, reward
configs/retool_sandbox_agent.yaml   # veRL agent-loop registration
scripts/run_verl_sft.sh             # veRL SFT launcher
scripts/run_dapo_smoke.sh           # parameterized DAPO/GRPO-style RL launcher
scripts/infer_hf.py                 # plain HF generation
scripts/infer_hf_with_sandbox.py    # HF generation with real sandbox execution
scripts/eval_aime2024_models.py     # base/SFT/RL AIME-style comparison
scripts/prepare_*.py                # data conversion/export helpers
docs/checkpoints.md                 # published checkpoint restore details
tests/                              # reward and sandbox regression tests
```

Do not commit `.env`, logs, `runs/`, `wandb/`, or temporary release assets.

## Setup

```bash
conda create -n retool python=3.11 -y
conda activate retool
pip install -r requirements.txt
cp .env.example .env
```

Set these in `.env` for SFT data generation:

```bash
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
GEN_MODEL=deepseek-v4-pro
```

## Generate SFT Rows

Generate one verified ReTool-style example:

```bash
set -a; source .env; set +a
python gen_data.py \
  --question "What is the sum of all integers from 1 to 100?" \
  --out data/sft/generated.jsonl \
  --model "$GEN_MODEL"
```

Each row is a two-message JSON array:

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

For batch generation, pass a JSONL file with a required `question` field:

```bash
python gen_data.py --in questions.jsonl --out data/sft/generated.jsonl --concurrency 4
```

`gen_data.py` validates generated code by default: it executes Python snippets,
materializes real `<interpreter>...</interpreter>` output, and requires a final
`Answer: ...` line before writing to the main output.

## Sandbox Inference

Plain model generation is not tool use. Use the sandbox path when you need real
ReTool execution:

```bash
python scripts/infer_hf_with_sandbox.py \
  --model /path/to/hf-model-or-checkpoint \
  --question "Compute 123456789123456789 * 987654321987654321." \
  --max-new-tokens 4096 \
  --step-max-new-tokens 1024 \
  --max-tool-calls 4 \
  --max-model-calls 8
```

The rollout loop is:

```text
model generates until </code>
-> sandbox executes the Python block
-> real <interpreter>stdout</interpreter> is appended
-> model continues until Answer: ...
```

Run the local sandbox demo with:

```bash
python examples/sandbox_rollout_demo.py
```

## Training

SFT on the remote veRL host:

```bash
cd /root/autodl-tmp/retool
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero

DATA_JSONL=/root/autodl-tmp/retool/data/sft/train.jsonl \
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-3B \
bash scripts/run_verl_sft.sh
```

Prepare the default RL MATH level 1-3 prompt dataset:

```bash
python scripts/prepare_math_level_data.py
```

This writes generated parquet files under `data/rl/math_l1_l3/`; those files are
ignored, while the prompt template and metadata are committed.

DAPO/GRPO-style RL uses the parameterized launcher:

```bash
cd /root/autodl-tmp/retool
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero

export MODE=lora
export LORA_RANK=64
export MODEL_PATH=/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf
export TRAIN_FILE=/root/autodl-tmp/retool/data/rl/math_l1_l3/train.parquet
export VAL_FILE=/root/autodl-tmp/retool/data/rl/math_l1_l3/val.parquet
export USE_RETOOL_SANDBOX=True
export ROLLOUT_N=8
export MAX_RESPONSE_LENGTH=2048
export SAVE_FREQ=50

mkdir -p logs
nohup bash scripts/run_dapo_smoke.sh \
  reward.custom_reward_function.path=/root/autodl-tmp/retool/retool_sandbox/math_reward.py \
  reward.custom_reward_function.name=compute_score \
  > logs/${EXPERIMENT_NAME:-retool-dapo}.log 2>&1 &
```

Before launching a full RL run, verify train/val split hygiene, reward schema
stability, sandbox execution, checkpoint retention, and rollout dumps. Do not
judge a run by process liveness or W&B initialization alone.

## Checkpoints

The published inference-ready RL checkpoint is:

```text
retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf
```

It is available as split tar assets on the GitHub release
`retool-rl-gs200-fused-hf`. Restore instructions and checksums are in
[`docs/checkpoints.md`](docs/checkpoints.md).

Use the merged/fused HF directory for normal inference. Raw veRL/FSDP checkpoint
directories under `runs/dapo/.../global_step_*` are not standard Transformers
models until merged.

## Evaluation

Compare base, SFT, and RL checkpoints on the same held-out set:

```bash
python scripts/eval_aime2024_models.py \
  --bench bench/aime2024/aime-2024.parquet \
  --limit 30 \
  --out-dir runs/eval_outputs/aime2024_30_three_models \
  --max-new-tokens 4096 \
  --step-max-new-tokens 1024 \
  --max-model-calls 8 \
  --max-tool-calls 4 \
  --dtype bf16 \
  --device-map cuda:0
```

For multi-GPU evaluation, run one process per checkpoint with `--only base`,
`--only sft`, or `--only rl`, and set `CUDA_VISIBLE_DEVICES` per process.

## Development Checks

```bash
python -m py_compile gen_data.py retool_sandbox/*.py scripts/*.py tests/*.py
python -m pytest tests
```
