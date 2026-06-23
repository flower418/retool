# retool - direct SFT data generation

This repository generates ReTool-style supervised fine-tuning data directly
from math/problem questions. It calls an OpenAI-compatible API, asks the model
to produce code-assisted reasoning, executes generated Python snippets locally,
and writes verified SFT conversations.

The included merged dataset is based on
[`JoeYing/ReTool-SFT`](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
plus two newly generated examples.

## Files

```text
gen_data.py                  # question -> verified SFT messages generator
prompts/solve_with_code.txt  # user prompt template with {question}
sft_train.jsonl              # two locally generated examples
retool_sft_merged.jsonl      # 2000 HF rows + the 2 local examples
requirements.txt             # Python dependencies
.env.example                 # API configuration template
scripts/infer_hf.py          # run a merged HF checkpoint for inference
scripts/run_verl_sft.sh      # train with verl SFT on the prepared dataset
scripts/prepare_verl_sft_data.py  # convert JSONL data to verl parquet
scripts/prepare_math_level_data.py # example dataset filter/export helper
scripts/run_dapo_smoke.sh     # parameterized veRL RL launcher used for GRPO/DAPO-style runs
retool_sandbox/              # async Python sandbox, veRL agent loop, reward helpers
configs/retool_sandbox_agent.yaml # veRL agent-loop registration config
```

Do not commit `.env`; it is ignored by Git.

## Setup

```bash
git clone <repo-url>
cd retool
conda create -n retool python=3.11 -y
conda activate retool
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
GEN_MODEL=deepseek-v4-pro
```

Any OpenAI-compatible endpoint can be used by changing `OPENAI_BASE_URL` and
`GEN_MODEL`.

## Generate One Example

```bash
python gen_data.py \
  --question 'What is the sum of all integers from 1 to 100?' \
  --out sft_train.jsonl \
  --model "$GEN_MODEL" \
  --max-tokens 8192
```

Each output row in `sft_train.jsonl` is a two-message JSON array:

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

The script also writes a sidecar metadata file such as
`sft_train.jsonl.meta.jsonl`; this is ignored by Git.

## Batch Generation

Create any JSONL file with a `question` field:

```json
{"id": 0, "question": "A real number x satisfies ..."}
```

Run:

```bash
python gen_data.py --in questions.jsonl --out sft_train.jsonl --concurrency 4
```

## Async Rollout Sandbox

For RL rollouts, use `retool_sandbox` instead of `gen_data.py`. It implements
the online loop:

```text
model generates until </code>
-> sandbox executes the Python block asynchronously
-> <interpreter>stdout</interpreter> is appended to the transcript
-> model continues generation
```

Minimal usage:

```python
from retool_sandbox import AsyncPythonSandboxPool, SandboxLimits, rollout_with_sandbox

async with AsyncPythonSandboxPool(
    num_workers=32,
    limits=SandboxLimits(timeout_s=2.0, max_output_bytes=20000),
) as sandbox:
    result = await rollout_with_sandbox(prompt, generate_until, sandbox)
```

`generate_until(transcript, stop_sequences)` is the adapter you provide for
vLLM, Transformers, or a remote rollout server. It should generate the next
assistant chunk with stop strings such as `</code>` and then end with a final
`Answer: <final answer>` line.

Run the local demo:

```bash
python examples/sandbox_rollout_demo.py
```

Run HF inference with online sandbox execution:

```bash
python scripts/infer_hf_with_sandbox.py \
  --model /path/to/hf-model-or-ckpt \
  --question "Compute 123456789123456789 * 987654321987654321." \
  --max-tokens 1024
```

On the training server, `--model` can be omitted when `MODEL_PATH` is set or
when the default local SFT checkpoint exists. The script uses an embedded ReTool
prompt by default; pass `--prompt-template prompts/solve_with_code.txt` only if
you want to force the dataset-generation prompt.

The sandbox uses persistent worker processes, so hot execution avoids spawning a
new Python interpreter per code block. Timeouts, output caps, per-run temporary
directories, and memory/file limits are applied per worker. This is intended for
model-generated math/code snippets during rollout, not as a hardened security
boundary for hostile code.

## RL Experiment Workflow

Use experiments as a spec, not as a hard-coded script. A natural-language request
should be normalized into these slots before launch:

```yaml
algorithm: dapo | grpo | ppo | custom
base_model: /path/to/hf-model-or-merged-ckpt
dataset:
  train: /path/to/train.parquet
  val: /path/to/val.parquet
  bench_only: optional held-out sets that must not leak into train
reward:
  function: module path and function name
  scale: e.g. +1/-1
  parser: final-answer extraction and normalization rules
tool_loop:
  enabled: true | false
  max_tool_calls: 3
  max_model_calls: 5
rollout:
  n: 8
  prompt_length: 1024
  response_length: 2048
batch:
  train_batch_size: 2
  ppo_mini_batch_size: 1
checkpointing:
  save_freq: 50
  keep: 1
logging:
  project: retool-dapo
  run_name: descriptive-sortable-name
```

When asking an agent to run an experiment, phrase it at this level:

```text
Use <algorithm> on <train data>, keep <bench data> only for eval,
reward is <definition>, sandbox <on/off>, rollout=<n>,
response=<tokens>, save every <steps>, keep <k> ckpts.
```

The agent should then produce a run card, verify data/reward/sandbox, launch the
job, and report PID, log path, checkpoint path, rollout dump path, and W&B link.

## veRL RL Launch Template

The current RL launcher is `scripts/run_dapo_smoke.sh`. Despite the historical
name, it is parameterized through environment variables and Hydra overrides.
Use it for GRPO/DAPO-style veRL runs when the requested algorithm matches that
trainer path. For PPO or a different trainer, inspect the installed veRL config
and adapt the algorithm-specific overrides instead of pretending this script is
universal.

Example command shape on a remote training host:

```bash
cd /root/autodl-tmp/retool
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero

RUN_NAME="retool-<algo>-<data>-<key-knobs>-$(date +%Y%m%d_%H%M%S)"
export VERL_ROOT=/root/verl
export RETOOL_REPO_DIR=/root/autodl-tmp/retool
export MODE=lora                 # or full, depending on the run card
export LORA_RANK=64              # ignored unless MODE=lora
export PROJECT_NAME=retool-dapo
export EXPERIMENT_NAME="$RUN_NAME"
export LOGGER='["console","wandb"]'
export MODEL_PATH=/path/to/hf-model-or-merged-ckpt
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/val.parquet
export CKPTS_DIR="/root/autodl-tmp/retool/runs/dapo/${RUN_NAME}"
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=2048
export TRAIN_BATCH_SIZE=2
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=1
export TOTAL_STEPS=1000
export SAVE_FREQ=50
export MAX_ACTOR_CKPT_TO_KEEP=1
export MAX_CRITIC_CKPT_TO_KEEP=1

# Enable only when the experiment needs real code execution during rollout.
export USE_RETOOL_SANDBOX=True
export RETOOL_MAX_TOOL_CALLS=3
export RETOOL_MAX_MODEL_CALLS=5
export RETOOL_SANDBOX_NO_SITE=0
export RETOOL_DUMP_ROLLOUTS_DIR="/root/autodl-tmp/retool/debug_rollouts/${RUN_NAME}"

nohup bash scripts/run_dapo_smoke.sh \
  reward.custom_reward_function.path=/root/autodl-tmp/retool/retool_sandbox/math_reward.py \
  reward.custom_reward_function.name=compute_score \
  > "/root/autodl-tmp/retool/logs/${RUN_NAME}.log" 2>&1 &
```

Before a full run, verify:

- the train and val files exist and do not include bench-only data;
- one raw row has the expected `prompt`, `reward_model.ground_truth`, and
  `extra_info` fields;
- the reward function returns the same keys for correct, wrong, malformed, and
  formatting-edge examples;
- the sandbox executes real code if `USE_RETOOL_SANDBOX=True`;
- `SAVE_FREQ` and checkpoint retention fit the disk budget.

## RL Debugging Checklist

Do not judge a run by process liveness or W&B initialization alone. Watch these
phases:

```text
config compose -> dataset load -> model/FSDP/vLLM load -> W&B init
-> rollout generation -> sandbox tool calls -> reward computation
-> old log-prob -> actor update -> checkpoint save
```

Common failure modes:

- **Benchmark leakage**: a held-out set accidentally appears in train or val.
- **Prompt pollution**: answer-format text such as `(without quotes)` becomes
  part of the model target and causes reward mismatches.
- **Fake tool use**: plain inference generates `<interpreter>` text, but no
  sandbox actually ran.
- **Reward schema crash**: veRL can fail with `reward_extra_keys` if custom
  reward branches return different dict keys.
- **Constant reward**: all `+1`, all `-1`, or all `0` gives no useful
  within-group learning signal; inspect rollout text before changing LR.
- **Extractor mismatch**: visible answers such as Markdown, LaTeX wrappers, or
  unit suffixes may be scored wrong if the parser is too narrow.
- **Missing final answer**: can be true truncation, a code-loop consuming the
  response budget, or an extractor that only accepts one answer format.
- **Stale worker code**: after editing reward or agent-loop code, restart the
  trainer/Ray workers; running workers may keep old imports.
- **Raw checkpoint confusion**: veRL/FSDP shards are not HF models. Merge before
  running normal HF inference or evaluation.

## Validation

By default, `gen_data.py`:

- requires at least one `<code>...</code>` block;
- executes generated Python locally;
- replaces or inserts `<interpreter>...</interpreter>` with real stdout;
- requires exactly one final `Answer: <final answer>` line at the end;
- checks that numeric final answers match the final interpreter output.

Rows that fail validation are not written to the main SFT file.

## Data Format

`retool_sft_merged.jsonl` uses Hugging Face-style rows:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

It currently contains 2002 rows: the original 2000 rows from
`JoeYing/ReTool-SFT` and 2 generated examples from this repository.

## Run a Fine-Tuned Checkpoint

Use the merged Hugging Face checkpoint, not the raw verl/FSDP shard directory.
On the training server, the latest merged model is typically under
`/root/autodl-tmp/retool/runs/merged/...-hf`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero
cd /root/autodl-tmp/retool

python scripts/infer_hf.py \
  --model /root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf \
  --question 'Let N be the number of ordered pairs of positive integers (a, b) such that a + b + ab <= 2026 and gcd(a, b) = 1. Find the remainder when N is divided by 1000.' \
  --max-new-tokens 2048
```

Run the original base model by changing `--model`:

```bash
python scripts/infer_hf.py \
  --model /root/autodl-tmp/models/Qwen2.5-3B \
  --question 'Your problem here' \
  --max-new-tokens 2048
```
