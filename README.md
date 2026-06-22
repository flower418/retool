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

## Validation

By default, `gen_data.py`:

- requires at least one `<code>...</code>` block;
- executes generated Python locally;
- replaces or inserts `<interpreter>...</interpreter>` with real stdout;
- requires exactly one final `<answer>...</answer>` block at the end;
- checks that numeric boxed answers match the final interpreter output.

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
