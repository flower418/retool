# Repository Guidelines

## Project Structure & Module Organization

This repository implements direct SFT data synthesis for ReTool-style training. The main entry point is `gen_data.py`, which reads question-only JSONL examples, calls an OpenAI-compatible chat API, and writes two-message SFT conversations.

- `gen_data.py`: async question-to-SFT generator with retry and resume support.
- `prompts/solve_with_code.txt`: user prompt template used by `--prompt`.
- `data/sft/train.jsonl`: Hugging Face-style SFT train rows with `{"messages": ...}`.
- `data/rl/math_l1_l3/prompt.txt`: RL prompt template for Hendrycks MATH levels 1-3.
- `data/rl/math_l1_l3/meta.json`: source/count metadata for the generated RL parquet files.
- Optional batch inputs can be supplied as external JSONL files with a required `question` field.
- `.env.example`: documented environment variables. Keep local secrets in `.env`.
- `requirements.txt`: Python runtime dependencies.

## Build, Test, and Development Commands

Set up a local environment:

```bash
conda create -n retool python=3.11 -y
conda activate retool
pip install -r requirements.txt
```

Configure credentials:

```bash
cp .env.example .env
set -a; source .env; set +a
```

Run a small generation job:

```bash
python gen_data.py --question "What is the sum of all integers from 1 to 100?" --out data/sft/generated.jsonl --model "$GEN_MODEL"
```

Check syntax quickly:

```bash
python -m py_compile gen_data.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and descriptive snake_case names for functions, variables, and CLI arguments. Keep functions small and focused on one responsibility: input loading, prompt construction, API calls, validation, or output writing. Prefer `pathlib.Path` for filesystem operations when touching paths. Preserve the SFT output schema: each JSONL row must be a list containing one `user` message and one `assistant` message.

## Testing Guidelines

Run `python -m py_compile gen_data.py retool_sandbox/*.py scripts/*.py tests/*.py` for syntax checks and `python -m pytest tests` for the sandbox/reward regression suite. For changes to generation logic, also run a one-question smoke test with a temporary output path. Verify that output rows are two-message arrays and that metadata records `model`, `finish_reason`, and any `validation_issues`.

## Commit & Pull Request Guidelines

This checkout has no Git history available, so no repository-specific commit convention can be inferred. Use concise imperative commit subjects, for example `Add execution validation for SFT rows`. Pull requests should describe the data or API behavior changed, include the exact command used for validation, and note whether generated JSONL files are examples or required artifacts.

## Security & Configuration Tips

Do not commit `.env` or API keys; `.gitignore` already excludes them. Prefer environment variables for `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `GEN_MODEL`. When sharing outputs, review generated reasoning traces for sensitive prompt or dataset content.
