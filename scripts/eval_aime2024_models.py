#!/usr/bin/env python3
"""Evaluate base, SFT, and RL checkpoints on the same AIME 2024 questions."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retool_sandbox import AsyncPythonSandboxPool, ModelStep, SandboxLimits, rollout_with_sandbox
from retool_sandbox.math_reward import compute_score

_INFER_SPEC = importlib.util.spec_from_file_location(
    "infer_hf_with_sandbox",
    Path(__file__).with_name("infer_hf_with_sandbox.py"),
)
if _INFER_SPEC is None or _INFER_SPEC.loader is None:
    raise ImportError("Could not load infer_hf_with_sandbox.py")
_infer_hf_with_sandbox = importlib.util.module_from_spec(_INFER_SPEC)
_INFER_SPEC.loader.exec_module(_infer_hf_with_sandbox)

DEFAULT_PROMPT_TEMPLATE = _infer_hf_with_sandbox.DEFAULT_PROMPT_TEMPLATE
TextStopCriteria = _infer_hf_with_sandbox.TextStopCriteria
first_model_device = _infer_hf_with_sandbox.first_model_device
load_tokenizer = _infer_hf_with_sandbox.load_tokenizer
resolve_dtype = _infer_hf_with_sandbox.resolve_dtype
trim_at_rollout_boundary = _infer_hf_with_sandbox.trim_at_rollout_boundary


BASE_PROMPT_TEMPLATE = """Solve the following AIME math problem. Work carefully and compute the answer directly.

The final answer must be the last line of your response and must use exactly this format:
Answer: <final answer>

Problem:
{question}
"""


MODEL_DEFAULTS = {
    "base": "/root/autodl-tmp/models/Qwen2.5-3B",
    "sft": "/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf",
    "rl": "/root/autodl-tmp/retool/runs/merged/retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="bench/aime2024/aime-2024.parquet")
    parser.add_argument("--out-dir", default="runs/eval_outputs/aime2024_30_three_models")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--models", nargs="*", help="Optional name=path entries.")
    parser.add_argument("--only", nargs="*", choices=("base", "sft", "rl"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--step-max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--sandbox-workers", type=int, default=2)
    parser.add_argument("--sandbox-timeout", type=float, default=6.0)
    parser.add_argument("--sandbox-output-bytes", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def coerce_cell(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def load_questions(path: str, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    by_index: dict[int, dict[str, Any]] = {}
    fallback_index = 0
    for _, row in df.iterrows():
        extra_info = coerce_cell(row.get("extra_info")) or {}
        reward_model = coerce_cell(row.get("reward_model")) or {}
        prompt = coerce_cell(row.get("prompt"))
        raw_problem = extra_info.get("raw_problem")
        if raw_problem is None and isinstance(prompt, list) and prompt:
            raw_problem = str(prompt[0].get("content", ""))
        if raw_problem is None:
            raw_problem = str(prompt)
        index = extra_info.get("index")
        if index is None:
            index = fallback_index
            fallback_index += 1
        index = int(index)
        if index not in by_index:
            by_index[index] = {
                "index": index,
                "question": raw_problem,
                "ground_truth": str(reward_model.get("ground_truth", "")),
                "data_source": str(row.get("data_source", "math_dapo")),
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
    questions = [by_index[key] for key in sorted(by_index)]
    if limit:
        questions = questions[:limit]
    return questions


def parse_model_overrides(entries: list[str] | None) -> dict[str, str]:
    models = dict(MODEL_DEFAULTS)
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Model override must be name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        models[name.strip()] = path.strip()
    return models


class LoadedModel:
    def __init__(self, model_path: str, args: argparse.Namespace) -> None:
        self.model_path = model_path
        self.args = args
        self.tokenizer = load_tokenizer(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=resolve_dtype(args.dtype),
            device_map=args.device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = first_model_device(self.model)
        self.generation_config = self._generation_config()
        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_token_ids.append(im_end_id)
        self.eos_token_ids = sorted(set(token_id for token_id in eos_token_ids if token_id is not None))

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generation_config(self) -> GenerationConfig:
        config = copy.deepcopy(self.model.generation_config)
        config.do_sample = self.args.temperature > 0
        if self.args.temperature > 0:
            config.temperature = self.args.temperature
            config.top_p = self.args.top_p
        else:
            config.temperature = None
            config.top_p = None
            config.top_k = None
        return config

    def model_prefix(self, user_prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return user_prompt

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def generate_direct(self, user_prompt: str) -> str:
        transcript = self.model_prefix(user_prompt)
        encoded = self.encode(transcript)
        input_len = encoded["input_ids"].shape[-1]
        kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "generation_config": self.generation_config,
            "repetition_penalty": self.args.repetition_penalty,
        }
        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **kwargs)
        text = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return trim_at_first_answer_line(text)


class SandboxRunner:
    def __init__(self, loaded: LoadedModel, user_prompt: str, args: argparse.Namespace) -> None:
        self.loaded = loaded
        self.model_prefix = loaded.model_prefix(user_prompt)
        self.args = args
        self.generated_tokens = 0

    async def generate_until(self, transcript: str, stop_sequences) -> ModelStep:
        del stop_sequences
        remaining = self.args.max_new_tokens - self.generated_tokens
        if remaining <= 0:
            return ModelStep(text="", finished=True)
        encoded = self.loaded.encode(transcript)
        input_len = encoded["input_ids"].shape[-1]
        step_max_new_tokens = min(self.args.step_max_new_tokens, remaining)
        stop_criteria = TextStopCriteria(self.loaded.tokenizer, input_len, ("</code>",))
        kwargs = {
            "max_new_tokens": step_max_new_tokens,
            "pad_token_id": self.loaded.tokenizer.pad_token_id,
            "eos_token_id": self.loaded.eos_token_ids,
            "generation_config": self.loaded.generation_config,
            "repetition_penalty": self.args.repetition_penalty,
            "stopping_criteria": torch.nn.ModuleList(),  # placeholder replaced below
        }
        from transformers import StoppingCriteriaList

        kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])
        with torch.inference_mode():
            output_ids = await asyncio.to_thread(self.loaded.model.generate, **encoded, **kwargs)
        new_ids = output_ids[0][input_len:]
        self.generated_tokens += int(new_ids.shape[-1])
        text = self.loaded.tokenizer.decode(new_ids, skip_special_tokens=True)
        text, post_stop = trim_at_rollout_boundary(text)
        hit_step_cap = int(new_ids.shape[-1]) >= step_max_new_tokens
        exhausted_budget = self.generated_tokens >= self.args.max_new_tokens
        matched = post_stop or stop_criteria.matched
        finished = matched is None and (not hit_step_cap or exhausted_budget)
        stop_text = None if matched == "__code_block__" or post_stop is not None else matched
        return ModelStep(text=text, stop_text=stop_text, finished=finished)


def trim_at_first_answer_line(text: str) -> str:
    match = re.search(r"(?im)^\s*Answer\s*:\s*\S[^\n]*", text)
    if match is None:
        return text.strip()
    line_end = text.find("\n", match.end())
    end = len(text) if line_end < 0 else line_end + 1
    return text[:end].strip()


def existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            seen.add(int(json.loads(line)["index"]))
        except Exception:
            pass
    return seen


def score_record(question: dict[str, Any], text: str, meta: dict[str, Any]) -> dict[str, Any]:
    score = compute_score(
        question["data_source"],
        text,
        question["ground_truth"],
        question["extra_info"],
    )
    return {
        "index": question["index"],
        "question": question["question"],
        "ground_truth": question["ground_truth"],
        "generated_text": text,
        "score": score,
        **meta,
    }


async def evaluate_sandbox_model(
    loaded: LoadedModel,
    questions: list[dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    seen = existing_indices(output_path) if args.resume else set()
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        async with AsyncPythonSandboxPool(
            num_workers=args.sandbox_workers,
            limits=SandboxLimits(
                timeout_s=args.sandbox_timeout,
                max_output_bytes=args.sandbox_output_bytes,
            ),
            no_site=False,
        ) as sandbox:
            for pos, question in enumerate(questions, start=1):
                if question["index"] in seen:
                    continue
                prompt = DEFAULT_PROMPT_TEMPLATE.replace("{question}", question["question"])
                runner = SandboxRunner(loaded, prompt, args)
                started = time.perf_counter()
                result = await rollout_with_sandbox(
                    runner.model_prefix,
                    runner.generate_until,
                    sandbox,
                    max_model_calls=args.max_model_calls,
                    max_tool_calls=args.max_tool_calls,
                )
                elapsed = time.perf_counter() - started
                record = score_record(
                    question,
                    result.text.strip(),
                    {
                        "mode": "sandbox",
                        "stop_reason": result.stop_reason,
                        "model_calls": result.model_calls,
                        "tool_calls": len(result.tool_results),
                        "tool_timeouts": sum(1 for item in result.tool_results if item.timed_out),
                        "elapsed_s": elapsed,
                    },
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{pos}/{len(questions)}] index={question['index']} "
                    f"score={record['score']['score']} reason={record['score']['reason']} "
                    f"stop={result.stop_reason} tools={len(result.tool_results)} elapsed={elapsed:.1f}s",
                    flush=True,
                )


def evaluate_direct_model(
    loaded: LoadedModel,
    questions: list[dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    seen = existing_indices(output_path) if args.resume else set()
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for pos, question in enumerate(questions, start=1):
            if question["index"] in seen:
                continue
            prompt = BASE_PROMPT_TEMPLATE.replace("{question}", question["question"])
            started = time.perf_counter()
            text = loaded.generate_direct(prompt)
            elapsed = time.perf_counter() - started
            record = score_record(
                question,
                text,
                {
                    "mode": "direct",
                    "stop_reason": "direct",
                    "model_calls": 1,
                    "tool_calls": 0,
                    "tool_timeouts": 0,
                    "elapsed_s": elapsed,
                },
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{pos}/{len(questions)}] index={question['index']} "
                f"score={record['score']['score']} reason={record['score']['reason']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )


def summarize_file(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    correct = sum(1 for row in rows if float(row["score"]["score"]) == 1.0)
    return {
        "file": str(path),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "format_ok": sum(1 for row in rows if row["score"].get("format_ok")),
        "answer_extracted": sum(1 for row in rows if row["score"].get("answer_extracted")),
        "avg_tool_calls": sum(float(row.get("tool_calls", 0)) for row in rows) / total if total else 0.0,
        "reasons": reason_counts(rows),
    }


def reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["score"].get("reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


async def main_async() -> None:
    args = parse_args()
    models = parse_model_overrides(args.models)
    names = args.only or ["base", "sft", "rl"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.bench, args.limit)
    manifest = {
        "bench": args.bench,
        "limit": args.limit,
        "question_indices": [item["index"] for item in questions],
        "models": {name: models[name] for name in names},
        "base_prompt": "direct",
        "sft_rl_prompt": "sandbox_code",
        "max_new_tokens": args.max_new_tokens,
        "step_max_new_tokens": args.step_max_new_tokens,
        "temperature": args.temperature,
    }
    run_suffix = "_".join(names)
    manifest_path = out_dir / ("manifest.json" if args.only is None else f"manifest_{run_suffix}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    for name in names:
        model_path = models[name]
        output_path = out_dir / f"{name}.jsonl"
        print(f"=== evaluating {name}: {model_path} ===", flush=True)
        loaded = LoadedModel(model_path, args)
        try:
            if name == "base":
                evaluate_direct_model(loaded, questions, output_path, args)
            else:
                await evaluate_sandbox_model(loaded, questions, output_path, args)
        finally:
            loaded.close()
        summary = summarize_file(output_path)
        (out_dir / f"{name}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"=== summary {name} ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    all_summary = {name: summarize_file(out_dir / f"{name}.jsonl") for name in names}
    summary_path = out_dir / ("summary.json" if args.only is None else f"summary_{run_suffix}.json")
    summary_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== all summary ===")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
