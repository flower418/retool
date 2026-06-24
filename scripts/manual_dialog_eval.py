#!/usr/bin/env python3
"""Run a small qualitative ReTool dialog eval with real sandbox execution."""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import importlib.util
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, StoppingCriteriaList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retool_sandbox import AsyncPythonSandboxPool, ModelStep, SandboxLimits, rollout_with_sandbox


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


MODEL_DEFAULTS = {
    "sft": "/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf",
    "rl": "/root/autodl-tmp/retool/runs/merged/retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf",
}

ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(?P<answer>\S[^\n]*)$")


def digit_sum_count(limit: int, target: int) -> int:
    return sum(1 for value in range(1, limit + 1) if sum(map(int, str(value))) == target)


MANUAL_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "sum_1_to_100",
        "question": "What is the sum of all integers from 1 to 100?",
        "expected_answer": "5050",
        "purpose": "format and first sandbox-call sanity check",
    },
    {
        "id": "stairs_6_steps_123",
        "question": (
            "A person climbs a staircase with 6 steps. Each move climbs 1, 2, "
            "or 3 steps. In how many different ways can the person reach the top?"
        ),
        "expected_answer": "24",
        "purpose": "small dynamic-programming combinatorics",
    },
    {
        "id": "mod_17017_power",
        "question": "Find the remainder when 17017^17 is divided by 12.",
        "expected_answer": "1",
        "purpose": "modular arithmetic, catches over-computation and answer formatting",
    },
    {
        "id": "square_divisors",
        "question": (
            "How many positive divisors of 2^6 * 3^4 * 5^2 are perfect squares?"
        ),
        "expected_answer": "24",
        "purpose": "factor-count reasoning with an easily verified integer answer",
    },
    {
        "id": "ordered_gcd_pairs",
        "question": (
            "How many ordered pairs (a, b) with 1 <= a, b <= 30 satisfy gcd(a, b) = 6?"
        ),
        "expected_answer": "19",
        "purpose": "checks whether code is used to verify a counting argument",
    },
    {
        "id": "digit_sum_to_500",
        "question": "How many positive integers n <= 500 have digit sum 7?",
        "expected_answer": str(digit_sum_count(500, 7)),
        "purpose": "enumeration task where sandbox execution should be decisive",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="Optional name=path entries.")
    parser.add_argument("--only", nargs="+", default=["sft", "rl"], help="Model names to run.")
    parser.add_argument(
        "--out",
        help="JSONL output path. Defaults to runs/manual_eval/retool_manual_dialog_<time>.jsonl.",
    )
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
    parser.add_argument(
        "--sandbox-no-site",
        action="store_true",
        help="Run sandbox workers with Python -S. Default keeps site packages available.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing model/question pairs.")
    return parser.parse_args()


def parse_model_overrides(entries: list[str] | None) -> dict[str, str]:
    models = dict(MODEL_DEFAULTS)
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Model override must be name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        models[name.strip()] = path.strip()
    return models


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "manual_eval" / f"retool_manual_dialog_{stamp}.jsonl"


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.removeprefix("$").removesuffix("$").strip()
    text = text.removeprefix("\\(").removesuffix("\\)").strip()
    text = text.removeprefix("\\[").removesuffix("\\]").strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    return text.rstrip(".").strip()


def extract_final_answer(text: str) -> str | None:
    matches = list(ANSWER_LINE_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("answer").strip()


def score_answer(expected: str | None, generated_text: str) -> dict[str, Any]:
    final_answer = extract_final_answer(generated_text)
    if expected is None:
        return {
            "final_answer": final_answer,
            "expected_answer": None,
            "exact_match": None,
            "score": None,
        }
    exact_match = normalize_answer(final_answer) == normalize_answer(expected)
    return {
        "final_answer": final_answer,
        "expected_answer": expected,
        "exact_match": exact_match,
        "score": 1.0 if exact_match else 0.0,
    }


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((str(row.get("model_name")), str(row.get("question_id"))))
    return keys


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
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = args.temperature > 0
        if args.temperature > 0:
            self.generation_config.temperature = args.temperature
            self.generation_config.top_p = args.top_p
        else:
            self.generation_config.temperature = None
            self.generation_config.top_p = None
            self.generation_config.top_k = None

        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_token_ids.append(im_end_id)
        self.eos_token_ids = sorted(
            set(token_id for token_id in eos_token_ids if token_id is not None)
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        generation_kwargs = {
            "max_new_tokens": step_max_new_tokens,
            "pad_token_id": self.loaded.tokenizer.pad_token_id,
            "eos_token_id": self.loaded.eos_token_ids,
            "generation_config": self.loaded.generation_config,
            "repetition_penalty": self.args.repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([stop_criteria]),
        }

        with torch.inference_mode():
            output_ids = await asyncio.to_thread(
                self.loaded.model.generate,
                **encoded,
                **generation_kwargs,
            )
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


async def run_one_model(
    *,
    model_name: str,
    model_path: str,
    output_path: Path,
    skipped: set[tuple[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"=== loading {model_name}: {model_path} ===", flush=True)
    loaded = LoadedModel(model_path, args)
    summary = {"total": 0, "graded": 0, "correct": 0, "tool_calls": 0}
    try:
        async with AsyncPythonSandboxPool(
            num_workers=args.sandbox_workers,
            limits=SandboxLimits(
                timeout_s=args.sandbox_timeout,
                max_output_bytes=args.sandbox_output_bytes,
            ),
            no_site=args.sandbox_no_site,
        ) as sandbox:
            with output_path.open("a", encoding="utf-8") as handle:
                for pos, question in enumerate(MANUAL_QUESTIONS, start=1):
                    key = (model_name, question["id"])
                    if key in skipped:
                        print(f"[skip] {model_name}/{question['id']}", flush=True)
                        continue
                    user_prompt = DEFAULT_PROMPT_TEMPLATE.replace("{question}", question["question"])
                    runner = SandboxRunner(loaded, user_prompt, args)
                    started = time.perf_counter()
                    result = await rollout_with_sandbox(
                        runner.model_prefix,
                        runner.generate_until,
                        sandbox,
                        max_model_calls=args.max_model_calls,
                        max_tool_calls=args.max_tool_calls,
                    )
                    elapsed_s = time.perf_counter() - started
                    score = score_answer(question.get("expected_answer"), result.text.strip())
                    record = {
                        "model_name": model_name,
                        "model_path": model_path,
                        "question_index": pos,
                        "question_id": question["id"],
                        "purpose": question["purpose"],
                        "question": question["question"],
                        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                        "user_prompt": user_prompt,
                        "generated_text": result.text.strip(),
                        "transcript": result.transcript,
                        "stop_reason": result.stop_reason,
                        "model_calls": result.model_calls,
                        "tool_calls": len(result.tool_results),
                        "tool_results": [asdict(item) for item in result.tool_results],
                        "elapsed_s": elapsed_s,
                        **score,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    skipped.add(key)

                    summary["total"] += 1
                    summary["tool_calls"] += len(result.tool_results)
                    if score["score"] is not None:
                        summary["graded"] += 1
                        summary["correct"] += int(bool(score["exact_match"]))
                    print(
                        f"[{model_name} {pos}/{len(MANUAL_QUESTIONS)}] "
                        f"{question['id']} pred={score['final_answer']!r} "
                        f"expected={score['expected_answer']!r} match={score['exact_match']} "
                        f"stop={result.stop_reason} tools={len(result.tool_results)} "
                        f"elapsed={elapsed_s:.1f}s",
                        flush=True,
                    )
    finally:
        loaded.close()
    return summary


async def main_async() -> None:
    args = parse_args()
    models = parse_model_overrides(args.models)
    missing = [name for name in args.only if name not in models]
    if missing:
        raise SystemExit(f"Unknown model name(s): {', '.join(missing)}")

    output_path = Path(args.out) if args.out else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")
    summary_path = output_path.with_suffix(".summary.json")

    selected_models = {name: models[name] for name in args.only}
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "models": selected_models,
        "questions": MANUAL_QUESTIONS,
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "step_max_new_tokens": args.step_max_new_tokens,
            "max_model_calls": args.max_model_calls,
            "max_tool_calls": args.max_tool_calls,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "sandbox_workers": args.sandbox_workers,
            "sandbox_timeout": args.sandbox_timeout,
            "sandbox_no_site": args.sandbox_no_site,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.resume:
        output_path.write_text("", encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    print(f"output:   {output_path}", flush=True)

    skipped = existing_keys(output_path) if args.resume else set()
    all_summary = {}
    for model_name, model_path in selected_models.items():
        all_summary[model_name] = await run_one_model(
            model_name=model_name,
            model_path=model_path,
            output_path=output_path,
            skipped=skipped,
            args=args,
        )

    for item in all_summary.values():
        if item["graded"]:
            item["accuracy"] = item["correct"] / item["graded"]
        else:
            item["accuracy"] = None
        if item["total"]:
            item["avg_tool_calls"] = item["tool_calls"] / item["total"]
        else:
            item["avg_tool_calls"] = None

    summary_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== summary ===")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False), flush=True)
    print(f"summary:  {summary_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
