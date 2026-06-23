#!/usr/bin/env python3
"""Run interactive ReTool inference with local sandbox execution."""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retool_sandbox import (
    AsyncPythonSandboxPool,
    ModelStep,
    SandboxLimits,
    find_next_unexecuted_code,
    rollout_with_sandbox,
)


DEFAULT_PROMPT_TEMPLATE = """Solve the following problem. You must use the Python sandbox at least once.

Write a complete, self-contained Python script in this exact format:
<code>
```python
result = ...
print(result)
```
</code>

Each sandbox call runs independently, so include all needed imports and
variables in every code block. The final printed line must be only the final
answer value.

The sandbox will execute the code and append:
<interpreter>output</interpreter>

Use the interpreter output as ground truth. After any successful interpreter
output, do not write another code block; immediately write the final answer.

The final answer must be the last line of your response and must use exactly
this format:
Answer: <final answer>

Problem:
{question}
"""


DEFAULT_MODEL_CANDIDATES = (
    "/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf",
    "/root/autodl-tmp/models/Qwen2.5-3B",
)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer\s*:\s*\S[^\n]*")


def resolve_default_model() -> str | None:
    for key in ("MODEL_PATH", "GEN_MODEL"):
        value = os.environ.get(key)
        if value:
            return value
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ReTool response while executing <code> blocks."
    )
    parser.add_argument("question_text", nargs="?", help="Problem text to solve.")
    parser.add_argument(
        "--model",
        default=resolve_default_model(),
        help=(
            "HF model directory or model id. Defaults to MODEL_PATH, GEN_MODEL, "
            "then the local SFT ckpt if present."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--question", help="Problem text to solve.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-new-tokens", "--max-tokens", type=int, default=2048)
    parser.add_argument("--step-max-new-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-model-calls", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--max-tool-calls", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Model dtype. auto uses bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sandbox-workers", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--sandbox-timeout", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--sandbox-output-bytes", type=int, default=20000, help=argparse.SUPPRESS)
    parser.add_argument("--print-stats", action="store_true")
    args = parser.parse_args()
    args.question = args.question or args.question_text
    if not args.question:
        parser.error("provide a question as a positional argument or with --question")
    if not args.model:
        parser.error("provide --model or set MODEL_PATH/GEN_MODEL")
    if args.step_max_new_tokens is None:
        args.step_max_new_tokens = min(args.max_new_tokens, 768)
    return args


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def first_model_device(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            text = str(device)
            if text.startswith("cuda") or text.isdigit():
                return torch.device(f"cuda:{text}" if text.isdigit() else text)
    return next(model.parameters()).device


def load_tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def trim_at_rollout_boundary(text: str) -> tuple[str, str | None]:
    match = ANSWER_LINE_RE.search(text)
    if match is not None:
        line_end = text.find("\n", match.end())
        end = len(text) if line_end < 0 else line_end + 1
        return text[:end], "__answer__"

    block = find_next_unexecuted_code(text)
    if block is not None:
        return text[: block.end], "__code_block__"

    return text, None


class TextStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer, input_len: int, stop_sequences: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.stop_sequences = tuple(stop_sequences)
        self.matched: str | None = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][self.input_len :], skip_special_tokens=True)
        for stop in self.stop_sequences:
            if text.endswith(stop):
                self.matched = stop
                return True
        block = find_next_unexecuted_code(text)
        if block is not None and not text[block.end :].strip():
            self.matched = "__code_block__"
            return True
        return False


class HFSandboxGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.prompt_template:
            prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")
        else:
            prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.user_prompt = prompt_template.replace("{question}", args.question)
        self.args = args

        tokenizer_path = args.tokenizer or args.model
        self.tokenizer = load_tokenizer(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if args.no_chat_template:
            self.model_prefix = self.user_prompt
        else:
            self.model_prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.user_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
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

    def encode_transcript(self, transcript: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(transcript, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    async def generate_until(
        self,
        transcript: str,
        stop_sequences: Sequence[str],
    ) -> ModelStep:
        del stop_sequences
        remaining = self.args.max_new_tokens - self.generated_tokens
        if remaining <= 0:
            return ModelStep(text="", finished=True)

        encoded = self.encode_transcript(transcript)
        input_len = encoded["input_ids"].shape[-1]
        step_max_new_tokens = min(self.args.step_max_new_tokens, remaining)
        stop_criteria = TextStopCriteria(
            self.tokenizer,
            input_len,
            ("</code>",),
        )
        generation_kwargs = {
            "max_new_tokens": step_max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "generation_config": self.generation_config,
            "repetition_penalty": self.args.repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([stop_criteria]),
        }

        with torch.inference_mode():
            output_ids = await asyncio.to_thread(
                self.model.generate,
                **encoded,
                **generation_kwargs,
            )
        new_ids = output_ids[0][input_len:]
        self.generated_tokens += int(new_ids.shape[-1])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        text, post_stop = trim_at_rollout_boundary(text)
        hit_step_cap = int(new_ids.shape[-1]) >= step_max_new_tokens
        exhausted_budget = self.generated_tokens >= self.args.max_new_tokens
        matched = post_stop or stop_criteria.matched
        finished = (
            matched is None
            and (not hit_step_cap or exhausted_budget)
        )
        stop_text = (
            None
            if matched == "__code_block__" or post_stop is not None
            else matched
        )
        return ModelStep(text=text, stop_text=stop_text, finished=finished)


async def main_async(args: argparse.Namespace) -> None:
    generator = HFSandboxGenerator(args)
    generator.generated_tokens = 0
    async with AsyncPythonSandboxPool(
        num_workers=args.sandbox_workers,
        limits=SandboxLimits(
            timeout_s=args.sandbox_timeout,
            max_output_bytes=args.sandbox_output_bytes,
        ),
    ) as sandbox:
        result = await rollout_with_sandbox(
            generator.model_prefix,
            generator.generate_until,
            sandbox,
            max_model_calls=args.max_model_calls,
            max_tool_calls=args.max_tool_calls,
        )

    print(result.text.strip())
    if args.print_stats:
        print()
        print(
            f"[sandbox_stats] stop_reason={result.stop_reason} "
            f"model_calls={result.model_calls} tool_calls={len(result.tool_results)}"
        )
        for idx, tool_result in enumerate(result.tool_results, start=1):
            print(
                f"[sandbox_tool_{idx}] ok={tool_result.ok} "
                f"elapsed_ms={tool_result.elapsed_ms:.1f} "
                f"timed_out={tool_result.timed_out}"
            )


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
