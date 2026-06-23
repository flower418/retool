"""veRL agent loop that executes ReTool-style Python code blocks."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op

from .async_python import AsyncPythonSandboxPool, SandboxLimits
from .retool import find_next_unexecuted_code, format_interpreter_block


ANSWER_RE = re.compile(r"(?i)\*{0,2}\s*Answer\s*:\s*\*{0,2}\s*\S[^\n]*")
ANSWER_TRAILING_PUNCT_RE = re.compile(r"(?i)((?:\*{0,2}\s*)?Answer\s*:\s*[^\n]*?)[.。．]+(\s*)$")
ANSWER_WITHOUT_QUOTES_RE = re.compile(
    r"(?i)((?:\*{0,2}\s*)?Answer\s*:\s*[^\n]*?)(?:[.。．]?\s*)\(\s*without\s+quotes\s*\)(\s*)$"
)
PROTECTED_BLOCK_RE = re.compile(
    r"<(?:code|interpreter)>.*?</(?:code|interpreter)>|```(?:python|py)?\s*\n.*?```",
    re.DOTALL | re.IGNORECASE,
)

RETOOL_PROMPT_PREFIX = """You have access to a Python sandbox during generation.
Follow this protocol strictly:
1. Before any prose, write one short self-contained Python block that computes
   or verifies the final answer. The block must include all needed imports and
   variables, because each sandbox call runs independently.
2. The final printed line of the block must be only the final answer value you
   plan to put after `Answer:`.
3. Treat the interpreter output as the source of truth. If the code errors,
   write a corrected short Python block.
4. Keep the solution concise after the sandbox result.
5. After any successful interpreter output, do not write more code; immediately
   write the final answer line.
6. End with exactly one final line in this format: Answer: <final answer>
   Do not use <answer> tags, boxes, or any text after the final answer line.

Use exactly this code format:
<code>
```python
print(...)
```
</code>

The sandbox will append:
<interpreter>output</interpreter>
"""


_SANDBOX_POOL: AsyncPythonSandboxPool | None = None
_SANDBOX_LOCK: asyncio.Lock | None = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _get_sandbox_pool() -> AsyncPythonSandboxPool:
    global _SANDBOX_LOCK, _SANDBOX_POOL
    if _SANDBOX_LOCK is None:
        _SANDBOX_LOCK = asyncio.Lock()
    async with _SANDBOX_LOCK:
        if _SANDBOX_POOL is None:
            _SANDBOX_POOL = AsyncPythonSandboxPool(
                num_workers=_env_int("RETOOL_SANDBOX_WORKERS", 2),
                limits=SandboxLimits(
                    timeout_s=_env_float("RETOOL_SANDBOX_TIMEOUT", 2.0),
                    max_output_bytes=_env_int("RETOOL_SANDBOX_OUTPUT_BYTES", 4000),
                    memory_mb=_env_int("RETOOL_SANDBOX_MEMORY_MB", 512),
                    file_mb=_env_int("RETOOL_SANDBOX_FILE_MB", 8),
                ),
                isolated=_env_bool("RETOOL_SANDBOX_ISOLATED", True),
                no_site=_env_bool("RETOOL_SANDBOX_NO_SITE", False),
            )
            await _SANDBOX_POOL.start()
        return _SANDBOX_POOL


def _inject_retool_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = [dict(message) for message in messages]
    for message in updated:
        if message.get("role") == "user":
            message["content"] = f"{RETOOL_PROMPT_PREFIX}\n\nProblem:\n{message.get('content', '')}"
            return updated
    return [{"role": "user", "content": RETOOL_PROMPT_PREFIX}, *updated]


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _find_answer_end(text: str) -> int | None:
    protected_spans = [match.span() for match in PROTECTED_BLOCK_RE.finditer(text)]
    for match in ANSWER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in protected_spans):
            continue
        line_end = text.find("\n", match.end())
        return len(text) if line_end < 0 else line_end + 1
    return None


def _clean_answer_text(text: str) -> str:
    text = ANSWER_WITHOUT_QUOTES_RE.sub(r"\1\2", text)
    return ANSWER_TRAILING_PUNCT_RE.sub(r"\1\2", text)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return repr(value)


def _maybe_dump_rollout(kwargs: dict[str, Any], generated_text: str, tool_results: list[Any]) -> None:
    dump_dir = os.getenv("RETOOL_DUMP_ROLLOUTS_DIR")
    if not dump_dir:
        return

    max_chars = _env_int("RETOOL_DUMP_MAX_CHARS", 20000)
    record = {
        "pid": os.getpid(),
        "raw_prompt": _jsonable(kwargs.get("raw_prompt")),
        "data_source": _jsonable(kwargs.get("data_source")),
        "reward_model": _jsonable(kwargs.get("reward_model")),
        "extra_info": _jsonable(kwargs.get("extra_info")),
        "index": _jsonable(kwargs.get("index")),
        "generated_text": generated_text[-max_chars:],
        "generated_text_truncated": len(generated_text) > max_chars,
        "tool_results": [
            {
                "ok": result.ok,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
                "timed_out": result.timed_out,
                "elapsed_ms": result.elapsed_ms,
            }
            for result in tool_results
        ],
    }
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"rollouts-{os.getpid()}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@register("retool_sandbox_agent")
class ReToolSandboxAgentLoop(AgentLoopBase):
    """Pause on generated Python code blocks, execute them, then continue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_calls = _env_int("RETOOL_MAX_MODEL_CALLS", 6)
        self.max_tool_calls = _env_int("RETOOL_MAX_TOOL_CALLS", 3)
        self.step_max_tokens = _env_int("RETOOL_STEP_MAX_TOKENS", min(768, self.response_length))
        self.stop_strings = ["</code>"]

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = _inject_retool_instruction(list(kwargs["raw_prompt"]))

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        current_prompt_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        generated_text = ""
        tool_results = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "compute_score": 0.0, "num_preempted": -1}
        extra_fields: dict[str, Any] = {}
        request_id = uuid4().hex
        routed_experts = None

        for _ in range(self.max_model_calls):
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break

            max_tokens = min(self.step_max_tokens, remaining)
            turn_sampling_params = {
                **sampling_params,
                "max_tokens": max_tokens,
                "stop": self.stop_strings,
                "include_stop_str_in_output": True,
            }

            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            metrics["generate_sequences"] += time.perf_counter() - started
            if output.num_preempted is not None:
                metrics["num_preempted"] = max(metrics["num_preempted"], output.num_preempted)
            if not extra_fields:
                extra_fields.update(output.extra_fields)
            else:
                for key, value in output.extra_fields.items():
                    extra_fields.setdefault(key, value)
            if output.routed_experts is not None:
                routed_experts = output.routed_experts

            token_ids = list(output.token_ids or [])
            if not token_ids:
                break
            logprobs = list(output.log_probs or [])
            if logprobs and len(logprobs) < len(token_ids):
                logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))

            current_prompt_ids.extend(token_ids)
            response_ids.extend(token_ids)
            response_mask.extend([1] * len(token_ids))
            if logprobs:
                response_logprobs.extend(logprobs[: len(token_ids)])
            generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            answer_end = _find_answer_end(generated_text)
            if answer_end is not None:
                answer_text = _clean_answer_text(generated_text[:answer_end])
                cleaned_response_ids = _encode_text(self.tokenizer, answer_text)
                keep = min(len(response_ids), len(cleaned_response_ids))
                response_ids = cleaned_response_ids
                response_mask = response_mask[:keep] + [1] * max(0, len(cleaned_response_ids) - keep)
                if response_logprobs:
                    response_logprobs = response_logprobs[:keep] + [0.0] * max(0, len(cleaned_response_ids) - keep)
                generated_text = answer_text
                break

            executed_this_turn = False
            while len(tool_results) < self.max_tool_calls:
                block = find_next_unexecuted_code(generated_text)
                if block is None:
                    break
                sandbox = await _get_sandbox_pool()
                started = time.perf_counter()
                result = await sandbox.run(block.code)
                metrics["tool_calls"] += time.perf_counter() - started
                tool_results.append(result)

                interpreter_text = format_interpreter_block(result)
                interpreter_ids = _encode_text(self.tokenizer, interpreter_text)
                room = self.response_length - len(response_ids)
                if room <= 0:
                    break
                if len(interpreter_ids) > room:
                    interpreter_ids = interpreter_ids[:room]
                    interpreter_text = self.tokenizer.decode(interpreter_ids, skip_special_tokens=True)

                response_ids.extend(interpreter_ids)
                response_mask.extend([0] * len(interpreter_ids))
                if response_logprobs:
                    response_logprobs.extend([0.0] * len(interpreter_ids))
                current_prompt_ids.extend(interpreter_ids)
                generated_text += interpreter_text
                executed_this_turn = True

            if executed_this_turn:
                continue
            if len(token_ids) < max_tokens:
                break

        if response_logprobs and len(response_logprobs) < len(response_ids):
            response_logprobs.extend([0.0] * (len(response_ids) - len(response_logprobs)))

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2 + len(tool_results),
            metrics=metrics,
            extra_fields=extra_fields,
        )
        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "sandbox_tool_calls": len(tool_results),
                "sandbox_timeouts": sum(1 for result in tool_results if result.timed_out),
            }
        )
        _maybe_dump_rollout(kwargs, generated_text, tool_results)
        return output
