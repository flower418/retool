"""ReTool tag parsing and async sandbox orchestration."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from .async_python import AsyncPythonSandboxPool, CodeExecutionResult


TAGGED_CODE_BLOCK_RE = re.compile(
    r"<code>\s*```(?:python|py)?\s*\n(?P<code>.*?)```\s*</code>",
    re.DOTALL,
)
FENCED_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)
INTERPRETER_RE = re.compile(r"\s*<interpreter>.*?</interpreter>", re.DOTALL)
ANSWER_DONE_RE = re.compile(r"(?im)^\s*Answer\s*:\s*\S[^\n]*")


@dataclass(frozen=True)
class CodeBlock:
    code: str
    start: int
    end: int


def _inside_tagged_code(content: str, position: int) -> bool:
    before = content[:position]
    return before.rfind("<code>") > before.rfind("</code>")


def _has_interpreter_after(content: str, end: int) -> bool:
    return INTERPRETER_RE.match(content[end:]) is not None


@dataclass(frozen=True)
class ModelStep:
    """One model continuation.

    `stop_text` should be the stop sequence that ended generation when the
    backend excludes stop strings from returned text, for example "</code>".
    """

    text: str
    stop_text: str | None = None
    finished: bool = False


@dataclass(frozen=True)
class RolloutResult:
    text: str
    transcript: str
    tool_results: list[CodeExecutionResult] = field(default_factory=list)
    model_calls: int = 0
    stop_reason: str = "unknown"


GenerateUntil = Callable[[str, Sequence[str]], Awaitable[str | ModelStep]]


def find_next_unexecuted_code(content: str) -> CodeBlock | None:
    """Return the first complete <code> block not followed by <interpreter>."""

    candidates: list[CodeBlock] = []
    for match in TAGGED_CODE_BLOCK_RE.finditer(content):
        if not _has_interpreter_after(content, match.end()):
            candidates.append(
                CodeBlock(
                    code=match.group("code"),
                    start=match.start(),
                    end=match.end(),
                )
            )

    for match in FENCED_CODE_BLOCK_RE.finditer(content):
        if _inside_tagged_code(content, match.start()):
            continue
        if not _has_interpreter_after(content, match.end()):
            candidates.append(
                CodeBlock(
                    code=match.group("code"),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return min(candidates, key=lambda block: block.start) if candidates else None


def format_interpreter_block(result: CodeExecutionResult) -> str:
    text = result.interpreter_text()
    text = text.replace("</interpreter>", "</ interpreter>")
    return f"\n<interpreter>{text}</interpreter>"


async def execute_next_unexecuted_code(
    content: str,
    sandbox: AsyncPythonSandboxPool,
) -> tuple[str, CodeExecutionResult | None]:
    block = find_next_unexecuted_code(content)
    if block is None:
        return content, None
    result = await sandbox.run(block.code)
    updated = content[: block.end] + format_interpreter_block(result) + content[block.end :]
    return updated, result


async def execute_all_pending_code(
    content: str,
    sandbox: AsyncPythonSandboxPool,
    *,
    max_calls: int = 8,
) -> tuple[str, list[CodeExecutionResult]]:
    results: list[CodeExecutionResult] = []
    for _ in range(max_calls):
        content, result = await execute_next_unexecuted_code(content, sandbox)
        if result is None:
            return content, results
        results.append(result)
    return content, results


def _normalize_step(raw: str | ModelStep) -> ModelStep:
    if isinstance(raw, ModelStep):
        return raw
    return ModelStep(text=raw, finished=True)


def _append_missing_stop_text(step: ModelStep) -> str:
    if step.stop_text and not step.text.endswith(step.stop_text):
        return step.text + step.stop_text
    return step.text


def _find_answer_end(text: str) -> int | None:
    match = ANSWER_DONE_RE.search(text)
    if match is None:
        return None
    line_end = text.find("\n", match.end())
    return len(text) if line_end < 0 else line_end + 1


async def rollout_with_sandbox(
    prompt: str,
    generate_until: GenerateUntil,
    sandbox: AsyncPythonSandboxPool,
    *,
    max_model_calls: int = 16,
    max_tool_calls: int = 8,
    stop_sequences: Sequence[str] = ("</code>",),
) -> RolloutResult:
    """Run a text rollout that pauses on code blocks and feeds back outputs.

    `generate_until` receives the current transcript and stop sequences. It
    should generate the next assistant continuation until one of the stop
    strings or EOS. When a completed code block appears, this function executes
    it and appends an `<interpreter>...</interpreter>` block before asking the
    model to continue.
    """

    transcript = prompt
    generated = ""
    tool_results: list[CodeExecutionResult] = []

    for model_calls in range(1, max_model_calls + 1):
        step = _normalize_step(await generate_until(transcript, stop_sequences))
        chunk = _append_missing_stop_text(step)
        generated += chunk
        transcript += chunk

        answer_end = _find_answer_end(generated)
        if answer_end is not None:
            generated = generated[:answer_end]
            transcript = prompt + generated
            return RolloutResult(
                text=generated,
                transcript=transcript,
                tool_results=tool_results,
                model_calls=model_calls,
                stop_reason="answer",
            )

        executed_this_call = False
        while len(tool_results) < max_tool_calls:
            generated, result = await execute_next_unexecuted_code(generated, sandbox)
            if result is None:
                break
            tool_results.append(result)
            transcript = prompt + generated
            executed_this_call = True

        if len(tool_results) >= max_tool_calls and find_next_unexecuted_code(generated):
            return RolloutResult(
                text=generated,
                transcript=transcript,
                tool_results=tool_results,
                model_calls=model_calls,
                stop_reason="max_tool_calls",
            )

        if step.finished and not executed_this_call:
            return RolloutResult(
                text=generated,
                transcript=transcript,
                tool_results=tool_results,
                model_calls=model_calls,
                stop_reason="finished",
            )

    return RolloutResult(
        text=generated,
        transcript=transcript,
        tool_results=tool_results,
        model_calls=max_model_calls,
        stop_reason="max_model_calls",
    )


async def batch_rollout_with_sandbox(
    prompts: Sequence[str],
    generate_until: GenerateUntil,
    sandbox: AsyncPythonSandboxPool,
    *,
    rollout_concurrency: int = 64,
    max_model_calls: int = 16,
    max_tool_calls: int = 8,
    stop_sequences: Sequence[str] = ("</code>",),
) -> list[RolloutResult]:
    """Run many independent ReTool rollouts concurrently.

    This helper is useful when the model backend already handles request-level
    scheduling, such as a vLLM server. For a native batched engine, provide a
    `generate_until` adapter that enqueues requests into that engine.
    """

    if rollout_concurrency < 1:
        raise ValueError("rollout_concurrency must be >= 1")
    sem = asyncio.Semaphore(rollout_concurrency)

    async def run_one(prompt: str) -> RolloutResult:
        async with sem:
            return await rollout_with_sandbox(
                prompt,
                generate_until,
                sandbox,
                max_model_calls=max_model_calls,
                max_tool_calls=max_tool_calls,
                stop_sequences=stop_sequences,
            )

    return list(await asyncio.gather(*(run_one(prompt) for prompt in prompts)))
