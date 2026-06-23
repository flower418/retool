#!/usr/bin/env python3
"""Minimal ReTool sandbox rollout demo.

Real rollout backends should replace `generate_until` with vLLM/Transformers
generation that stops on `</code>` and emits a final `Answer:` line.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retool_sandbox import AsyncPythonSandboxPool, ModelStep, SandboxLimits, rollout_with_sandbox


async def main() -> None:
    chunks = [
        ModelStep(
            text=(
                "We can compute it directly.\n"
                "<code>\n"
                "```python\n"
                "print(sum(range(1, 101)))\n"
                "```"
            ),
            stop_text="</code>",
        ),
        ModelStep(
            text="\nThe computation gives 5050.\nAnswer: 5050\n",
            finished=True,
        ),
    ]

    async def generate_until(transcript, stop_sequences):
        del transcript, stop_sequences
        return chunks.pop(0)

    async with AsyncPythonSandboxPool(
        num_workers=2,
        limits=SandboxLimits(timeout_s=1.0, max_output_bytes=4000),
    ) as sandbox:
        result = await rollout_with_sandbox(
            "Question: What is the sum of all integers from 1 to 100?\n",
            generate_until,
            sandbox,
        )

    print(result.text)
    print(f"\nstop_reason={result.stop_reason} tool_calls={len(result.tool_results)}")


if __name__ == "__main__":
    asyncio.run(main())
