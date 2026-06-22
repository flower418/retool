import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retool_sandbox import (
    AsyncPythonSandboxPool,
    ModelStep,
    SandboxLimits,
    batch_rollout_with_sandbox,
    execute_next_unexecuted_code,
    rollout_with_sandbox,
)


class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_basic_execution(self):
        async with AsyncPythonSandboxPool(
            num_workers=1,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            result = await sandbox.run("print(2 + 3)")

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "5")

    async def test_concurrent_execution(self):
        async with AsyncPythonSandboxPool(
            num_workers=2,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            results = await asyncio.gather(
                *(sandbox.run(f"print({i} * {i})") for i in range(8))
            )

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual([result.stdout.strip() for result in results], [str(i * i) for i in range(8)])

    async def test_timeout_replaces_worker(self):
        async with AsyncPythonSandboxPool(
            num_workers=1,
            limits=SandboxLimits(timeout_s=0.2),
        ) as sandbox:
            timed_out = await sandbox.run("while True:\n    pass")
            after = await sandbox.run("print('alive')")

        self.assertTrue(timed_out.timed_out)
        self.assertTrue(after.ok)
        self.assertEqual(after.stdout.strip(), "alive")

    async def test_execute_next_unexecuted_code(self):
        content = "Compute.\n<code>\n```python\nprint(6 * 7)\n```\n</code>"
        async with AsyncPythonSandboxPool(
            num_workers=1,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            updated, result = await execute_next_unexecuted_code(content, sandbox)

        self.assertIsNotNone(result)
        self.assertIn("<interpreter>42</interpreter>", updated)

    async def test_execute_bare_python_fence(self):
        content = "Compute.\n```python\nprint(11 * 11)\n```"
        async with AsyncPythonSandboxPool(
            num_workers=1,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            updated, result = await execute_next_unexecuted_code(content, sandbox)

        self.assertIsNotNone(result)
        self.assertIn("<interpreter>121</interpreter>", updated)

    async def test_rollout_with_sandbox(self):
        calls = [
            ModelStep(
                text="Let Python compute it.\n<code>\n```python\nprint(40 + 2)\n```",
                stop_text="</code>",
            ),
            ModelStep(
                text="\nSo the answer is 42.\n<answer>\n\\boxed{42}\n",
                stop_text="</answer>",
            ),
        ]

        async def generate_until(_transcript, _stops):
            return calls.pop(0)

        async with AsyncPythonSandboxPool(
            num_workers=1,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            result = await rollout_with_sandbox("question\n", generate_until, sandbox)

        self.assertEqual(result.stop_reason, "answer")
        self.assertEqual(len(result.tool_results), 1)
        self.assertIn("<interpreter>42</interpreter>", result.text)
        self.assertTrue(result.text.rstrip().endswith("</answer>"))

    async def test_batch_rollout_with_sandbox(self):
        async def generate_until(transcript, _stops):
            if "<interpreter>" not in transcript:
                n = int(transcript.rsplit(" ", 1)[-1])
                return ModelStep(
                    text=f"<code>\n```python\nprint({n} + 1)\n```",
                    stop_text="</code>",
                )
            value = transcript.rsplit("<interpreter>", 1)[1].split("</interpreter>", 1)[0]
            return ModelStep(
                text=f"\n<answer>\n\\boxed{{{value}}}\n",
                stop_text="</answer>",
            )

        prompts = [f"question {i}" for i in range(4)]
        async with AsyncPythonSandboxPool(
            num_workers=2,
            limits=SandboxLimits(timeout_s=1.0),
        ) as sandbox:
            results = await batch_rollout_with_sandbox(
                prompts,
                generate_until,
                sandbox,
                rollout_concurrency=4,
            )

        self.assertEqual([result.stop_reason for result in results], ["answer"] * 4)
        self.assertEqual(
            [result.tool_results[0].stdout.strip() for result in results],
            ["1", "2", "3", "4"],
        )


if __name__ == "__main__":
    unittest.main()
