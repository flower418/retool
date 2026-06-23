"""High-throughput asyncio pool for executing Python snippets."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxLimits:
    """Limits applied to each execution request or worker process."""

    timeout_s: float = 2.0
    max_output_bytes: int = 20000
    memory_mb: int = 512
    file_mb: int = 8


@dataclass(frozen=True)
class CodeExecutionResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    worker_crashed: bool = False
    elapsed_ms: float = 0.0
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def interpreter_text(self) -> str:
        """Return compact text suitable for a ReTool <interpreter> block."""

        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr and (not self.ok or self.stderr_truncated):
            stderr = self.stderr.rstrip()
            if self.ok:
                parts.append(stderr)
            else:
                last_line = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), stderr)
                parts.append(f"[sandbox error] {last_line}")
        if self.error and not self.stderr:
            parts.append(self.error.rstrip())
        if self.timed_out:
            parts.append("[sandbox timeout]")
        if self.worker_crashed:
            parts.append("[sandbox worker crashed]")
        if self.stdout_truncated or self.stderr_truncated:
            parts.append("[sandbox output truncated]")
        return "\n".join(part for part in parts if part).strip()


class _Worker:
    def __init__(
        self,
        *,
        worker_id: int,
        python_executable: str,
        limits: SandboxLimits,
        isolated: bool,
        no_site: bool,
        root_dir: Path,
    ) -> None:
        self.worker_id = worker_id
        self.python_executable = python_executable
        self.limits = limits
        self.isolated = isolated
        self.no_site = no_site
        self.root_dir = root_dir
        self.proc: asyncio.subprocess.Process | None = None
        self.worker_dir = root_dir / f"worker-{worker_id}"

    async def start(self) -> None:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        worker_script = Path(__file__).with_name("_worker.py")
        cmd = [self.python_executable]
        if self.isolated:
            cmd.append("-I")
        if self.no_site:
            cmd.append("-S")
        cmd.extend(
            [
                str(worker_script),
                "--base-dir",
                str(self.worker_dir),
                "--max-output-bytes",
                str(self.limits.max_output_bytes),
                "--memory-mb",
                str(self.limits.memory_mb),
                "--file-mb",
                str(self.limits.file_mb),
            ]
        )
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def close(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        if proc.returncode is not None:
            self.proc = None
            return
        try:
            await self._write_message({"type": "shutdown"})
            await asyncio.wait_for(proc.wait(), timeout=0.5)
        except Exception:
            proc.kill()
            await proc.wait()
        finally:
            self.proc = None

    async def kill(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    async def run(self, code: str) -> CodeExecutionResult:
        if self.proc is None or self.proc.returncode is not None:
            return CodeExecutionResult(
                ok=False,
                error="worker is not running",
                worker_crashed=True,
            )

        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            await self._write_message(
                {"type": "run", "id": request_id, "code": code}
            )
            response = await asyncio.wait_for(
                self._read_message(),
                timeout=self.limits.timeout_s,
            )
        except asyncio.TimeoutError:
            await self.kill()
            return CodeExecutionResult(
                ok=False,
                timed_out=True,
                error=f"timeout after {self.limits.timeout_s:.3g}s",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            await self.kill()
            return CodeExecutionResult(
                ok=False,
                worker_crashed=True,
                error=f"worker failed: {exc}",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        if response.get("id") != request_id:
            await self.kill()
            return CodeExecutionResult(
                ok=False,
                worker_crashed=True,
                error="worker returned a mismatched response id",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        return CodeExecutionResult(
            ok=bool(response.get("ok")),
            stdout=str(response.get("stdout") or ""),
            stderr=str(response.get("stderr") or ""),
            error=response.get("error"),
            elapsed_ms=float(response.get("elapsed_ms") or 0.0),
            stdout_truncated=bool(response.get("stdout_truncated")),
            stderr_truncated=bool(response.get("stderr_truncated")),
        )

    async def _write_message(self, message: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("worker stdin is unavailable")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.proc.stdin.write(str(len(payload)).encode("ascii") + b"\n")
        self.proc.stdin.write(payload + b"\n")
        await self.proc.stdin.drain()

    async def _read_message(self) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("worker stdout is unavailable")
        header = await self.proc.stdout.readline()
        if not header:
            raise RuntimeError("worker exited before sending a response")
        size = int(header.strip())
        payload = await self.proc.stdout.readexactly(size)
        await self.proc.stdout.readline()
        return json.loads(payload.decode("utf-8"))


class AsyncPythonSandboxPool:
    """Async worker pool for running Python snippets at rollout time.

    This is optimized for trusted or semi-trusted model-generated math/code
    snippets. It is an execution sandbox with process isolation and limits, not
    a hardened security boundary against malicious Python.
    """

    def __init__(
        self,
        *,
        num_workers: int = 4,
        limits: SandboxLimits | None = None,
        python_executable: str | None = None,
        isolated: bool = True,
        no_site: bool = True,
        root_dir: str | Path | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = num_workers
        self.limits = limits or SandboxLimits()
        self.python_executable = python_executable or sys.executable
        self.isolated = isolated
        self.no_site = no_site
        self._own_root_dir = root_dir is None
        self.root_dir = Path(root_dir) if root_dir is not None else Path(
            tempfile.mkdtemp(prefix="retool-sandbox-")
        )
        self._workers: list[_Worker] = []
        self._available: asyncio.Queue[_Worker] = asyncio.Queue()
        self._started = False

    async def __aenter__(self) -> "AsyncPythonSandboxPool":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._started:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for worker_id in range(self.num_workers):
            worker = self._new_worker(worker_id)
            await worker.start()
            self._workers.append(worker)
            self._available.put_nowait(worker)
        self._started = True

    async def close(self) -> None:
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(
            *(worker.close() for worker in self._workers),
            return_exceptions=True,
        )
        self._workers.clear()
        self._started = False
        if self._own_root_dir:
            shutil.rmtree(self.root_dir, ignore_errors=True)

    async def run(self, code: str) -> CodeExecutionResult:
        if not self._started:
            await self.start()
        worker = await self._available.get()
        result = await worker.run(code)
        if result.timed_out or result.worker_crashed:
            worker = await self._replace_worker(worker)
        self._available.put_nowait(worker)
        return result

    def _new_worker(self, worker_id: int) -> _Worker:
        return _Worker(
            worker_id=worker_id,
            python_executable=self.python_executable,
            limits=self.limits,
            isolated=self.isolated,
            no_site=self.no_site,
            root_dir=self.root_dir,
        )

    async def _replace_worker(self, old_worker: _Worker) -> _Worker:
        await old_worker.kill()
        worker = self._new_worker(old_worker.worker_id)
        await worker.start()
        for idx, existing in enumerate(self._workers):
            if existing.worker_id == old_worker.worker_id:
                self._workers[idx] = worker
                break
        return worker
