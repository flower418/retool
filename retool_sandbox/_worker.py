#!/usr/bin/env python3
"""Persistent worker process used by AsyncPythonSandboxPool.

The parent process talks to this worker with length-prefixed JSON messages over
stdin/stdout. User code stdout/stderr file descriptors are redirected while code
runs, so even os.write(1, ...) cannot corrupt the control protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


def configure_resource_limits(memory_mb: int, file_mb: int) -> None:
    try:
        import resource
    except Exception:
        return

    if memory_mb > 0:
        limit = memory_mb * 1024 * 1024
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            resource_id = getattr(resource, name, None)
            if resource_id is None:
                continue
            try:
                resource.setrlimit(resource_id, (limit, limit))
            except (OSError, ValueError):
                pass

    if file_mb > 0:
        limit = file_mb * 1024 * 1024
        resource_id = getattr(resource, "RLIMIT_FSIZE", None)
        if resource_id is not None:
            try:
                resource.setrlimit(resource_id, (limit, limit))
            except (OSError, ValueError):
                pass

    resource_id = getattr(resource, "RLIMIT_NOFILE", None)
    if resource_id is not None:
        try:
            hard = resource.getrlimit(resource_id)[1]
            soft = min(128, hard if hard > 0 else 128)
            resource.setrlimit(resource_id, (soft, hard))
        except (OSError, ValueError):
            pass


def read_message() -> dict | None:
    header = sys.stdin.buffer.readline()
    if not header:
        return None
    size = int(header.strip())
    payload = sys.stdin.buffer.read(size)
    sys.stdin.buffer.readline()
    return json.loads(payload.decode("utf-8"))


def write_message(protocol_out, message: dict) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    protocol_out.write(str(len(payload)).encode("ascii") + b"\n")
    protocol_out.write(payload + b"\n")
    protocol_out.flush()


def read_capped(path: Path, max_bytes: int) -> tuple[str, bool]:
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def run_code(code: str, base_dir: Path, max_output_bytes: int) -> dict:
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=base_dir))
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    snippet_path = run_dir / "snippet.py"
    snippet_path.write_text(code, encoding="utf-8")

    old_cwd = os.getcwd()
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    started = time.perf_counter()
    ok = True
    error = None

    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            os.chdir(run_dir)
            try:
                namespace = {
                    "__builtins__": __builtins__,
                    "__file__": str(snippet_path),
                    "__name__": "__main__",
                }
                compiled = compile(code, str(snippet_path), "exec")
                exec(compiled, namespace, namespace)
            except BaseException:
                ok = False
                error = traceback.format_exc()
                traceback.print_exc()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.chdir(old_cwd)
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

    elapsed_ms = (time.perf_counter() - started) * 1000
    stdout, stdout_truncated = read_capped(stdout_path, max_output_bytes)
    stderr, stderr_truncated = read_capped(stderr_path, max_output_bytes)
    shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "ok": ok,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "elapsed_ms": elapsed_ms,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--max-output-bytes", type=int, default=20000)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--file-mb", type=int, default=8)
    args = parser.parse_args()

    configure_resource_limits(args.memory_mb, args.file_mb)
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    protocol_out = os.fdopen(os.dup(1), "wb", buffering=0)

    while True:
        message = read_message()
        if message is None:
            return 0
        if message.get("type") == "shutdown":
            return 0
        if message.get("type") != "run":
            write_message(
                protocol_out,
                {"id": message.get("id"), "ok": False, "error": "unknown message type"},
            )
            continue

        result = run_code(
            str(message.get("code", "")),
            base_dir=base_dir,
            max_output_bytes=args.max_output_bytes,
        )
        result["id"] = message.get("id")
        write_message(protocol_out, result)


if __name__ == "__main__":
    raise SystemExit(main())
