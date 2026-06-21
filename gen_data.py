#!/usr/bin/env python3
"""
Generate ReTool-style SFT data directly from questions.

Each output line is a two-message SFT conversation:
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]

The assistant content is produced by an OpenAI-compatible chat model and should
contain executable <code> blocks, matching <interpreter> outputs, and a final
<answer> block.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("pip install 'openai>=1.0' tqdm   (see requirements.txt)")

try:
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    sys.exit("pip install tqdm")


DEFAULT_QUESTION = (
    "Find the number of integers less than or equal to 100 that are equal to "
    "$a+b+ab$ for some choice of distinct positive integers a and b."
)


def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_user_content(template: str, question: str) -> str:
    return template.replace("{question}", question)


def read_questions(infile: str | None, question: str | None) -> list[dict]:
    if question:
        return [{"id": 0, "question": question}]
    if not infile:
        return [{"id": 0, "question": DEFAULT_QUESTION}]

    rows = []
    with open(infile, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj:
                raise ValueError(f"line {i + 1} missing required field: question")
            rows.append({"id": obj.get("id", i), "question": str(obj["question"])})
    return rows


def load_done_questions(out_path: str, template: str) -> set[str]:
    done = set()
    if not os.path.exists(out_path):
        return done

    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
                if isinstance(messages, list) and len(messages) >= 2:
                    user_content = messages[0].get("content", "")
                    assistant_content = messages[1].get("content", "")
                    question = extract_question_from_user_content(
                        template, user_content
                    )
                    if question and not validate_assistant_content(
                        assistant_content,
                        execute_code=False,
                    ):
                        done.add(question)
            except Exception:
                continue
    return done


def extract_question_from_user_content(template: str, user_content: str) -> str | None:
    if "{question}" not in template:
        return None
    prefix, suffix = template.split("{question}", 1)
    if not user_content.startswith(prefix):
        return None
    if suffix and not user_content.endswith(suffix):
        return None
    end = len(user_content) - len(suffix) if suffix else len(user_content)
    return user_content[len(prefix):end]


def extract_code_interpreter_pairs(content: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"<code>\s*```python\n(.*?)```\s*</code>\s*<interpreter>(.*?)</interpreter>",
        re.S,
    )
    return pattern.findall(content)


def materialize_interpreter_outputs(content: str, exec_timeout: float) -> tuple[str, list[str]]:
    issues = []

    def replace_match(match):
        code_block = match.group(1)
        code = match.group("code")
        actual_output, error = run_python_code(code, exec_timeout)
        if error:
            issues.append(f"code execution error: {error}")
            return match.group(0)
        return f"{code_block}\n<interpreter>{actual_output.rstrip()}</interpreter>"

    pattern = re.compile(
        r"(<code>\s*```python\n(?P<code>.*?)```\s*</code>)(?:\s*<interpreter>.*?</interpreter>)?",
        re.S,
    )
    return pattern.sub(replace_match, content), issues


def run_python_code(code: str, timeout: float) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as exc:
        return None, f"execution failed: {exc}"

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        return None, f"nonzero exit {proc.returncode}: {stderr}"
    return proc.stdout, None


def extract_boxed_answer(content: str) -> str | None:
    match = re.search(r"<answer>\s*\\boxed\{([^{}]+)\}\s*</answer>\s*$", content.strip(), re.S)
    if not match:
        return None
    return match.group(1).strip()


def validate_final_answer_against_last_output(content: str) -> list[str]:
    issues = []
    boxed = extract_boxed_answer(content)
    if boxed is None:
        return issues

    pairs = extract_code_interpreter_pairs(content)
    if not pairs:
        return issues

    last_output = pairs[-1][1].strip()
    last_line = last_output.splitlines()[-1].strip() if last_output else ""
    if re.fullmatch(r"-?\d+", boxed) and not re.fullmatch(r"-?\d+", last_line):
        issues.append(
            f"last interpreter output line {last_line!r} is not a numeric final answer"
        )
    elif re.fullmatch(r"-?\d+", boxed) and re.fullmatch(r"-?\d+", last_line):
        if boxed != last_line:
            issues.append(f"boxed answer {boxed!r} does not match last interpreter output {last_line!r}")
    return issues


def validate_assistant_content(content: str, execute_code: bool = False, exec_timeout: float = 10.0) -> list[str]:
    issues = []
    stripped = content.strip()
    if "<code>" not in content or "</code>" not in content:
        issues.append("missing <code> block")
    if "<interpreter>" not in content or "</interpreter>" not in content:
        issues.append("missing <interpreter> block")
    if "<answer>" not in content or "</answer>" not in content:
        issues.append("missing <answer> block")
    if content.count("<answer>") != 1 or content.count("</answer>") != 1:
        issues.append("answer block count must be exactly one")
    if "</answer>" in content and not stripped.endswith("</answer>"):
        issues.append("answer block must be at the end")
    if "\\boxed{" not in content:
        issues.append("missing boxed final answer")
    code_open = len(re.findall(r"<code>\s*```python", content))
    code_close = len(re.findall(r"</code>", content))
    interpreter_open = len(re.findall(r"<interpreter>", content))
    interpreter_close = len(re.findall(r"</interpreter>", content))
    pairs = extract_code_interpreter_pairs(content)
    if code_open != code_close:
        issues.append("code block count mismatch")
    if interpreter_open != interpreter_close:
        issues.append("interpreter block count mismatch")
    if code_open != interpreter_open:
        issues.append("each code block must have one interpreter block")
    if code_open != len(pairs):
        issues.append("each code block must be immediately followed by interpreter output")
    if execute_code and not issues:
        for idx, (code, claimed_output) in enumerate(pairs, start=1):
            actual_output, error = run_python_code(code, exec_timeout)
            if error:
                issues.append(f"code block {idx} {error}")
                continue
            if actual_output.rstrip() != claimed_output.rstrip():
                issues.append(
                    f"code block {idx} interpreter mismatch: "
                    f"expected {actual_output.rstrip()!r}, got {claimed_output.rstrip()!r}"
                )
    if not issues:
        issues.extend(validate_final_answer_against_last_output(content))
    return issues


async def call_one(
    client,
    sem,
    model,
    template,
    row,
    max_tokens,
    temperature,
    retries,
    format_retries,
    validate_execution,
    exec_timeout,
    materialize_interpreter,
):
    async with sem:
        user_content = build_user_content(template, row["question"])
        base_messages = [{"role": "user", "content": user_content}]
        last_err = None
        last_invalid = None

        for format_attempt in range(format_retries + 1):
            messages = list(base_messages)
            if last_invalid:
                messages.extend([
                    {"role": "assistant", "content": last_invalid["content"][:2000]},
                    {
                        "role": "user",
                        "content": (
                            "Regenerate the full solution from scratch. The previous output violated "
                            f"these requirements: {', '.join(last_invalid['issues'])}. "
                            "Use exactly one final <answer> block, put it at the very end, and include "
                            "at least one <code> block followed by its exact <interpreter> output."
                        ),
                    },
                ])

            for attempt in range(retries):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    choice = resp.choices[0]
                    assistant_content = choice.message.content or ""
                    reasoning_content = getattr(choice.message, "reasoning_content", None)
                    if reasoning_content and assistant_content:
                        assistant_content = f"{reasoning_content}\n\n{assistant_content}"
                    elif reasoning_content:
                        assistant_content = reasoning_content
                    materialize_issues = []
                    if materialize_interpreter:
                        assistant_content, materialize_issues = materialize_interpreter_outputs(
                            assistant_content,
                            exec_timeout,
                        )
                    issues = validate_assistant_content(
                        assistant_content,
                        execute_code=validate_execution,
                        exec_timeout=exec_timeout,
                    )
                    issues.extend(materialize_issues)
                    if choice.finish_reason == "length":
                        issues.append("finish_reason length")
                    if not issues:
                        return {
                            "id": row["id"],
                            "question": row["question"],
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ],
                            "model": model,
                            "finish_reason": choice.finish_reason,
                            "validation_issues": issues,
                            "format_attempts": format_attempt + 1,
                        }
                    last_invalid = {"content": assistant_content, "issues": issues}
                    break
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(min(2 ** attempt, 30))

        if last_invalid:
            return {
                "id": row["id"],
                "question": row["question"],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": last_invalid["content"]},
                ],
                "model": model,
                "finish_reason": "format_error",
                "validation_issues": last_invalid["issues"],
            }

        return {
            "id": row["id"],
            "question": row["question"],
            "messages": base_messages,
            "model": model,
            "finish_reason": "error",
            "validation_issues": ["api_error"],
            "error": str(last_err),
        }


async def main_async(args):
    template = load_prompt_template(args.prompt)
    rows = read_questions(args.infile, args.question)
    done_questions = load_done_questions(args.outfile, template)
    todo = [r for r in rows if r["question"] not in done_questions]
    print(
        f"total={len(rows)} done={len(done_questions)} todo={len(todo)} "
        f"model={args.model} concurrency={args.concurrency}"
    )
    if not todo:
        print("nothing to do.")
        return

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    sem = asyncio.Semaphore(args.concurrency)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_warn = n_err = 0
    meta_path = Path(args.outfile).with_suffix(Path(args.outfile).suffix + ".meta.jsonl")
    with open(args.outfile, "a", encoding="utf-8") as fout, open(meta_path, "a", encoding="utf-8") as mdf:
        tasks = [
            call_one(
                client,
                sem,
                args.model,
                template,
                row,
                args.max_tokens,
                args.temperature,
                args.retries,
                args.format_retries,
                not args.no_exec_validate,
                args.exec_timeout,
                not args.no_materialize_interpreter,
            )
            for row in todo
        ]
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="generating"):
            res = await coro
            if res["finish_reason"] == "error":
                n_err += 1
                mdf.write(json.dumps(res, ensure_ascii=False) + "\n")
                mdf.flush()
                continue

            if res["validation_issues"]:
                n_warn += 1
                mdf.write(json.dumps({
                    "id": res["id"],
                    "question": res["question"],
                    "model": res["model"],
                    "finish_reason": res["finish_reason"],
                    "validation_issues": res["validation_issues"],
                }, ensure_ascii=False) + "\n")
                mdf.flush()
                continue
            else:
                n_ok += 1

            fout.write(json.dumps(res["messages"], ensure_ascii=False) + "\n")
            fout.flush()
            mdf.write(json.dumps({
                "id": res["id"],
                "question": res["question"],
                "model": res["model"],
                "finish_reason": res["finish_reason"],
                "validation_issues": res["validation_issues"],
                "format_attempts": res.get("format_attempts"),
            }, ensure_ascii=False) + "\n")
            mdf.flush()

    print(f"done. ok={n_ok} warn={n_warn} err={n_err} -> {args.outfile}")
    print(f"metadata -> {meta_path}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Generate direct ReTool SFT messages from questions")
    ap.add_argument("--in", dest="infile", default=None, help="JSONL input with a required question field")
    ap.add_argument("--question", default=None, help="Generate one SFT example from this question")
    ap.add_argument("--out", dest="outfile", default="sft_train.jsonl")
    ap.add_argument("--prompt", default="prompts/solve_with_code.txt")
    ap.add_argument("--model", default=os.environ.get("GEN_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--format-retries", dest="format_retries", type=int, default=2)
    ap.add_argument("--exec-timeout", dest="exec_timeout", type=float, default=10.0)
    ap.add_argument(
        "--no-exec-validate",
        action="store_true",
        help="Do not execute generated Python code to verify interpreter outputs",
    )
    ap.add_argument(
        "--no-materialize-interpreter",
        action="store_true",
        help="Do not overwrite or insert interpreter outputs with locally executed stdout",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
