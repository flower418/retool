"""MATH reward wrapper with robust final-answer normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from verl.utils.reward_score import math_dapo

try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:  # pragma: no cover - reward still works with exact matching.
    _sympy = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


ANSWER_LINE_RE = re.compile(r"(?im)^\s*\*{0,2}\s*Answer\s*:\s*\*{0,2}\s*(?P<answer>\S[^\n]*)$")
ANSWER_PHRASE_RE = re.compile(
    r"(?i)\b(?:the\s+)?final\s+answer\s+is\s+(?:indeed\s+)?(?P<final>[^\n]+)|"
    r"\bthe\s+answer\s+is\s+(?:indeed\s+)?(?P<answer>[^\n]+)"
)
PROTECTED_BLOCK_RE = re.compile(
    r"<(?:code|interpreter)>.*?</(?:code|interpreter)>|```(?:python|py)?\s*\n.*?```",
    re.DOTALL | re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
UNIT_SUFFIX_RE = re.compile(
    r"(?i)([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:kpa|pa|mpa|atm|psi|liters?|litres?|l|ml|"
    r"degrees?|deg|units?|meters?|metres?|m|cm|mm|km|"
    r"feet|foot|ft|inches?|in|hours?|hrs?|minutes?|mins?|"
    r"seconds?|secs?|s|kg|g|grams?|pounds?|lbs?)\.?$"
)
ALLOWED_SYMPY_WORDS = {"e", "pi", "sqrt"}

SUPERSCRIPT_MAP = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207a": "+",
    "\u207b": "-",
    "\u207d": "(",
    "\u207e": ")",
    "\u207f": "n",
}
SUBSCRIPT_MAP = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
}
UNICODE_FRACTIONS = {
    "\u00bd": "1/2",
    "\u2153": "1/3",
    "\u2154": "2/3",
    "\u00bc": "1/4",
    "\u00be": "3/4",
    "\u2155": "1/5",
    "\u2156": "2/5",
    "\u2157": "3/5",
    "\u2158": "4/5",
    "\u2159": "1/6",
    "\u215a": "5/6",
    "\u215b": "1/8",
    "\u215c": "3/8",
    "\u215d": "5/8",
    "\u215e": "7/8",
}


@dataclass(frozen=True)
class ExtractedAnswer:
    raw: str
    kind: str
    format_ok: bool


def _stable_result(
    *,
    score: float,
    acc: bool,
    pred: str,
    raw_pred: str,
    ground_truth: Any,
    format_ok: bool,
    answer_extracted: bool,
    reason: str,
    match_type: str,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "acc": bool(acc),
        "pred": pred,
        "raw_pred": raw_pred,
        "ground_truth": str(ground_truth),
        "format_ok": bool(format_ok),
        "answer_extracted": bool(answer_extracted),
        "reason": reason,
        "match_type": match_type,
    }


def _strip_outer_braces(text: str) -> str:
    text = text.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        text = text[1:-1].strip()
    return text


def _extract_braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _last_boxed(text: str) -> str | None:
    last: str | None = None
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            index = text.find(command, start)
            if index < 0:
                break
            brace_index = text.find("{", index + len(command))
            if brace_index < 0:
                break
            extracted = _extract_braced(text, brace_index)
            if extracted is not None:
                last = extracted[0]
                start = extracted[1]
            else:
                start = index + len(command)
    return last


def _replace_unicode_scripts(text: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in SUPERSCRIPT_MAP:
            converted: list[str] = []
            while index < len(text) and text[index] in SUPERSCRIPT_MAP:
                converted.append(SUPERSCRIPT_MAP[text[index]])
                index += 1
            exponent = "".join(converted)
            out.append("^" + (exponent if len(exponent) == 1 else f"({exponent})"))
            continue
        if char in SUBSCRIPT_MAP:
            converted = []
            while index < len(text) and text[index] in SUBSCRIPT_MAP:
                converted.append(SUBSCRIPT_MAP[text[index]])
                index += 1
            out.append("_" + "".join(converted))
            continue
        out.append(UNICODE_FRACTIONS.get(char, char))
        index += 1
    return "".join(out)


def _strip_light_markup(answer: str) -> str:
    answer = _replace_unicode_scripts(str(answer))
    answer = answer.strip().strip("*").strip()
    answer = answer.removeprefix("`").removesuffix("`").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left) : -len(right)].strip()
    boxed = _last_boxed(answer)
    if boxed is not None:
        answer = boxed
    answer = re.sub(r"\\(?:text|textbf|mathrm)\{([^{}]*)\}", r"\1", answer)
    answer = answer.strip().strip("*").strip()
    return answer.rstrip(".\u3002\uff0e").strip()


def _visible_text(solution_str: str) -> str:
    return PROTECTED_BLOCK_RE.sub("\n", solution_str)


def extract_final_answer(solution_str: str) -> ExtractedAnswer | None:
    """Extract the final top-level answer declaration from a model response."""

    protected_spans = [match.span() for match in PROTECTED_BLOCK_RE.finditer(solution_str)]
    answer_candidates: list[ExtractedAnswer] = []
    for match in ANSWER_LINE_RE.finditer(solution_str):
        if any(start <= match.start() < end for start, end in protected_spans):
            continue
        raw = _strip_light_markup(match.group("answer"))
        if raw:
            answer_candidates.append(
                ExtractedAnswer(
                    raw=raw,
                    kind="answer_line",
                    format_ok=not solution_str[match.end() :].strip(),
                )
            )
    if answer_candidates:
        return answer_candidates[-1]

    visible = _visible_text(solution_str)
    phrase_candidates: list[ExtractedAnswer] = []
    for match in ANSWER_PHRASE_RE.finditer(visible[-1200:]):
        raw_group = match.group("final") or match.group("answer") or ""
        raw = _strip_light_markup(raw_group.split("<", 1)[0])
        if raw:
            phrase_candidates.append(ExtractedAnswer(raw=raw, kind="answer_phrase", format_ok=False))
    if phrase_candidates:
        return phrase_candidates[-1]

    boxed = _last_boxed(visible[-1200:])
    if boxed is not None:
        raw = _strip_light_markup(boxed)
        if raw:
            return ExtractedAnswer(raw=raw, kind="boxed", format_ok=False)

    return None


def _fix_latex_shorthand(text: str) -> str:
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\frac\s*([^\s{}])\s*([^\s{}])", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\sqrt\s*([^\s{}])", r"\\sqrt{\1}", text)
    return text


def _latex_to_plain(text: str) -> str:
    text = _fix_latex_shorthand(text)
    text = text.replace("\\left", "").replace("\\right", "")
    replacements = {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pi": "pi",
        "\u2212": "-",
        "\u00d7": "*",
        "\u00f7": "/",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", text)

    text = re.sub(r"\\[!,;:\s]+", "", text)
    text = text.replace("\\", "")
    return text


def _strip_numeric_unit(answer: str) -> str:
    answer = answer.strip().strip("$")
    match = UNIT_SUFFIX_RE.fullmatch(answer)
    if match:
        return match.group(1)
    return answer


def _canonical_text(answer: str) -> str:
    answer = _strip_light_markup(answer)
    answer = _latex_to_plain(answer)
    answer = answer.replace("**", "^")
    answer = answer.replace(" ", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer):
        answer = answer.replace(",", "")
    answer = _strip_numeric_unit(answer)
    try:
        answer = math_dapo.normalize_final_answer(answer)
    except Exception:
        pass
    return answer.strip()


def _decimal(text: str) -> Decimal | None:
    normalized = _canonical_text(text)
    normalized = _strip_numeric_unit(normalized)
    if not NUMERIC_RE.fullmatch(normalized):
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _sympy_expr(text: str):
    if _sympy is None or parse_expr is None:
        return None

    expression = _canonical_text(text)
    expression = expression.replace("^", "**")
    expression = _strip_outer_braces(expression)
    if not expression:
        return None
    if re.search(r"[^0-9A-Za-z_+\-*/().,\[\]{}]", expression):
        return None
    words = re.findall(r"[A-Za-z]+", expression)
    if any(len(word) > 1 and word not in ALLOWED_SYMPY_WORDS for word in words):
        return None

    local_dict = {name: _sympy.Symbol(name) for name in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    local_dict.update({"sqrt": _sympy.sqrt, "pi": _sympy.pi, "e": _sympy.E})
    global_dict = {
        "Symbol": _sympy.Symbol,
        "Integer": _sympy.Integer,
        "Rational": _sympy.Rational,
        "Float": _sympy.Float,
        "Pow": _sympy.Pow,
        "Add": _sympy.Add,
        "Mul": _sympy.Mul,
        "sqrt": _sympy.sqrt,
    }
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(
            expression,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception:
        return None


def _equivalent(pred: str, ground_truth: Any) -> tuple[bool, str, str]:
    gt = str(ground_truth)
    pred_norm = _canonical_text(pred)
    gt_norm = _canonical_text(gt)
    if pred_norm == gt_norm:
        return True, "normalized_exact", pred_norm

    pred_decimal = _decimal(pred)
    gt_decimal = _decimal(gt)
    if pred_decimal is not None and gt_decimal is not None and pred_decimal == gt_decimal:
        return True, "decimal_exact", pred_norm

    pred_expr = _sympy_expr(pred)
    gt_expr = _sympy_expr(gt)
    if pred_expr is not None and gt_expr is not None:
        try:
            if bool(_sympy.simplify(pred_expr - gt_expr) == 0):
                return True, "sympy_equivalent", pred_norm
        except Exception:
            pass
        try:
            if bool(_sympy.Eq(pred_expr, gt_expr)):
                return True, "sympy_equal", pred_norm
        except Exception:
            pass

    return False, "mismatch", pred_norm


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Return +1/-1 based on the response's final declared answer.

    The prompt standard is a final line `Answer: <final answer>`. The scorer is
    intentionally more robust than the prompt: it also extracts common final
    answer phrases and boxed answers so a mathematically correct response is not
    lost only because it used a nearby legacy surface form. The `format_ok`
    field records whether the strict prompt format was followed.
    """

    del data_source, extra_info, kwargs
    extracted = extract_final_answer(str(solution_str))
    if extracted is None:
        return _stable_result(
            score=-1.0,
            acc=False,
            pred="[NO_ANSWER]",
            raw_pred="",
            ground_truth=ground_truth,
            format_ok=False,
            answer_extracted=False,
            reason="missing_final_answer",
            match_type="none",
        )

    canonical_solution = f"Answer: {extracted.raw}"
    try:
        dapo_result = math_dapo.compute_score(canonical_solution, ground_truth)
    except Exception:
        dapo_result = {"score": -1.0, "acc": False, "pred": _canonical_text(extracted.raw)}

    if dapo_result.get("score") == 1.0:
        pred = str(dapo_result.get("pred") or _canonical_text(extracted.raw))
        return _stable_result(
            score=1.0,
            acc=True,
            pred=pred,
            raw_pred=extracted.raw,
            ground_truth=ground_truth,
            format_ok=extracted.format_ok,
            answer_extracted=True,
            reason="correct",
            match_type=f"math_dapo:{extracted.kind}",
        )

    correct, match_type, pred_norm = _equivalent(extracted.raw, ground_truth)
    if correct:
        return _stable_result(
            score=1.0,
            acc=True,
            pred=pred_norm,
            raw_pred=extracted.raw,
            ground_truth=ground_truth,
            format_ok=extracted.format_ok,
            answer_extracted=True,
            reason="correct",
            match_type=f"{match_type}:{extracted.kind}",
        )

    return _stable_result(
        score=-1.0,
        acc=False,
        pred=str(dapo_result.get("pred") or pred_norm),
        raw_pred=extracted.raw,
        ground_truth=ground_truth,
        format_ok=extracted.format_ok,
        answer_extracted=True,
        reason="answer_mismatch",
        match_type=f"mismatch:{extracted.kind}",
    )
