import re
import sys
import types
import unittest


def _normalize_final_answer(text):
    text = str(text).strip().strip("$").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
    return re.sub(r"\s+", "", text.rstrip(".。．"))


def _compute_score(solution_str, ground_truth):
    matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
    pred = _normalize_final_answer(matches[-1]) if matches else ""
    acc = pred == _normalize_final_answer(ground_truth)
    return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}


math_dapo = types.SimpleNamespace(
    normalize_final_answer=_normalize_final_answer,
    compute_score=_compute_score,
)
reward_score_module = types.ModuleType("verl.utils.reward_score")
reward_score_module.math_dapo = math_dapo
utils_module = types.ModuleType("verl.utils")
utils_module.reward_score = reward_score_module
verl_module = types.ModuleType("verl")
verl_module.utils = utils_module
sys.modules.setdefault("verl", verl_module)
sys.modules.setdefault("verl.utils", utils_module)
sys.modules.setdefault("verl.utils.reward_score", reward_score_module)

from retool_sandbox.math_reward import compute_score  # noqa: E402


EXPECTED_KEYS = {
    "score",
    "acc",
    "pred",
    "raw_pred",
    "ground_truth",
    "format_ok",
    "answer_extracted",
    "reason",
    "match_type",
}


class MathRewardTests(unittest.TestCase):
    def test_numeric_unit_fallback_keeps_reward_keys_stable(self):
        result = compute_score("math_dapo", "Answer: 2 kPa", "2")

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["acc"])
        self.assertTrue(result["format_ok"])
        self.assertEqual(result["pred"], "2")
        self.assertEqual(set(result), EXPECTED_KEYS)

    def test_markdown_answer_is_canonicalized(self):
        result = compute_score("math_dapo", "**Answer:** \\(10x\\)", "10x")

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["format_ok"])
        self.assertEqual(set(result), EXPECTED_KEYS)

    def test_final_answer_phrase_fallback(self):
        result = compute_score("math_dapo", "Therefore, the final answer is 10x.", "10x")

        self.assertEqual(result["score"], 1.0)
        self.assertFalse(result["format_ok"])

    def test_latex_fraction_matches_plain_fraction(self):
        result = compute_score("math_dapo", "Answer: 2/3", r"\dfrac23")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reason"], "correct")

    def test_unicode_superscript_matches_caret_exponent(self):
        result = compute_score("math_dapo", "Answer: 162x\u2074", "162x^4")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reason"], "correct")

    def test_wrong_symbolic_answer_stays_negative(self):
        result = compute_score("math_dapo", "Answer: 10", "10x")

        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["reason"], "answer_mismatch")

    def test_decimal_noise_does_not_round_to_integer(self):
        result = compute_score("math_dapo", "Answer: 1.9999999999999998", "2")

        self.assertEqual(result["score"], -1.0)

    def test_missing_final_answer_is_explicit_format_failure(self):
        result = compute_score("math_dapo", "The value is 42.", "42")

        self.assertEqual(result["score"], -1.0)
        self.assertFalse(result["answer_extracted"])
        self.assertEqual(result["reason"], "missing_final_answer")

    def test_answer_followed_by_more_output_is_not_format_ok(self):
        result = compute_score(
            "math_dapo",
            "Answer: 23\n\n<code>\n```python\nprint(23)\n```\n</code>",
            "23",
        )

        self.assertEqual(result["score"], 1.0)
        self.assertFalse(result["format_ok"])


if __name__ == "__main__":
    unittest.main()
