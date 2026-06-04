#!/usr/bin/env python3
"""
Phase 1 / Gate G1-RL — Check 2: Countdown reward functions (pure Python, no GPU).

We unit-test d1's ACTUAL reward code (diffu-grpo/reward_func.py), not a copy, so the
numbers we train against are the numbers d1 ships. The countdown reward path
(compute_score / validate_equation / evaluate_equation / extract_solution) is
self-contained; reward_func.py only imports math500_utils at module load for the
*math500* reward, which we never touch here — so we stub that one unrelated import
to avoid pulling sympy. Nothing in the countdown path is stubbed.

Reward spec (from reward_func.compute_score, format_score=0.1, score=1.0):
  - no parseable <answer>...</answer>                      -> 0    (None)
  - answer parses but numbers != available multiset        -> 0.1  (invalid eq)
  - valid numbers, evaluates, but != target                -> 0.1
  - valid numbers, evaluates within 1e-5 of target         -> 1.0

Run locally:  python3 src/test_rewards.py
Run in env :  python  src/test_rewards.py   (folded into exp/phase1_gate.sbatch)
Exit 0 = all pass; non-zero = a failure (CI-style).
"""
import os
import sys
import types

# --- locate d1's reward_func.py (cluster path first, then local read-only ref) ---
CANDIDATES = [
    "/scratch/gupta.yashv/diffusion-rl/d1/diffu-grpo",
    "/Volumes/Crucial_X9/Projects/_refs/d1-ro/diffu-grpo",
    os.path.join(os.path.dirname(__file__), "..", "d1", "diffu-grpo"),
]
D1_DIR = next((p for p in CANDIDATES if os.path.isfile(os.path.join(p, "reward_func.py"))), None)
if D1_DIR is None:
    print("FATAL: could not find d1 diffu-grpo/reward_func.py in", CANDIDATES)
    sys.exit(2)
sys.path.insert(0, D1_DIR)

# Stub the unrelated math500 import (sympy-heavy, never used on the countdown path).
if "math500_utils" not in sys.modules:
    _stub = types.ModuleType("math500_utils")
    for _name in ("remove_boxed", "last_boxed_only_string", "is_equiv", "boxed_in_answer"):
        setattr(_stub, _name, lambda *a, **k: None)
    sys.modules["math500_utils"] = _stub

import reward_func as RF  # the REAL d1 module

# ----------------------------------------------------------------------------------
# Test cases: (name, solution_str, numbers, target, expected_score)
# ----------------------------------------------------------------------------------
ANS = lambda body: f"<reasoning>\nthink\n</reasoning>\n<answer>\n{body}\n</answer>"

CASES = [
    # --- correct -> 1.0 ---
    ("correct_basic",        ANS("3 + 4 * 5"),        [3, 4, 5],  23, 1.0),
    ("correct_division",     ANS("6 / 2"),            [6, 2],      3, 1.0),
    ("correct_paren",        ANS("(2 + 3) * 4"),      [2, 3, 4],  20, 1.0),
    ("correct_subtract",     ANS("100 - 30 - 93"),    [30, 100, 93], -23, 1.0),  # ordering-free multiset
    ("correct_reorder_nums", ANS("5 * 4 + 3"),        [3, 4, 5],  23, 1.0),       # numbers in any order

    # --- valid numbers, wrong result -> 0.1 (format_score) ---
    ("valid_wrong_result",   ANS("3 + 4 + 5"),        [3, 4, 5], 100, 0.1),

    # --- numbers don't match available multiset -> 0.1 ---
    ("wrong_number_used",    ANS("3 + 4 + 6"),        [3, 4, 5],  13, 0.1),   # 6 not available
    ("number_reused",        ANS("3 + 3 + 4 + 5"),    [3, 4, 5],  15, 0.1),   # 3 used twice
    ("number_missing",       ANS("3 + 4"),            [3, 4, 5],   7, 0.1),   # 5 unused

    # --- no parseable answer -> 0 ---
    ("no_answer_tag",        "I think the answer is 3+4*5",        [3, 4, 5], 23, 0),
    ("empty_completion",     "",                                   [3, 4, 5], 23, 0),

    # --- robustness of parsing ---
    ("takes_last_answer",    ANS("9 + 9") + ANS("3 + 4 * 5"),      [3, 4, 5], 23, 1.0),  # last <answer> wins
    ("answer_with_spaces",   ANS("  3 + 4 * 5  "),                 [3, 4, 5], 23, 1.0),
]


def run():
    passed, failed = 0, 0
    print(f"reward_func from: {D1_DIR}")
    print("-" * 72)
    for name, sol, numbers, target, expected in CASES:
        gt = {"target": target, "numbers": numbers}
        got = RF.compute_score(sol, gt)  # the real function
        ok = abs(float(got) - float(expected)) < 1e-9
        flag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{flag}] {name:22s} expected={expected:<4} got={got}")

    # --- direct checks on the helper predicates (the multiset rule) ---
    print("-" * 72)
    helper_checks = [
        ("validate ok",      RF.validate_equation("3 + 4 * 5", [3, 4, 5]),  True),
        ("validate reused",  RF.validate_equation("3 + 3 + 5", [3, 4, 5]),  False),
        ("validate missing", RF.validate_equation("3 + 4",     [3, 4, 5]),  False),
        ("eval arithmetic",  RF.evaluate_equation("3 + 4 * 5"),             23),
        ("eval bad chars",   RF.evaluate_equation("import os"),             None),
        ("extract none",     RF.extract_solution("no tags here"),          None),
    ]
    for name, got, expected in helper_checks:
        ok = got == expected
        flag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{flag}] {name:22s} expected={expected!r:<6} got={got!r}")

    print("-" * 72)
    print(f"REWARD TESTS: {passed} passed, {failed} failed")
    # Document a known reward-spec quirk we observed in the source (not a failure):
    print("note: evaluate_equation's allow-list permits '**' (power) and unary +/-;"
          " countdown task construction only emits + - * /, so this never fires in practice.")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
