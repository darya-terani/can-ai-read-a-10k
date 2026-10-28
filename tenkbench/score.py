"""Grade each extracted figure and classify every mistake.

A single accuracy number hides what matters for deployment: some errors are
caught by a cheap check, others can only be caught by a human. So every
wrong answer gets exactly one label, checked in this order:

  correct         within 0.5% of the true value (per-share: within $0.01)
  missing         the model answered null
  sign            right size, wrong sign (a loss reported as a profit)
  unit            off by a power of 1,000: millions read as thousands, etc.
  wrong_period    equals this line item's value for another period in the
                  same filing (the model read the prior-year column)
  wrong_line      a number that IS in the document, but the wrong one
  hallucinated    a number that appears nowhere in the full report
"""

from __future__ import annotations

import random

from .numbers import appears_in

LABELS = ["correct", "missing", "sign", "unit", "wrong_period", "wrong_line", "hallucinated"]


def close(pred: float, true: float, unit: str) -> bool:
    if unit == "USD/shares":
        return abs(pred - true) <= 0.011
    return abs(pred - true) <= 0.005 * abs(true)


def classify(pred: float | None, truth: dict, full_text: str) -> str:
    true, unit = truth["value"], truth["unit"]
    if pred is None:
        return "missing"
    if close(pred, true, unit):
        return "correct"
    if close(-pred, true, unit):
        return "sign"
    if unit != "USD/shares":
        for k in (1e3, 1e6, 1e9):
            if close(pred * k, true, unit) or close(pred / k, true, unit):
                return "unit"
    if any(close(pred, other, unit) for other in truth.get("other_periods", [])):
        return "wrong_period"
    if appears_in(full_text, pred, unit):
        return "wrong_line"
    return "hallucinated"


def accuracy_ci(per_company: list[tuple[int, int]], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for pooled accuracy, resampling companies (not facts):
    one company's odd formatting tends to break several of its figures at once."""
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = rng.choices(per_company, k=len(per_company))
        correct = sum(c for c, _ in sample)
        total = sum(t for _, t in sample)
        stats.append(correct / total if total else 0.0)
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]
