"""A cheap automatic check: can each answer be found in the source text?

Take the scale the statements declare ("in millions" is most common),
render the model's answer at that scale, and look for it in the excerpt the
model was given. An answer that can't be found is flagged for review.

This should catch unit errors (the right digits at the wrong scale won't
match) and hallucinations (the digits aren't there at all). It cannot catch
a real number copied from the wrong row or column. The benchmark measures
how many errors it flags and how many correct answers it flags by mistake.
"""

from __future__ import annotations

import re

SCALE_WORDS = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


def stated_scale(context: str) -> float:
    low = context.lower()
    counts = {w: len(re.findall(rf"in {w}", low)) for w in SCALE_WORDS}
    word = max(counts, key=counts.get)
    return SCALE_WORDS[word] if counts[word] else 1.0


def verified(pred: float | None, unit: str, context: str) -> bool:
    if pred is None:
        return True                        # nothing to verify; "missing" is already visible
    if unit == "USD/shares":
        candidates = [f"{abs(pred):.2f}"]
    else:
        scaled = abs(pred) / stated_scale(context)
        candidates = [f"{round(scaled):,}", f"{scaled:,.1f}"]
    return any(re.search(rf"(?<![\d.,]){re.escape(c)}(?![\d]|,\d)", context) for c in candidates)
