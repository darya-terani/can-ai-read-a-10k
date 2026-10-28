"""Find numbers in report text the way they are printed.

A 10-K usually prints figures in millions ("416,161" for $416,161,000,000),
sometimes thousands, and shows negatives in parentheses. These helpers
render a true value in each common scale so we can check whether it is
present in a piece of text.
"""

from __future__ import annotations

import re

SCALES = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


def renderings(value: float, unit: str) -> list[str]:
    """How a value could appear in the document."""
    if unit == "USD/shares":                         # per-share figures print as-is
        return [f"{abs(value):.2f}"]
    out = []
    for scale in SCALES.values():
        scaled = abs(value) / scale
        if scaled < 1:
            continue
        if abs(scaled - round(scaled)) < 1e-6:
            out.append(f"{round(scaled):,}")
        elif abs(scaled * 10 - round(scaled * 10)) < 1e-6 or scale == 1e9:
            out.append(f"{scaled:,.1f}")              # "32,019.7" (millions) or "416.2" (billions)
    return out


def appears_in(text: str, value: float, unit: str) -> bool:
    for s in renderings(value, unit):
        if re.search(rf"(?<![\d.,]){re.escape(s)}(?![\d]|,\d)", text):
            return True
    return False
