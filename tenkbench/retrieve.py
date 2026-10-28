"""Pick the parts of a 10-K that contain the financial statements.

A full 10-K runs 200,000 to 1,200,000 characters, too long and too costly to
send whole. The numbers we want live in a few statement pages (income
statement, balance sheet, cash flow statement), so we split the text into
blocks and keep the highest-scoring ones.

This is a deliberately simple, transparent retriever (keyword and number
density, no embeddings). The benchmark measures it on its own: "context
recall" is the share of true values that survive into the selected text.
That separates two failure modes the final accuracy would otherwise blur:
the right page was never shown, versus the model misread the right page.
"""

from __future__ import annotations

import re

BLOCK_CHARS = 3000
TOP_K = 14

# Phrases that mark the primary statements and the specific line items we need.
KEYWORDS = {
    r"consolidated balance sheets?|statements? of financial (position|condition)": 6,
    r"consolidated statements? of (operations|income|earnings)": 6,
    r"consolidated statements? of cash flows?": 6,
    r"total assets": 3,
    r"total (stockholders|shareholders|shareowners)['’]? equity": 3,
    r"net (income|earnings)": 2,
    r"net cash (provided|from|used).{0,30}operating activities": 3,
    r"diluted": 2,
    r"cash and cash equivalents": 2,
    r"operating income|income from operations": 2,
    r"(total )?(net )?(revenues?|sales)": 1,
    r"in millions|in thousands": 2,
}
NUMBER = re.compile(r"\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?")


def blocks(text: str, size: int = BLOCK_CHARS) -> list[str]:
    """Split on line breaks into blocks of roughly `size` characters."""
    out, current, n = [], [], 0
    for line in text.splitlines():
        current.append(line)
        n += len(line) + 1
        if n >= size:
            out.append("\n".join(current))
            current, n = [], 0
    if current:
        out.append("\n".join(current))
    return out


def score(block: str) -> float:
    low = block.lower()
    kw = sum(w * min(len(re.findall(p, low)), 3) for p, w in KEYWORDS.items())
    numbers = len(NUMBER.findall(block))
    return kw + 0.15 * min(numbers, 80)          # statement pages are dense with numbers


def select(text: str, top_k: int = TOP_K) -> str:
    """The top-scoring blocks, kept in document order so statements read naturally."""
    bs = blocks(text)
    ranked = sorted(range(len(bs)), key=lambda i: score(bs[i]), reverse=True)[:top_k]
    return "\n\n[...]\n\n".join(bs[i] for i in sorted(ranked))
