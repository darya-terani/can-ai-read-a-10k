"""Ask a model to read the retrieved statement pages and report eight figures.

The model must answer through a tool whose schema lists every metric, so the
output is always structured. Two design choices matter for the evaluation:

  * We ask for FULL dollar values (416161000000), not "as printed" (416,161).
    Reports print in millions or thousands, and converting correctly is
    exactly the step where unit errors happen. We want to measure it.
  * The model may answer null. A benchmark that forces an answer can't tell
    honest "not found" from a confident wrong number, and in finance the
    second is far worse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .dataset import METRICS

DESCRIPTIONS = {
    "revenue": "Total revenue (or total net sales) for the fiscal year",
    "operating_income": "Operating income (income from operations) for the fiscal year",
    "net_income": "Net income attributable to the company for the fiscal year",
    "eps_diluted": "Diluted earnings per share for the fiscal year, in dollars per share",
    "total_assets": "Total assets at fiscal year end",
    "stockholders_equity": "Total stockholders' equity attributable to the company at fiscal year end",
    "cash": "Cash and cash equivalents at fiscal year end",
    "operating_cash_flow": "Net cash provided by (used in) operating activities for the fiscal year",
}

SYSTEM = """You extract figures from a company's annual report (Form 10-K).
Report each value as a full number in US dollars, converting any stated scale:
if a table says "in millions" and shows 416,161, report 416161000000.
Per-share values are reported as printed (e.g. 7.46).
Negative values (shown in parentheses) must be negative.
Report the fiscal year that ends on the date given, not a prior year.
If a figure is not in the text provided, use null. Never guess."""

TOOL = {
    "name": "report_figures",
    "description": "Report the requested figures from the 10-K.",
    "input_schema": {
        "type": "object",
        # anyOf (rather than a type list) is the form Anthropic, Gemini and Ollama all accept.
        "properties": {m: {"anyOf": [{"type": "number"}, {"type": "null"}], "description": d} for m, d in DESCRIPTIONS.items()},
        "required": list(METRICS),
    },
}


@dataclass
class Extraction:
    values: dict[str, float | None]
    input_tokens: int
    output_tokens: int
    seconds: float


def build_prompt(context: str, company: str, period_end: str) -> str:
    return (
        f"Company: {company}\nFiscal year ending: {period_end}\n\n"
        f"Excerpts from the 10-K:\n<report>\n{context}\n</report>"
    )


def clean_values(raw: dict) -> dict[str, float | None]:
    """Keep only the requested metrics; accept numbers sent as strings ("1,234")."""
    out = {}
    for m in METRICS:
        v = raw.get(m)
        if isinstance(v, str):
            try:
                v = float(v.replace(",", "").replace("$", ""))
            except ValueError:
                v = None
        out[m] = v if isinstance(v, (int, float)) else None
    return out


def extract(context: str, company: str, period_end: str, model: str, client=None) -> Extraction:
    """Route to the right provider by the model name's prefix:
    "gemini:<model>" (Google), "ollama:<model>" (local), anything else (Anthropic)."""
    prompt = build_prompt(context, company, period_end)
    if model.startswith("gemini:"):
        from .providers import extract_gemini

        return extract_gemini(prompt, model.split(":", 1)[1], client=client)
    if model.startswith("ollama:"):
        from .providers import extract_ollama

        return extract_ollama(prompt, model.split(":", 1)[1], post=client)

    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0,                     # same input, same output: makes runs comparable
        system=SYSTEM,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "report_figures"},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return Extraction(clean_values(block.input), response.usage.input_tokens, response.usage.output_tokens, time.time() - start)
