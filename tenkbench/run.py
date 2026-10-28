"""Run the benchmark for one or more models and write results/.

    export ANTHROPIC_API_KEY=...
    python -m tenkbench.run claude-haiku-5-5 claude-sonnet-5-5

Each model's raw answers are cached in results/raw/<model>/, so re-running
only calls the API for companies not done yet.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import DATA, METRICS
from .extract import extract
from .guardrail import verified
from .retrieve import select
from .score import LABELS, accuracy_ci, classify

RESULTS = Path("results")


def run_model(model: str, client=None) -> list[dict]:
    """One row per (company, metric): the answer, its label, and the guardrail flag."""
    companies = json.loads((DATA / "companies.json").read_text())
    raw_dir = RESULTS / "raw" / model.replace(":", "_").replace("/", "_")   # safe folder name on every OS
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in companies:
        full = (DATA / "filings" / f"{c['ticker']}.txt").read_text()
        context = select(full)
        cache = raw_dir / f"{c['ticker']}.json"
        if cache.exists():
            ex = json.loads(cache.read_text())
        else:
            e = extract(context, c["name"], c["period_end"], model, client=client)
            ex = {"values": e.values, "input_tokens": e.input_tokens, "output_tokens": e.output_tokens, "seconds": e.seconds}
            cache.write_text(json.dumps(ex, indent=2))
        for metric, truth in c["truth"].items():
            pred = ex["values"].get(metric)
            rows.append({
                "model": model, "ticker": c["ticker"], "metric": metric,
                "true": truth["value"], "pred": pred,
                "label": classify(pred, truth, full),
                "flagged": not verified(pred, truth["unit"], context),
                "input_tokens": ex["input_tokens"], "output_tokens": ex["output_tokens"], "seconds": ex["seconds"],
            })
        print(f"{model} {c['ticker']}: {sum(r['label'] == 'correct' for r in rows if r['ticker'] == c['ticker'])}/{len(c['truth'])}")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_company = defaultdict(lambda: [0, 0])
    for r in rows:
        by_company[r["ticker"]][0] += r["label"] == "correct"
        by_company[r["ticker"]][1] += 1
    labels = Counter(r["label"] for r in rows)
    errors = [r for r in rows if r["label"] != "correct"]
    flagged_errors = sum(r["flagged"] for r in errors)
    false_flags = sum(r["flagged"] for r in rows if r["label"] == "correct")
    companies = {r["ticker"]: r for r in rows}.values()              # one row per company for usage
    n = len(rows)
    return {
        "facts": n,
        "accuracy": labels["correct"] / n,
        "accuracy_ci": accuracy_ci([tuple(v) for v in by_company.values()]),
        "labels": {k: labels.get(k, 0) for k in LABELS},
        "guardrail_catches": flagged_errors / len(errors) if errors else None,
        "guardrail_false_flags": false_flags,
        "accuracy_after_guardrail": (labels["correct"] - false_flags) / max(1, n - sum(r["flagged"] for r in rows)),
        "input_tokens_per_filing": sum(c["input_tokens"] for c in companies) / len(companies),
        "output_tokens_per_filing": sum(c["output_tokens"] for c in companies) / len(companies),
        "seconds_per_filing": sum(c["seconds"] for c in companies) / len(companies),
        "by_metric": {m: sum(r["label"] == "correct" for r in rows if r["metric"] == m) / max(1, sum(r["metric"] == m for r in rows)) for m in METRICS},
    }


def main(models: list[str]) -> None:
    all_rows, summaries = [], {}
    for model in models:
        rows = run_model(model)
        all_rows += rows
        summaries[model] = summarize(rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers.json").write_text(json.dumps(all_rows, indent=2))
    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Name at least one model, e.g. python -m tenkbench.run gemini:<model> ollama:qwen2.5:7b")
    main(sys.argv[1:])
