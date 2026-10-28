# Can AI Read a 10-K?

When banks and funds use AI models to pull figures out of financial reports, how often are the answers wrong, and *how* are they wrong? This benchmark tests models on eight figures from the latest annual reports (Form 10-K) of 40 large U.S. companies, scored against the companies' own machine-readable filings, and classifies every mistake.

> **Status:** pipeline built and tested; model results pending. Results tables below are filled from `results/summary.json`.

## Results

| Model | Accuracy (95% CI) | Unit errors | Wrong period | Wrong line | Hallucinated | Missing | Errors caught by guardrail | Cost per filing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _pending_ | | | | | | | | |

## The benchmark

**Companies.** 40 S&P 500 companies across technology, consumer, health care, financials, industrials, energy and media, each with its most recent 10-K.

**Figures.** Revenue, operating income, net income, diluted EPS, total assets, stockholders' equity, cash and equivalents, and operating cash flow: **295 facts** in all (not every company reports every line, e.g. banks don't report operating income).

**Ground truth, without hand-labeling.** Companies file their statements in XBRL, a machine-readable format, alongside each report. The answer key takes only facts from the *same filing* the model reads, for the *same fiscal year*, using full-year periods only. A validation step confirms all 295 true values appear in their report's text.

## Pipeline

```
10-K text ──► retrieval ──► model extraction ──► scoring + error taxonomy
 (200K–1.2M      (14 blocks,      (structured tool        (correct / missing / sign / unit /
  characters)     ~44K chars)      call, full-dollar        wrong period / wrong line /
                                   values or null)          hallucinated)
                                         │
                                         └──► guardrail: is the answer in the source text
                                              at the report's stated scale?
```

**Retrieval, measured on its own.** A full report is too long to send, so a transparent keyword-and-number-density retriever keeps the statement pages. It keeps **98.3% of true values** (290 of 295) in about a tenth of each report. Measuring this separately means a final error can be traced to either "the model never saw the page" or "the model misread the page."

**Extraction.** The model answers through a tool with one field per figure, so output is always structured. It must report **full dollar values** (converting "in millions"), because that conversion is exactly where unit errors happen, and it may answer **null**: a benchmark that forces an answer can't tell honest uncertainty from a confident wrong number.

**Error taxonomy.** Every wrong answer gets exactly one label:

| Label | Meaning |
| --- | --- |
| missing | Answered null |
| sign | Right size, wrong sign |
| unit | Off by a power of 1,000 (millions read as thousands) |
| wrong_period | Matches the same line item for another period in the filing (read the prior-year column) |
| wrong_line | A real number from the report, but the wrong one |
| hallucinated | Appears nowhere in the report |

**Guardrail.** A cheap automatic check renders each answer at the report's stated scale and looks for it in the excerpt. It should catch unit errors and hallucinations but not wrong-line or wrong-period errors; the benchmark measures exactly how many errors it catches and how many correct answers it wrongly flags.

**Uncertainty.** Accuracy gets a 95% bootstrap confidence interval that resamples *companies*, not facts, because one company's unusual formatting tends to break several of its figures at once.

## Run it

Models are chosen by name prefix, and two options cost nothing:

| Provider | Model name to pass | Cost | Setup |
| --- | --- | --- | --- |
| Google Gemini | `gemini:<model>` | Free tier | Key from [Google AI Studio](https://aistudio.google.com), then `export GEMINI_API_KEY=...` |
| Local, via Ollama | `ollama:<model>` | Free, runs on your computer | Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b` |
| Anthropic Claude | e.g. `claude-haiku-5-5` | Paid API | `export ANTHROPIC_API_KEY=...` |

```bash
pip install -r requirements.txt
python -m tenkbench.dataset                          # downloads 10-Ks and ground truth from SEC EDGAR
python -m tenkbench.run gemini:<model> ollama:qwen2.5:7b
pytest                                               # 19 tests; fake models, no keys needed
```

The Gemini run pauses a few seconds between requests to stay inside free-tier rate limits.

Answers are cached per company in `results/raw/`, so a re-run only calls the API for what's missing.

## Structure

```
tenkbench/
  dataset.py    companies, 10-K download, XBRL ground truth
  numbers.py    how values are printed (millions, thousands, decimals, parentheses)
  retrieve.py   statement-page retrieval
  extract.py    structured model extraction (Anthropic) and provider routing
  providers.py  free providers: Gemini free tier and local models via Ollama
  score.py      grading, error taxonomy, bootstrap CI
  guardrail.py  verify-in-source check
  run.py        runs models, writes results/
tests/          taxonomy, guardrail, retrieval and end-to-end tests
```

## Limitations

- 40 companies, one filing each. Enough to separate error types; small for fine model rankings, hence the confidence intervals.
- Retrieval was designed while looking at these same filings, so its 98.3% recall may be optimistic on new ones.
- XBRL tagging choices differ by company; the answer key uses the most common tag for each figure.

## Author

Darya Terani, B.S. Applied Statistics and B.A. Economics, UC Davis.
