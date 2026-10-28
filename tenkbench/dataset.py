"""Build the benchmark: for each company, its latest 10-K text and the true
values of eight financial metrics.

Ground truth comes from XBRL, the machine-readable data companies file with
the SEC alongside each report. Each XBRL fact records which filing it came
from (accession number) and which period it covers, so we take only facts
from the same filing we give the model, for the fiscal year that filing reports.

    python -m tenkbench.dataset   # writes data/companies.json and data/filings/*.txt
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Darya Terani darya.terani1@gmail.com"}  # the SEC requires a contact in the user agent
DATA = Path("data")

TICKERS = [
    # technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "ORCL", "ADBE", "CSCO", "INTC", "CRM",
    # consumer
    "AMZN", "WMT", "COST", "HD", "NKE", "MCD", "KO", "PEP", "PG", "SBUX",
    # health care
    "JNJ", "PFE", "UNH", "MRK", "ABBV",
    # financials
    "JPM", "BAC", "GS", "V", "MA",
    # industrials and energy
    "CAT", "BA", "GE", "COP", "CVX", "UPS", "DE",
    # media and telecom
    "DIS", "NFLX", "VZ",
]

# Each metric: candidate us-gaap tags (companies don't all use the same one),
# and whether it covers a period (income/cash flow) or a point in time (balance sheet).
METRICS = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "duration"),
    "operating_income": (["OperatingIncomeLoss"], "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "duration"),
    "eps_diluted": (["EarningsPerShareDiluted"], "duration"),
    "total_assets": (["Assets"], "instant"),
    "stockholders_equity": (["StockholdersEquity"], "instant"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "duration"),
}


def get(url: str) -> requests.Response:
    time.sleep(0.15)  # stay well under the SEC's 10 requests per second
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r


def latest_10k(cik: int) -> dict:
    recent = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "filed": recent["filingDate"][i],
                "period_end": recent["reportDate"][i],
                "document": recent["primaryDocument"][i],
            }
    raise LookupError(f"No 10-K for CIK {cik}")


def ground_truth(facts: dict, accession: str, period_end: str) -> dict:
    """True value of each metric for the fiscal year ending `period_end`, taken
    only from facts filed in this exact 10-K."""
    gaap = facts["facts"].get("us-gaap", {})
    out = {}
    for metric, (tags, kind) in METRICS.items():
        for tag in tags:
            if tag not in gaap:
                continue
            for unit, rows in gaap[tag]["units"].items():
                for row in rows:
                    if row.get("accn") != accession or row.get("end") != period_end:
                        continue
                    if kind == "duration":
                        days = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days
                        if not 350 <= days <= 380:      # full fiscal year only, not a quarter
                            continue
                    # Values of the same line item for OTHER periods in this filing
                    # (prior years, quarters). A prediction equal to one of these
                    # means the model read the wrong column, a distinct error type.
                    others = sorted({r["val"] for r in rows if r.get("accn") == accession and r["val"] != row["val"]})
                    out[metric] = {"value": row["val"], "unit": unit, "tag": tag, "other_periods": others}
                    break
                if metric in out:
                    break
            if metric in out:
                break
    return out


def html_to_text(html: bytes) -> str:
    """Readable text that keeps table rows on single lines, so a statement row
    like 'Total net sales 416,161 391,035' stays together."""
    import warnings

    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for hidden in soup.select("ix\\:header, [style*='display:none']"):
        hidden.decompose()
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        row.replace_with(" | ".join(c for c in cells if c) + "\n")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def build() -> None:
    (DATA / "filings").mkdir(parents=True, exist_ok=True)
    tickers = {v["ticker"]: v["cik_str"] for v in get("https://www.sec.gov/files/company_tickers.json").json().values()}
    companies = []
    for t in TICKERS:
        try:
            cik = tickers[t]
            filing = latest_10k(cik)
            facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
            truth = ground_truth(facts, filing["accession"], filing["period_end"])
            acc = filing["accession"].replace("-", "")
            html = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{filing['document']}").content
            text = html_to_text(html)
            (DATA / "filings" / f"{t}.txt").write_text(text)
            companies.append({"ticker": t, "cik": cik, "name": facts.get("entityName"), **filing, "truth": truth})
            print(f"{t:6s} {filing['period_end']}  metrics {len(truth)}/8  text {len(text):,} chars")
        except Exception as err:  # keep going; report what failed
            print(f"{t:6s} FAILED: {err}")
    (DATA / "companies.json").write_text(json.dumps(companies, indent=2))


if __name__ == "__main__":
    build()
