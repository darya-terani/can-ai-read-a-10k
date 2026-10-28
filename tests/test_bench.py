import json
from types import SimpleNamespace

import pytest

from tenkbench import run as runmod
from tenkbench.guardrail import stated_scale, verified
from tenkbench.numbers import appears_in, renderings
from tenkbench.retrieve import blocks, select
from tenkbench.score import accuracy_ci, classify

DOC = """Annual report
CONSOLIDATED STATEMENTS OF OPERATIONS (In millions, except per-share amounts)
Total net sales | 416,161 | 391,035
Operating income | 133,050 | 123,216
Net income | 112,010 | 93,736
Diluted | 7.46 | 6.08
CONSOLIDATED BALANCE SHEETS (In millions)
Total assets | 359,241 | 364,980
"""
REV = {"value": 416_161_000_000, "unit": "USD", "other_periods": [391_035_000_000]}
EPS = {"value": 7.46, "unit": "USD/shares", "other_periods": [6.08]}


# ---- finding numbers in text ------------------------------------------------

def test_renderings_cover_common_scales():
    assert "416,161" in renderings(416_161_000_000, "USD")          # millions
    assert "32,019.7" in renderings(32_019_700_000, "USD")           # millions with a decimal
    assert renderings(7.46, "USD/shares") == ["7.46"]


def test_appears_in_respects_number_boundaries():
    assert appears_in("Total | 416,161 |", 416_161_000_000, "USD")
    assert not appears_in("Total | 1,416,161 |", 416_161_000_000, "USD")


# ---- error taxonomy ---------------------------------------------------------

@pytest.mark.parametrize("pred,label", [
    (416_161_000_000, "correct"),
    (416_200_000_000, "correct"),            # within 0.5%: rounding, not an error
    (None, "missing"),
    (-416_161_000_000, "sign"),
    (416_161_000, "unit"),                   # read millions as thousands
    (391_035_000_000, "wrong_period"),       # prior-year column
    (133_050_000_000, "wrong_line"),         # operating income reported as revenue
    (500_000_000_000, "hallucinated"),       # appears nowhere
])
def test_classify(pred, label):
    assert classify(pred, REV, DOC) == label


def test_classify_per_share():
    assert classify(7.46, EPS, DOC) == "correct"
    assert classify(6.08, EPS, DOC) == "wrong_period"


# ---- guardrail ----------------------------------------------------------------

def test_guardrail_reads_stated_scale():
    assert stated_scale(DOC) == 1e6


def test_guardrail_passes_correct_and_flags_unit_and_made_up_answers():
    assert verified(416_161_000_000, "USD", DOC)
    assert not verified(416_161_000, "USD", DOC)          # unit error: caught
    assert not verified(500_000_000_000, "USD", DOC)      # hallucination: caught
    assert verified(133_050_000_000, "USD", DOC)          # wrong line: a real number, NOT caught


# ---- retrieval ------------------------------------------------------------------

def test_retrieval_keeps_statement_pages():
    filler = "\n".join(f"Risk factor paragraph number {i} with no figures." for i in range(2000))
    text = filler + "\n" + DOC + "\n" + filler
    ctx = select(text, top_k=3)
    assert "416,161" in ctx and len(ctx) < len(text) / 10
    assert sum(len(b) for b in blocks(text)) >= len(text) - text.count("\n")


def test_bootstrap_ci_is_ordered():
    low, high = accuracy_ci([(7, 8), (8, 8), (6, 8), (5, 8)] * 5)
    assert 0 <= low <= high <= 1


# ---- end to end with a fake model (no API key needed) -------------------------

class FakeClient:
    """Answers revenue correctly, net income at the wrong scale, assets as null."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        answer = {"revenue": 416_161_000_000, "net_income": 112_010_000, "total_assets": None}
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=answer)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        )


def test_end_to_end(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "filings").mkdir(parents=True)
    (data / "filings" / "TEST.txt").write_text(DOC)
    truth = {
        "revenue": REV,
        "net_income": {"value": 112_010_000_000, "unit": "USD", "other_periods": []},
        "total_assets": {"value": 359_241_000_000, "unit": "USD", "other_periods": []},
    }
    (data / "companies.json").write_text(json.dumps([{"ticker": "TEST", "name": "Test Co", "period_end": "2025-09-27", "truth": truth}]))
    monkeypatch.setattr(runmod, "DATA", data)
    monkeypatch.setattr(runmod, "RESULTS", tmp_path / "results")

    rows = runmod.run_model("fake-model", client=FakeClient())
    labels = {r["metric"]: r["label"] for r in rows}
    assert labels == {"revenue": "correct", "net_income": "unit", "total_assets": "missing"}
    s = runmod.summarize(rows)
    assert s["accuracy"] == pytest.approx(1 / 3)
    assert s["guardrail_catches"] == pytest.approx(0.5)   # catches the unit error; "missing" isn't flagged
    assert s["guardrail_false_flags"] == 0


# ---- free providers, with fakes (no keys or local install needed) ---------------

def test_gemini_adapter_parses_json_and_usage():
    from tenkbench.providers import extract_gemini
    import tenkbench.providers as prov

    prov.GEMINI_PAUSE_SECONDS = 0
    captured = {}

    def generate_content(model, contents, config):
        captured.update(model=model, config=config)
        return SimpleNamespace(
            text='{"revenue": 416161000000, "net_income": "112,010,000,000", "cash": null}',
            usage_metadata=SimpleNamespace(prompt_token_count=900, candidates_token_count=40, thoughts_token_count=10),
        )

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    ex = extract_gemini("prompt", "gemini-test", client=fake)
    assert ex.values["revenue"] == 416_161_000_000
    assert ex.values["net_income"] == 112_010_000_000          # string with commas accepted
    assert ex.values["cash"] is None and ex.values["total_assets"] is None
    assert (ex.input_tokens, ex.output_tokens) == (900, 50)
    assert captured["config"].temperature == 0 and captured["config"].response_json_schema is not None


def test_ollama_adapter_sends_schema_and_large_context():
    from tenkbench.providers import extract_ollama

    sent = {}

    def post(url, payload):
        sent.update(url=url, payload=payload)
        return {"message": {"content": '{"eps_diluted": 7.46}'}, "prompt_eval_count": 1200, "eval_count": 30}

    ex = extract_ollama("prompt", "qwen2.5:7b", post=post, host="http://localhost:11434")
    assert ex.values["eps_diluted"] == 7.46
    assert sent["url"].endswith("/api/chat")
    assert sent["payload"]["options"]["num_ctx"] >= 32768      # default context would truncate the report
    assert sent["payload"]["format"]["properties"]["revenue"]["anyOf"]


def test_model_prefix_routes_to_provider():
    from tenkbench.extract import extract

    calls = []

    def post(url, payload):
        calls.append(payload["model"])
        return {"message": {"content": "{}"}}

    extract("ctx", "Co", "2025-12-31", "ollama:llama3.1:8b", client=post)
    assert calls == ["llama3.1:8b"]


def test_gemini_retries_when_servers_are_busy(monkeypatch):
    from google.genai import errors

    import tenkbench.providers as prov

    monkeypatch.setattr(prov.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def generate_content(model, contents, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return SimpleNamespace(text='{"revenue": 1}', usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1))

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    ex = prov.extract_gemini("prompt", "gemini-test", client=fake)
    assert calls["n"] == 2 and ex.values["revenue"] == 1
