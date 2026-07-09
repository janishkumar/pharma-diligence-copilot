"""API contract tests — M5 (API-free: pipeline mocked, no model loads).

TestClient is used WITHOUT the context manager so the lifespan pre-warm
(which loads real models) never runs in tests.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api as api
from src.schemas import AskResponse, Citation, TimingBreakdown

client = TestClient(api.app)


def _fake_response(**overrides):
    base = dict(
        answer="Risks include pricing pressure [1].",
        citations=[Citation(n=1, company="Pfizer", fiscal_year=2025, section="Item 1A",
                            chunk_id="abc", source_url="https://sec.gov/x", snippet="Risks…")],
        abstained=False,
        timing=TimingBreakdown(retrieval_ms=10, rerank_ms=10, generation_ms=10, total_ms=30),
        model_version="anthropic/claude-sonnet-4-6",
        corpus_snapshot_hash="deadbeef",
        truncated=False,
    )
    base.update(overrides)
    return AskResponse(**base)


def test_health_reports_readiness():
    with patch("src.index.store.chunk_count", return_value=1582):
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_chunks"] == 1582
    assert "reranker_available" in body


def test_health_degraded_when_index_missing():
    with patch("src.index.store.chunk_count", return_value=0):
        r = client.get("/health")
    assert r.json()["status"] == "degraded"


def test_ask_success_contract():
    with patch.object(api, "ask", return_value=_fake_response()):
        r = client.post("/ask", json={"question": "What are Pfizer's risks?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Risks include")
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["snippet"]  # §9.8 source snippet present
    assert body["abstained"] is False
    assert "timing" in body and "model_version" in body


def test_ask_generation_error_returns_502_contract():
    err = _fake_response(answer="unavailable", citations=[], error="RateLimitError: boom")
    with patch.object(api, "ask", return_value=err):
        r = client.post("/ask", json={"question": "anything at all"})
    assert r.status_code == 502
    body = r.json()
    assert body["error_code"] == "generation_failed"
    assert body["request_id"]
    assert "boom" not in body["message"]  # internal detail must not leak


def test_ask_internal_exception_returns_500_without_leaking():
    with patch.object(api, "ask", side_effect=RuntimeError("secret internal path /Users/x")):
        r = client.post("/ask", json={"question": "anything at all"})
    assert r.status_code == 500
    body = r.json()
    assert body["error_code"] == "internal_error"
    assert "secret" not in body["message"] and "/Users" not in body["message"]


def test_422_uses_error_contract_and_does_not_echo_input():
    """Validation errors must return the ErrorResponse shape (error_code/
    message/request_id) and never echo the submitted input back."""
    marker = "ZZUNIQUEMARKERZZ" * 200  # oversized question containing a sentinel
    r = client.post("/ask", json={"question": marker})
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"error_code", "message", "request_id"}
    assert body["error_code"] == "validation_error"
    assert "ZZUNIQUEMARKERZZ" not in r.text  # input not reflected


def test_oversized_body_rejected_413():
    big = {"question": "valid question", "filters": {"ticker": {"$in": ["X" * 10] * 20000}}}
    r = client.post("/ask", json=big)
    assert r.status_code == 413
    assert r.json()["error_code"] == "payload_too_large"


def test_filter_validation():
    ok, err = api._validate_filters({"ticker": {"$eq": "PFE"}})
    assert ok == {"ticker": {"$eq": "PFE"}} and err is None
    # bare scalar is normalized to $eq
    ok, _ = api._validate_filters({"ticker": "PFE", "fiscal_year": 2025})
    assert ok == {"ticker": {"$eq": "PFE"}, "fiscal_year": {"$eq": 2025}}
    # unknown keys are dropped; empty result collapses to None
    ok, err = api._validate_filters({"$where": "1=1", "section": {"$eq": "Item 8"}})
    assert ok is None and err is None
    # wrong value type on a known key is REJECTED, not silently dropped
    _, err = api._validate_filters({"fiscal_year": {"$eq": "not-a-year"}})
    assert err is not None
    # unsupported operator on a known key is rejected
    _, err = api._validate_filters({"ticker": {"$ne": "PFE"}})
    assert err is not None
    # $in bounded: 13 tickers exceeds the 12-issuer corpus bound
    _, err = api._validate_filters({"ticker": {"$in": [f"T{i}" for i in range(13)]}})
    assert err is not None


def test_invalid_filter_returns_422_contract():
    r = client.post("/ask", json={"question": "valid question",
                                  "filters": {"ticker": {"$ne": "PFE"}}})
    assert r.status_code == 422
    body = r.json()
    assert body["error_code"] == "invalid_filter"
    assert body["request_id"]


def test_filters_are_validated_before_pipeline():
    captured = {}

    def spy(question, filters=None):
        captured["filters"] = filters
        return _fake_response()

    with patch.object(api, "ask", side_effect=spy):
        r = client.post("/ask", json={"question": "What are Pfizer's risks?",
                                      "filters": {"ticker": {"$eq": "PFE"}, "evil": {"$ne": 0}}})
    assert r.status_code == 200
    assert captured["filters"] == {"ticker": {"$eq": "PFE"}}
