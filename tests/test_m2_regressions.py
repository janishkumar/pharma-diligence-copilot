"""Regression guards for the two HIGH bugs found in the M2 QA audit:
1. Reranker used signal.SIGALRM, which only works on the main thread, so it was
   silently bypassed under FastAPI/Streamlit worker threads.
2. Multi-key ChromaDB metadata filters (ticker + fiscal_year) crashed with HTTP 500
   because ChromaDB requires an explicit $and wrapper.
"""
import concurrent.futures

import pytest


def test_rerank_works_off_main_thread(monkeypatch):
    """Reranker must run from a worker thread (no signal.SIGALRM main-thread dep)."""
    from src.retrieve import rerank as rr

    class FakeReranker:
        def predict(self, pairs):
            # higher score for earlier docs, deterministic
            return [1.0 / (i + 1) for i in range(len(pairs))]

    monkeypatch.setattr(rr, "_load_reranker", lambda: FakeReranker())
    monkeypatch.setattr(rr, "RERANK_ENABLED", True)

    candidates = [{"chunk_id": str(i), "text": f"document {i}"} for i in range(6)]

    # Call rerank from a non-main thread — the old signal-based code raised
    # "ValueError: signal only works in main thread" here.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        result = ex.submit(rr.rerank, "a query", candidates).result(timeout=15)

    assert result, "rerank returned nothing off the main thread"
    assert len(result) <= rr.TOP_N


def test_rerank_does_not_use_signal():
    """The rerank module must not depend on the signal module for its timeout."""
    import src.retrieve.rerank as rr
    import inspect
    src = inspect.getsource(rr)
    assert "import signal" not in src and "signal.alarm" not in src, (
        "rerank still references signal — not thread-safe under a web server"
    )


def test_multikey_filter_wrapped_in_and():
    """Multi-condition filters must be wrapped in $and; single/logical untouched."""
    from src.index.store import _normalize_where

    multi = _normalize_where({"ticker": {"$eq": "PFE"}, "fiscal_year": {"$eq": 2025}})
    assert "$and" in multi
    assert {"ticker": {"$eq": "PFE"}} in multi["$and"]
    assert {"fiscal_year": {"$eq": 2025}} in multi["$and"]

    # single-key filter passes through unchanged
    assert _normalize_where({"ticker": {"$eq": "PFE"}}) == {"ticker": {"$eq": "PFE"}}
    # an already-logical filter is left alone
    assert _normalize_where({"$or": [{"a": 1}, {"b": 2}]}) == {"$or": [{"a": 1}, {"b": 2}]}
    assert _normalize_where(None) is None


def test_multikey_filter_query_does_not_crash():
    """End-to-end: a ticker+year filtered query must not 500; it returns a list
    scoped to the requested company (skips if no index)."""
    from src.index import store, embed
    if store.chunk_count() == 0:
        pytest.skip("No index built")
    results = store.query(
        embed.embed_query("total revenue"),
        top_k=3,
        where={"ticker": {"$eq": "PFE"}, "fiscal_year": {"$eq": 2025}},
    )
    assert isinstance(results, list)
    assert all(r["metadata"]["ticker"] == "PFE" for r in results)
