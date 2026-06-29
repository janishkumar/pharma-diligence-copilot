"""Cross-encoder reranking with a thread-safe timeout.

The timeout must work when called from a server worker thread (FastAPI/uvicorn,
Streamlit), so it uses concurrent.futures rather than signal.SIGALRM (which only
works on the main thread and silently raised ValueError off it, bypassing the
reranker on every API/UI call).
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("rerank")

RERANK_ENABLED = cfg["rerank"]["enabled"]
RERANK_MODEL = cfg["rerank"]["model_name"]
TOP_N = cfg["rerank"]["top_n"]
TIMEOUT = cfg["rerank"]["timeout_sec"]
SCORE_THRESHOLD = cfg["rerank"]["score_threshold"]

_reranker = None
_reranker_disabled = False
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rerank")


def _load_reranker():
    global _reranker, _reranker_disabled
    if _reranker_disabled:
        return None
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
        log.info("reranker_loaded", model=RERANK_MODEL)
    except Exception as e:
        log.error("reranker_load_failed", error=str(e))
        _reranker_disabled = True
    return _reranker


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not RERANK_ENABLED or not candidates:
        return candidates[:TOP_N]

    reranker = _load_reranker()
    if reranker is None:
        log.warning("reranker_unavailable_falling_back_to_dense")
        return candidates[:TOP_N]

    pairs = [(query, c["text"]) for c in candidates]

    # Thread-safe deadline: run predict on a worker and bound the wait. Works in
    # any thread. A timed-out predict keeps running in the background but we
    # return the dense order rather than block the request.
    future = _executor.submit(reranker.predict, pairs)
    try:
        scores = future.result(timeout=TIMEOUT)
    except FuturesTimeout:
        log.warning("reranker_timeout", timeout_sec=TIMEOUT, fallback="dense_scores")
        return candidates[:TOP_N]
    except Exception as e:
        # A genuine reranker failure is NOT a timeout — log loudly so it is not
        # mistaken for the (acceptable) timeout fallback.
        log.error("reranker_failed", error=str(e), fallback="dense_scores")
        return candidates[:TOP_N]

    scored = [(float(score), chunk) for score, chunk in zip(scores, candidates) if score >= SCORE_THRESHOLD]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_N]]
