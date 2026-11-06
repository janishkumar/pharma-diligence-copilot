"""Cross-encoder reranking with a thread-safe timeout.

The timeout must work when called from a server worker thread (FastAPI/uvicorn,
Streamlit), so it uses concurrent.futures rather than signal.SIGALRM (which only
works on the main thread and silently raised ValueError off it, bypassing the
reranker on every API/UI call).

Concurrency: predict() is serialized by a lock (torch modules are not safe for
concurrent forward passes), acquired with a timeout so a hung predict degrades
later requests to the dense fallback instead of wedging them forever. Each call
gets its own single-use executor — a shared 1-worker pool would be permanently
occupied by one hung predict, silently disabling reranking for the process
lifetime. A hung call leaks one thread (bounded, logged), nothing more.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("rerank")

RERANK_ENABLED = cfg["rerank"]["enabled"]
RERANK_MODEL = cfg["rerank"]["model_name"]
TOP_N = cfg["rerank"]["top_n"]
# The local fine-tuned model was trained on 3-chunk prompts and served with a
# 4096-token context (PRD 11.5); 8 chunks (~6-7k tokens) overflow it.
if os.getenv("GENERATOR_BACKEND_OVERRIDE") == "ollama":
    TOP_N = cfg["serving"].get("local_top_n", TOP_N)
TIMEOUT = cfg["rerank"]["timeout_sec"]
SCORE_THRESHOLD = cfg["rerank"]["score_threshold"]

_reranker = None
_reranker_disabled = False
_predict_lock = threading.Lock()


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

    # Serialize predicts; a hung predict keeps the lock until it actually ends,
    # so later requests fall back to dense (never a concurrent forward pass).
    if not _predict_lock.acquire(timeout=TIMEOUT):
        log.warning("reranker_busy", timeout_sec=TIMEOUT, fallback="dense_scores")
        return candidates[:TOP_N]

    def _run():
        # The lock is released HERE, when predict truly finishes — not by the
        # caller on timeout, which would allow a second concurrent predict.
        try:
            return reranker.predict(pairs)
        finally:
            _predict_lock.release()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rerank")
    try:
        future = executor.submit(_run)
        scores = future.result(timeout=TIMEOUT)
    except FuturesTimeout:
        log.warning("reranker_timeout", timeout_sec=TIMEOUT, fallback="dense_scores")
        return candidates[:TOP_N]
    except Exception as e:
        # A genuine reranker failure is NOT a timeout — log loudly so it is not
        # mistaken for the (acceptable) timeout fallback.
        log.error("reranker_failed", error=str(e), fallback="dense_scores")
        return candidates[:TOP_N]
    finally:
        executor.shutdown(wait=False)

    scored = [(float(score), chunk) for score, chunk in zip(scores, candidates) if score >= SCORE_THRESHOLD]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_N]]
