import signal
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

    try:
        def _handler(signum, frame):
            raise TimeoutError("reranker timeout")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(TIMEOUT)
        try:
            scores = reranker.predict(pairs)
        finally:
            signal.alarm(0)
    except TimeoutError:
        log.warning("reranker_timeout", fallback="dense_scores")
        return candidates[:TOP_N]
    except Exception as e:
        log.warning("reranker_error", error=str(e), fallback="dense_scores")
        return candidates[:TOP_N]

    scored = [(score, chunk) for score, chunk in zip(scores, candidates) if score >= SCORE_THRESHOLD]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_N]]
