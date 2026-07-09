"""FastAPI service exposing the RAG pipeline.

Models (embedder + reranker) are pre-warmed at startup so the first request
does not pay the ~25s cold-load inside a client's timeout window. ALL error
paths — including 413/422/500/502 — return the PRD §14.2 ErrorResponse contract
(error_code/message/request_id); internal exception details and submitted input
are logged server-side, never echoed to clients.
"""
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from src.logging_setup import get_logger  # noqa: E402
from src.pipeline import ask  # noqa: E402
from src.schemas import AskRequest, AskResponse, ErrorResponse  # noqa: E402

log = get_logger("api")

# Whitelisted filter keys, their scalar type, and the max $in list length.
FILTER_SPEC = {"ticker": (str, 12), "fiscal_year": (int, 10)}
MAX_BODY_BYTES = 32 * 1024  # /ask bodies are a question + tiny filters


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    log.info("prewarm_start")
    try:
        from src.index import embed
        embed.embed_query("warmup")  # loads the embedding model
    except Exception as e:
        # Without an embedder no request can be served — fail fast, but emit a
        # structured log first (a bare startup crash is invisible to log pipelines).
        log.error("prewarm_embedder_failed", error=type(e).__name__, detail=str(e)[:200])
        raise
    try:
        from src.retrieve import rerank
        rerank._load_reranker()
        if rerank._reranker_disabled:
            log.warning("prewarm_reranker_unavailable", note="serving with dense-only fallback")
    except Exception as e:
        log.warning("prewarm_reranker_failed", error=str(e)[:200], note="dense-only fallback")
    log.info("prewarm_done", seconds=round(time.time() - t0, 1))
    yield


app = FastAPI(title="Pharma Diligence Copilot", version="1.0", lifespan=lifespan)


def _error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error_code=code, message=message, request_id=request_id).model_dump(),
    )


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        request_id = uuid.uuid4().hex[:12]
        log.warning("body_too_large", request_id=request_id, bytes=int(length))
        return _error(413, "payload_too_large",
                      f"Request body exceeds {MAX_BODY_BYTES // 1024}KB.", request_id)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    # Log field locations/types only — never the submitted input values.
    summary = "; ".join(
        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', 'invalid')}"
        for e in exc.errors()[:5]
    )
    log.warning("validation_failed", request_id=request_id, errors=summary[:300])
    return _error(422, "validation_error", summary or "Invalid request.", request_id)


def _validate_filters(filters: dict | None) -> tuple[dict | None, str | None]:
    """Strictly validate external filters: whitelisted keys, scalar or
    {$eq}/{$in} shapes, correct value types, bounded list sizes. Returns
    (clean_filters, error_message). Unknown keys are dropped; malformed values
    for known keys are rejected (not silently ignored)."""
    if not filters:
        return None, None
    clean: dict = {}
    for key, cond in filters.items():
        if key not in FILTER_SPEC:
            continue
        typ, max_in = FILTER_SPEC[key]
        if isinstance(cond, typ) and not isinstance(cond, bool):
            clean[key] = {"$eq": cond}
        elif isinstance(cond, dict) and set(cond) == {"$eq"} and isinstance(cond["$eq"], typ):
            clean[key] = cond
        elif (isinstance(cond, dict) and set(cond) == {"$in"}
              and isinstance(cond["$in"], list)
              and 0 < len(cond["$in"]) <= max_in
              and all(isinstance(v, typ) and not isinstance(v, bool) for v in cond["$in"])):
            clean[key] = cond
        else:
            return None, (f"invalid filter for '{key}': expected a {typ.__name__}, "
                          f"{{\"$eq\": ...}}, or {{\"$in\": [...]}} with at most {max_in} values")
    return (clean or None), None


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    request_id = uuid.uuid4().hex[:12]

    filters, filter_error = _validate_filters(req.filters)
    if filter_error:
        log.warning("filter_rejected", request_id=request_id, reason=filter_error)
        return _error(422, "invalid_filter", filter_error, request_id)

    try:
        resp = ask(req.question, filters=filters)
    except Exception as e:
        # Internal detail goes to logs only; the client gets the contract shape.
        log.error("ask_failed", request_id=request_id, error=type(e).__name__, detail=str(e)[:200])
        return _error(500, "internal_error", "The server failed to process the question.", request_id)

    if resp.error:
        log.error("generation_error", request_id=request_id, detail=resp.error[:200])
        return _error(502, "generation_failed",
                      "The answer generator is currently unavailable. Please retry.", request_id)

    return resp


@app.get("/health")
def health():
    """Readiness, not just liveness: reports index and reranker state."""
    from src.index import store
    from src.retrieve import rerank
    try:
        chunks = store.chunk_count()
    except Exception:
        chunks = -1
    reranker_ok = (rerank._reranker is not None) and not rerank._reranker_disabled
    status = "ok" if chunks > 0 else "degraded"
    return {"status": status, "index_chunks": chunks, "reranker_available": reranker_ok}
