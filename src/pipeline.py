import hashlib
import time
from pathlib import Path

from src.config import cfg, PROJECT_ROOT
from src.generate import generator
from src.generate.prompt import ABSTENTION_STRING
from src.logging_setup import get_logger
from src.retrieve import retriever, rerank
from src.schemas import AskResponse, Citation, TimingBreakdown

log = get_logger("pipeline")

MODEL_VERSION = f"{cfg['generation']['backend']}/{cfg['generation']['model']}"


def _corpus_snapshot_hash() -> str:
    chroma_dir = PROJECT_ROOT / cfg["vector_store"]["persist_dir"]
    if not chroma_dir.exists():
        return "no-index"
    h = hashlib.md5()
    for p in sorted(chroma_dir.rglob("*")):
        if p.is_file():
            h.update(p.stat().st_mtime_ns.to_bytes(8, "little"))
    return h.hexdigest()[:12]


def _parse_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    citations = []
    for i, chunk in enumerate(chunks, 1):
        marker = f"[{i}]"
        if marker in answer:
            meta = chunk.get("metadata", {})
            citations.append(Citation(
                n=i,
                company=meta.get("company", ""),
                fiscal_year=int(meta.get("fiscal_year", 0)),
                section=meta.get("section", ""),
                chunk_id=chunk.get("chunk_id", ""),
                source_url=meta.get("source_url", ""),
            ))
    return citations


def ask(question: str, filters: dict | None = None) -> AskResponse:
    t0 = time.time()

    t_ret = time.time()
    candidates = retriever.retrieve(question, filters=filters)
    retrieval_ms = int((time.time() - t_ret) * 1000)

    t_rer = time.time()
    top_chunks = rerank.rerank(question, candidates)
    rerank_ms = int((time.time() - t_rer) * 1000)

    t_gen = time.time()
    if not top_chunks:
        answer = ABSTENTION_STRING
    else:
        answer = generator.generate(question, top_chunks)
    generation_ms = int((time.time() - t_gen) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    abstained = answer.strip() == ABSTENTION_STRING

    citations = [] if abstained else _parse_citations(answer, top_chunks)

    log.info(
        "query",
        question=question[:120],
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        chunks_retrieved=len(candidates),
        chunks_used=len(top_chunks),
        abstained=abstained,
        citations=len(citations),
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        abstained=abstained,
        timing=TimingBreakdown(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
        model_version=MODEL_VERSION,
        corpus_snapshot_hash=_corpus_snapshot_hash(),
    )
