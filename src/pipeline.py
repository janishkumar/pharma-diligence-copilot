import hashlib
import re
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

# Matches a bracketed citation group: [1], [1,2], [1, 2], [1-3], [1, 2-4].
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")
_ABSTENTION_CORE = ABSTENTION_STRING.rstrip(".").lower()


def _extract_marker_numbers(answer: str) -> list[int]:
    """All distinct citation numbers referenced anywhere in the answer, in order
    of first appearance. Handles [1][2], [1,2], [1, 2], and [1-3] ranges."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(answer):
        parts = re.split(r"\s*,\s*", group)
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                rng = range(int(a), int(b) + 1) if a.strip().isdigit() and b.strip().isdigit() else []
                nums = list(rng)
            elif part.strip().isdigit():
                nums = [int(part)]
            else:
                nums = []
            for n in nums:
                if n not in seen:
                    seen.append(n)
    return seen


_snapshot_hash_cache: str | None = None


def _is_abstention(answer: str) -> bool:
    """Robustly detect the abstention phrase as the leading content of the answer,
    tolerating surrounding quotes/markdown/whitespace and a trailing elaboration
    (the model often explains WHY it cannot answer)."""
    norm = answer.strip().lstrip('>#*-").“”\' ').lower()
    return norm.startswith(_ABSTENTION_CORE)


def _corpus_snapshot_hash() -> str:
    """Hash of the index, computed once per process (the index does not change
    mid-session). Memoized so we do not rglob the 85MB .chroma dir on every query."""
    global _snapshot_hash_cache
    if _snapshot_hash_cache is not None:
        return _snapshot_hash_cache
    chroma_dir = PROJECT_ROOT / cfg["vector_store"]["persist_dir"]
    if not chroma_dir.exists():
        return "no-index"
    h = hashlib.md5()
    for p in sorted(chroma_dir.rglob("*")):
        if p.is_file():
            h.update(p.stat().st_mtime_ns.to_bytes(8, "little"))
    _snapshot_hash_cache = h.hexdigest()[:12]
    return _snapshot_hash_cache


def _parse_citations(answer: str, chunks: list[dict]) -> tuple[list[Citation], list[int]]:
    """Return (citations, out_of_range_markers). Citations are built for every
    in-range marker referenced (any grouping style), in order of appearance.
    Out-of-range markers (e.g. [10] when 5 chunks exist) are reported, not silently
    dropped — they signal an ungrounded/hallucinated reference."""
    referenced = _extract_marker_numbers(answer)
    citations = []
    out_of_range = []
    for n in referenced:
        if 1 <= n <= len(chunks):
            meta = chunks[n - 1].get("metadata", {})
            citations.append(Citation(
                n=n,
                company=meta.get("company", ""),
                fiscal_year=int(meta.get("fiscal_year", 0)),
                section=meta.get("section", ""),
                chunk_id=chunks[n - 1].get("chunk_id", ""),
                source_url=meta.get("source_url", ""),
            ))
        else:
            out_of_range.append(n)
    return citations, out_of_range


def ask(question: str, filters: dict | None = None) -> AskResponse:
    t0 = time.time()

    t_ret = time.time()
    candidates = retriever.retrieve(question, filters=filters)
    retrieval_ms = int((time.time() - t_ret) * 1000)

    t_rer = time.time()
    top_chunks = rerank.rerank(question, candidates)
    rerank_ms = int((time.time() - t_rer) * 1000)

    t_gen = time.time()
    truncated = False
    gen_error = None
    if not top_chunks:
        answer = ABSTENTION_STRING
    else:
        gen = generator.generate(question, top_chunks)
        answer = gen.text
        truncated = gen.truncated
        gen_error = gen.error
    generation_ms = int((time.time() - t_gen) * 1000)

    total_ms = int((time.time() - t0) * 1000)

    # A generation failure must be surfaced as an error, NOT disguised as a
    # content abstention (which would wrongly imply the corpus lacks the answer).
    if gen_error:
        log.error("generation_error_surfaced", error=gen_error[:160])
        return AskResponse(
            answer="The system was unable to generate an answer due to a generation error. Please try again.",
            citations=[],
            abstained=False,
            timing=TimingBreakdown(
                retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
                generation_ms=generation_ms, total_ms=total_ms,
            ),
            model_version=MODEL_VERSION,
            corpus_snapshot_hash=_corpus_snapshot_hash(),
            error=gen_error[:200],
        )

    abstained = _is_abstention(answer)
    if abstained:
        citations, out_of_range = [], []
    else:
        citations, out_of_range = _parse_citations(answer, top_chunks)

    # Groundedness guard: a non-abstaining answer with no valid citations is
    # unsupported. Refuse it rather than present ungrounded claims as fact.
    ungrounded = (not abstained) and (not citations)
    if ungrounded:
        log.warning("ungrounded_answer_suppressed", question=question[:120])
        answer = ABSTENTION_STRING
        abstained = True
        citations = []
    if out_of_range:
        log.warning("out_of_range_citations", markers=out_of_range, valid_range=len(top_chunks))

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
        ungrounded=ungrounded,
        truncated=truncated,
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
        truncated=truncated,
    )
