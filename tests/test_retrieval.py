"""Retrieval and chunk-level integrity tests — PRD Section 19.1 (chunk-level) + 19.2.

The chunk-level §19.1 checks deferred from M1 (deduplication, chunk-size
distribution, per-chunk metadata correctness) live here because they require
the built index. Tests that need the index skip cleanly when it is absent.
"""
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Logic tests (no index required)
# ---------------------------------------------------------------------------

def test_query_and_doc_prefixes_configured():
    """The embedding model uses asymmetric task prefixes; the query and document
    prefixes must both be set and must differ (e.g. nomic search_query/search_document)."""
    from src.index.embed import QUERY_PREFIX, DOC_PREFIX
    assert QUERY_PREFIX, "Query prefix must be set"
    # If the model uses a document prefix, it must differ from the query prefix
    # so passages and queries are encoded in their correct roles.
    if DOC_PREFIX:
        assert DOC_PREFIX != QUERY_PREFIX, "Doc and query prefixes must differ"


def test_empty_filter_does_not_widen():
    """When a filter returns zero results, retrieve() must return empty, not widen."""
    from unittest.mock import patch
    with patch("src.retrieve.retriever.store.query", return_value=[]):
        from src.retrieve.retriever import retrieve
        result = retrieve("test question", filters={"ticker": {"$eq": "FAKECO"}})
        assert result == [], "Empty filtered result must return empty, not silently widen"


def test_chunk_size_within_embedding_limit():
    """Configured chunk size must fit the embedding model's max_seq_length, else
    chunks are silently truncated at embed time. Regression guard for the
    chunk-size vs embedding-limit mismatch."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    max_seq = cfg["embeddings"].get("max_seq_length", 512)
    assert cfg["chunking"]["chunk_size_tokens"] < max_seq, (
        f"chunk_size_tokens must be < embedding max_seq_length ({max_seq})"
    )


# ---------------------------------------------------------------------------
# Index-backed tests (skip if the index has not been built)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed():
    from src.index import store
    col = store.get_collection()
    if col.count() == 0:
        pytest.skip("No index built; run scripts/02_index.py")
    data = col.get(include=["documents", "metadatas"])
    return data


def test_index_nontrivial(indexed):
    assert len(indexed["ids"]) > 1000, f"Suspiciously few chunks: {len(indexed['ids'])}"


def test_no_duplicate_chunk_ids(indexed):
    """Deduplication (§19.1): chunk ids must be unique."""
    ids = indexed["ids"]
    assert len(ids) == len(set(ids)), "Duplicate chunk ids in the index"


def test_chunk_metadata_complete(indexed):
    """Per-chunk metadata correctness (§19.1): every chunk carries the fields
    the citation contract (§14.2) depends on."""
    required = {"company", "ticker", "cik", "fiscal_year", "section",
                "accession_number", "source_url"}
    for cid, meta in zip(indexed["ids"], indexed["metadatas"]):
        missing = required - set(meta.keys())
        assert not missing, f"chunk {cid}: missing metadata {missing}"
        assert str(meta["source_url"]).startswith("https://"), f"chunk {cid}: bad source_url"
        assert meta["ticker"], f"chunk {cid}: empty ticker"


def test_all_companies_indexed(indexed):
    """Corpus completeness at chunk level: every config ticker has chunks."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    indexed_tickers = {m["ticker"] for m in indexed["metadatas"]}
    missing = set(cfg["corpus"]["tickers"]) - indexed_tickers
    assert not missing, f"Tickers absent from index: {missing}"


def test_mandatory_sections_indexed(indexed):
    """Each company must contribute its mandatory sections to the index."""
    by_ticker = {}
    for m in indexed["metadatas"]:
        by_ticker.setdefault(m["ticker"], set()).add(m["section"])
    for ticker, sections in by_ticker.items():
        for sec in ("Item 1", "Item 1A", "Item 7"):
            assert sec in sections, f"{ticker}: no chunks for mandatory {sec}"


def test_chunk_text_is_verbatim_not_lowercased(indexed):
    """Regression guard for the uncased-tokenizer decode bug: stored chunk text
    must preserve original casing/punctuation, not be a lowercased token decode.
    If every chunk were all-lowercase, the bug is back."""
    docs = indexed["documents"]
    sample = docs[: min(200, len(docs))]
    with_upper = sum(1 for d in sample if any(c.isupper() for c in d))
    assert with_upper > 0.9 * len(sample), (
        "Most chunks lack uppercase letters — text may be a lowercased token decode"
    )
    # token-decode of the uncased tokenizer mangles abbreviations to "u. s.";
    # the verbatim source keeps "U.S." Check the canonical case survives somewhere.
    assert any("U.S." in d or "FDA" in d for d in sample), (
        "No 'U.S.'/'FDA' in any sampled chunk — casing/punctuation may be mangled"
    )


def test_chunk_size_distribution(indexed):
    """Chunk-size distribution (§19.1): no stored chunk exceeds the embedding
    model's max_seq_length (would be truncated at embed time)."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    max_seq = cfg["embeddings"].get("max_seq_length", 512)
    from src.index.chunk import _get_tokenizer
    tok = _get_tokenizer()
    docs = indexed["documents"]
    sample = docs[:: max(1, len(docs) // 300)]  # ~300 evenly-spaced chunks
    oversize = [d for d in sample if len(tok.encode(d, add_special_tokens=False)) > max_seq]
    assert not oversize, f"{len(oversize)} sampled chunks exceed {max_seq} tokens"


def test_retrieval_is_relevant():
    """End-to-end semantic sanity: a risk-factors query should surface Item 1A."""
    from src.index import store, embed
    col = store.get_collection()
    if col.count() == 0:
        pytest.skip("No index built")
    results = store.query(embed.embed_query("What are the principal risk factors?"), top_k=5)
    assert results, "Query returned no results"
    sections = [r["metadata"]["section"] for r in results]
    assert "Item 1A" in sections, f"Risk-factor query did not surface Item 1A; got {sections}"
