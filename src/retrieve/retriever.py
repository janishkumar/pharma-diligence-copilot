from src.config import cfg
from src.index import embed, store
from src.logging_setup import get_logger

log = get_logger("retriever")

TOP_K_DENSE = cfg["retrieval"]["top_k_dense"]
HYBRID = cfg["retrieval"]["hybrid_enabled"]
BM25_TOP_K = cfg["retrieval"]["bm25_top_k"]
RRF_K = cfg["retrieval"]["rrf_k"]
# Bound the fused candidate set handed to the reranker, matching the dense budget.
RRF_TOP_K = max(TOP_K_DENSE, BM25_TOP_K)

_bm25 = None
_bm25_corpus: list[dict] = []


def _rrf_fuse(dense: list[dict], keyword: list[dict], k: int = RRF_K, top_k: int = RRF_TOP_K) -> list[dict]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    for rank, item in enumerate(dense):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = item
    for rank, item in enumerate(keyword):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked[:top_k]]


def _bm25_search(query: str, top_k: int) -> list[dict]:
    global _bm25, _bm25_corpus
    if _bm25 is None:
        _build_bm25_index()
    if not _bm25_corpus:
        return []
    from rank_bm25 import BM25Okapi
    import re
    import string
    from nltk.corpus import stopwords
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        stops = set(stopwords.words("english"))

    def tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
        return [w for w in text.split() if w not in stops]

    tokens = tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_bm25_corpus[i] for i in top_indices if scores[i] > 0]


def _build_bm25_index():
    global _bm25, _bm25_corpus
    import re, string
    from rank_bm25 import BM25Okapi
    from nltk.corpus import stopwords
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        stops = set(stopwords.words("english"))

    col = store.get_collection()
    all_docs = col.get(include=["documents", "metadatas"])
    _bm25_corpus = []

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
        return [w for w in text.split() if w not in stops]

    tokenized = []
    for doc_id, doc, meta in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
        _bm25_corpus.append({"chunk_id": doc_id, "text": doc, "metadata": meta, "score": 0.0})
        tokenized.append(tokenize(doc))

    # BM25Okapi divides by the average document length and raises ZeroDivisionError
    # on an empty/all-empty corpus. Leave the index unbuilt and let _bm25_search
    # short-circuit on the empty corpus instead.
    if not any(tokenized):
        _bm25 = None
        log.warning("bm25_index_empty", corpus_size=len(_bm25_corpus))
        return

    _bm25 = BM25Okapi(tokenized)
    log.info("bm25_index_built", corpus_size=len(_bm25_corpus))


def retrieve(query: str, filters: dict | None = None) -> list[dict]:
    q_emb = embed.embed_query(query)

    where = filters or None
    dense_results = store.query(q_emb, top_k=TOP_K_DENSE, where=where)

    if not dense_results and where:
        log.warning("empty_filtered_result", filters=filters)
        return []

    if not HYBRID:
        return dense_results

    keyword_results = _bm25_search(query, BM25_TOP_K)
    fused = _rrf_fuse(dense_results, keyword_results)
    return fused
