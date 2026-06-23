import chromadb
from src.config import cfg, PROJECT_ROOT
from src.logging_setup import get_logger
from src.schemas import Chunk

log = get_logger("store")

PERSIST_DIR = str(PROJECT_ROOT / cfg["vector_store"]["persist_dir"])
COLLECTION_NAME = cfg["vector_store"]["collection_name"]

_client: chromadb.PersistentClient | None = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]):
    col = get_collection()
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "company": c.company,
            "ticker": c.ticker,
            "cik": c.cik,
            "fiscal_year": c.fiscal_year,
            "fiscal_year_end_date": c.fiscal_year_end_date,
            "form_type": c.form_type,
            "section": c.section,
            "accession_number": c.accession_number,
            "source_url": c.source_url,
            "chunk_index_in_section": c.chunk_index_in_section,
        }
        for c in chunks
    ]
    documents = [c.text for c in chunks]
    col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    log.info("upserted", count=len(chunks))


def query(
    embedding: list[float],
    top_k: int = 20,
    where: dict | None = None,
) -> list[dict]:
    col = get_collection()
    kwargs = {"query_embeddings": [embedding], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    if not results["ids"] or not results["ids"][0]:
        return []
    out = []
    for i, doc_id in enumerate(results["ids"][0]):
        out.append({
            "chunk_id": doc_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })
    return out


def chunk_count() -> int:
    return get_collection().count()
