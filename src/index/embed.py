import torch
from sentence_transformers import SentenceTransformer
from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("embed")

_model: SentenceTransformer | None = None

MODEL_NAME = cfg["embeddings"]["model_name"]
NORMALIZE = cfg["embeddings"]["normalize"]
QUERY_PREFIX = cfg["embeddings"]["query_instruction"]
DOC_PREFIX = cfg["embeddings"].get("document_instruction", "")
TRUST_REMOTE_CODE = cfg["embeddings"].get("trust_remote_code", False)
MAX_SEQ_LENGTH = cfg["embeddings"].get("max_seq_length")


def _get_device() -> str:
    # EMBED_DEVICE env var overrides config — used to force CPU for large batch
    # indexing, since nomic on Apple MPS can hang on long-sequence batches.
    import os
    override = os.getenv("EMBED_DEVICE")
    if override:
        return override
    setting = cfg["embeddings"]["device"]
    if setting != "auto":
        return setting
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = _get_device()
        log.info("loading_embedding_model", model=MODEL_NAME, device=device)
        _model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=TRUST_REMOTE_CODE)
        if MAX_SEQ_LENGTH:
            _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_model()
    device = _get_device()
    batch_size = cfg["embeddings"]["batch_size_gpu"] if device != "cpu" else cfg["embeddings"]["batch_size_cpu"]
    # The doc prefix is applied only to the embedding input; stored text stays verbatim.
    inputs = [DOC_PREFIX + t for t in texts] if DOC_PREFIX else texts
    embeddings = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_model()
    prefixed = QUERY_PREFIX + query
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=NORMALIZE,
        show_progress_bar=False,
    )
    return embedding[0].tolist()
