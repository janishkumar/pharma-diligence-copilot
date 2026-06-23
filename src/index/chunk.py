import hashlib
from src.config import cfg
from src.logging_setup import get_logger
from src.schemas import Chunk, Filing

log = get_logger("chunk")

CHUNK_SIZE = cfg["chunking"]["chunk_size_tokens"]
CHUNK_OVERLAP = cfg["chunking"]["chunk_overlap_tokens"]
SPLITTER = cfg["chunking"]["sentence_splitter"]


def _get_tokenizer():
    from transformers import AutoTokenizer
    model_name = cfg["embeddings"]["model_name"]
    return AutoTokenizer.from_pretrained(model_name)


def _sent_tokenize(text: str) -> list[str]:
    if SPLITTER == "pysbd":
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return seg.segment(text)
    import nltk
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)


def _chunk_section(text: str, tokenizer) -> list[str]:
    sentences = _sent_tokenize(text)
    chunks = []
    current_tokens = []
    current_len = 0

    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if current_len + len(sent_ids) > CHUNK_SIZE and current_tokens:
            chunks.append(tokenizer.decode(current_tokens))
            # keep overlap
            overlap_tokens = current_tokens[-CHUNK_OVERLAP:]
            current_tokens = overlap_tokens + sent_ids
            current_len = len(current_tokens)
        else:
            current_tokens.extend(sent_ids)
            current_len += len(sent_ids)

    if current_tokens:
        chunks.append(tokenizer.decode(current_tokens))

    return chunks


def _make_chunk_id(accession: str, section: str, index: int, text: str) -> str:
    raw = f"{accession}|{section}|{index}|{text[:200]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def chunk_filing(filing: Filing) -> list[Chunk]:
    tokenizer = _get_tokenizer()
    chunks = []

    for section, text in filing.parsed_sections.items():
        if not text or not text.strip():
            continue

        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count <= CHUNK_SIZE:
            section_chunks = [text.strip()]
        else:
            section_chunks = _chunk_section(text.strip(), tokenizer)

        for idx, chunk_text in enumerate(section_chunks):
            chunk_id = _make_chunk_id(filing.accession_number, section, idx, chunk_text)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                company=filing.company_name,
                ticker=filing.ticker,
                cik=filing.cik,
                fiscal_year=filing.fiscal_year,
                fiscal_year_end_date=filing.fiscal_year_end_date,
                form_type=filing.form_type,
                section=section,
                accession_number=filing.accession_number,
                source_url=filing.source_url,
                chunk_index_in_section=idx,
            ))

    log.info("chunked", ticker=filing.ticker, total_chunks=len(chunks))
    return chunks
