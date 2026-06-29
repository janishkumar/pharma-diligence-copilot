"""Token-aware, sentence-boundary chunking.

The embedding tokenizer (bge-small-en-v1.5) is UNCASED, so decoding token ids
back to text lowercases and mangles punctuation ("U.S." -> "u. s.", "12%" ->
"12 %"). The stored chunk text is what is shown in citations and fed to the LLM
as context, so it must stay verbatim. We therefore slice the ORIGINAL text by
sentence boundaries and use the tokenizer only to MEASURE token counts.
"""
import hashlib

from src.config import cfg
from src.logging_setup import get_logger
from src.schemas import Chunk, Filing

log = get_logger("chunk")

CHUNK_SIZE = cfg["chunking"]["chunk_size_tokens"]
CHUNK_OVERLAP = cfg["chunking"]["chunk_overlap_tokens"]
SPLITTER = cfg["chunking"]["sentence_splitter"]

# Joining sentences can tokenize slightly higher than the sum of per-sentence
# counts, so enforce a hard ceiling just above the target as a final guard. The
# ceiling stays well under the embedding model's max_seq_length so nothing is
# truncated at embed time.
MODEL_MAX_TOKENS = cfg["embeddings"].get("max_seq_length", 512)
SAFE_MAX_TOKENS = min(MODEL_MAX_TOKENS - 12, int(CHUNK_SIZE * 1.05))
# Chunks below this are context-free fragments; merge them into a neighbor.
MIN_CHUNK_TOKENS = 32

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(cfg["embeddings"]["model_name"])
    return _tokenizer


def _sent_tokenize(text: str) -> list[str]:
    if SPLITTER == "pysbd":
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s for s in seg.segment(text) if s.strip()]
    import nltk
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)
    return [s for s in sents if s.strip()]


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tail_slice(text: str, n_tokens: int, tokenizer) -> str:
    """Return the text span of approximately the last n_tokens of `text`,
    sliced from the original via offset mapping (casing/punctuation preserved)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if len(offsets) <= n_tokens or not offsets:
        return text
    start_char = offsets[-n_tokens][0]
    return text[start_char:].strip()


def _hard_split(sentence: str, tokenizer) -> list[str]:
    """Split a single over-long sentence (e.g. a flattened table) into
    token-sized windows, slicing the ORIGINAL text via offset mapping so casing
    and punctuation are preserved exactly."""
    enc = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return [sentence]
    stride = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    pieces = []
    i = 0
    n = len(offsets)
    while i < n:
        window = offsets[i:i + CHUNK_SIZE]
        start_char = window[0][0]
        end_char = window[-1][1]
        piece = sentence[start_char:end_char].strip()
        if piece:
            pieces.append(piece)
        i += stride
    return pieces


def _chunk_section(text: str, tokenizer) -> list[str]:
    sentences = _sent_tokenize(text)
    if not sentences:
        return []

    # Batch-measure token counts for all sentences at once (fast tokenizer).
    encoded = tokenizer(sentences, add_special_tokens=False)["input_ids"]
    counts = [len(ids) for ids in encoded]

    chunks: list[str] = []
    buf: list[str] = []
    buf_counts: list[int] = []
    buf_total = 0

    def flush():
        if buf:
            chunks.append(" ".join(buf).strip())

    for sent, cnt in zip(sentences, counts):
        # A single sentence larger than the budget: emit current buffer, then
        # hard-split the giant sentence on its own.
        if cnt > CHUNK_SIZE:
            flush()
            buf, buf_counts, buf_total = [], [], 0
            chunks.extend(_hard_split(sent, tokenizer))
            continue

        if buf_total + cnt > CHUNK_SIZE and buf:
            flush()
            # Carry trailing sentences (~CHUNK_OVERLAP tokens) into the next chunk.
            overlap_sents, overlap_counts, overlap_total = [], [], 0
            for s, c in zip(reversed(buf), reversed(buf_counts)):
                if overlap_total + c > CHUNK_OVERLAP:
                    break
                overlap_sents.insert(0, s)
                overlap_counts.insert(0, c)
                overlap_total += c
            # If the boundary sentence alone exceeds CHUNK_OVERLAP, the reverse
            # walk carries nothing — seed the overlap with a token-tail slice of
            # that sentence so consecutive chunks always share context.
            if overlap_total == 0 and buf:
                tail = _tail_slice(buf[-1], CHUNK_OVERLAP, tokenizer)
                if tail:
                    overlap_sents = [tail]
                    overlap_counts = [_token_len(tail, tokenizer)]
                    overlap_total = overlap_counts[0]
            buf, buf_counts, buf_total = overlap_sents, overlap_counts, overlap_total

        buf.append(sent)
        buf_counts.append(cnt)
        buf_total += cnt

    flush()
    chunks = _merge_small([c for c in chunks if c], tokenizer)
    return _enforce_cap(chunks, tokenizer)


def _merge_small(chunks: list[str], tokenizer) -> list[str]:
    """Merge degenerate sub-threshold chunks into a neighbor so context-free
    fragments (e.g. an orphan overlap carry before a hard-split) are not embedded
    standalone. Single-chunk sections are left as-is."""
    if len(chunks) <= 1:
        return chunks
    out: list[str] = []
    for c in chunks:
        if out and _token_len(c, tokenizer) < MIN_CHUNK_TOKENS:
            out[-1] = (out[-1] + " " + c).strip()
        else:
            out.append(c)
    # A tiny leading fragment has no previous neighbor; fold it into the next.
    if len(out) >= 2 and _token_len(out[0], tokenizer) < MIN_CHUNK_TOKENS:
        out[1] = (out[0] + " " + out[1]).strip()
        out = out[1:]
    return out


def _enforce_cap(chunks: list[str], tokenizer) -> list[str]:
    """Final guard: hard-split any chunk whose true (joined) token length
    exceeds the safe ceiling, so nothing is silently truncated at embed time."""
    out: list[str] = []
    for c in chunks:
        if _token_len(c, tokenizer) > SAFE_MAX_TOKENS:
            out.extend(_hard_split(c, tokenizer))
        else:
            out.append(c)
    return out


def _make_chunk_id(accession: str, section: str, index: int, text: str) -> str:
    raw = f"{accession}|{section}|{index}|{text[:200]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def chunk_filing(filing: Filing) -> list[Chunk]:
    tokenizer = _get_tokenizer()
    chunks = []

    for section, text in filing.parsed_sections.items():
        if not text or not text.strip():
            continue

        text = text.strip()
        if _token_len(text, tokenizer) <= CHUNK_SIZE:
            section_chunks = [text]
        else:
            section_chunks = _chunk_section(text, tokenizer)

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
