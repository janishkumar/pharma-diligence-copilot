import re
from datetime import datetime, timezone

from src.config import PROJECT_ROOT
from src.logging_setup import get_logger
from src.schemas import Filing

log = get_logger("parse")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MANDATORY_SECTIONS = {"Item 1", "Item 1A", "Item 7"}
OPTIONAL_SECTIONS = {"Item 7A", "Item 8", "Item 9A"}
TARGET_ITEMS = ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8", "Item 9A"]

# A captured section must clear this many chars to count as real content, not a
# stub/heading. Optional sections often legitimately cross-reference (e.g. a
# company folds market-risk into Item 7), so the floor is low; mandatory
# sections get a much higher floor in _validate_mandatory_sections.
MIN_SECTION_CHARS = 30
MIN_MANDATORY_CHARS = 500


# Running page header, e.g. "Pfizer Inc.2025 Form 10-K" or "2025 Form 10-K | "
_PAGE_HEADER = re.compile(r"(?im)^.{0,60}?20\d\d\s+form\s+10-?k.*$")
# Standalone "Table of Contents" navigation line
_TOC = re.compile(r"(?im)^[ \t]*table of contents[ \t]*$")
# A bare page-number line (1-4 digits, optionally bracketed by pipes/spaces)
_PAGE_NUM_LINE = re.compile(r"(?m)^[ \t]*\|?[ \t]*\d{1,4}[ \t]*\|?[ \t]*$")
# Leading stray page number before the first recognized ITEM header
_LEADING_PAGE_NUM = re.compile(r"(?is)^\s*\d{1,4}\s+(?=item\s)")


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip page furniture for clean chunking.

    Removes recurring EDGAR page-furniture (running 'Form 10-K' headers, bare
    page-number lines, 'Table of Contents' nav, and a stray leading page number
    before the item header) so chunks are not polluted by repetitive boilerplate
    (PRD Section 19.1 junk/near-duplicate criteria).
    """
    # normalize unicode spaces (nbsp, etc.) to plain space
    text = "".join(" " if (ch.isspace() and ch not in "\n\t") else ch for ch in text)
    text = _PAGE_HEADER.sub("", text)
    text = _TOC.sub("", text)
    text = _PAGE_NUM_LINE.sub("", text)
    text = _LEADING_PAGE_NUM.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing(ticker: str, filing_obj) -> Filing:
    """Parse an edgartools filing object into a Filing schema."""
    try:
        sections = _extract_sections_edgartools(filing_obj)
    except Exception as e:
        log.warning("edgartools_parse_failed", ticker=ticker, error=str(e))
        sections = _extract_sections_fallback(filing_obj)

    _validate_mandatory_sections(ticker, sections)

    form_type = getattr(filing_obj, "form", "10-K")
    accession = getattr(filing_obj, "accession_number", "") or getattr(filing_obj, "accession_no", "")

    # source_url: prefer the primary document URL, fall back to the filing index page
    source_url = (
        getattr(filing_obj, "filing_url", "")
        or getattr(filing_obj, "homepage_url", "")
        or getattr(filing_obj, "url", "")
    )

    filing = Filing(
        company_name=getattr(filing_obj, "company", ticker),
        ticker=ticker,
        cik=str(getattr(filing_obj, "cik", "")),
        form_type=form_type if form_type in ("10-K", "10-K/A") else "10-K",
        fiscal_year=_parse_fiscal_year(filing_obj),
        fiscal_year_end_date=str(getattr(filing_obj, "period_of_report", "")),
        filing_date=str(getattr(filing_obj, "filing_date", "")),
        accession_number=accession,
        accession_number_original=None,
        source_url=source_url,
        file_sha256="",  # filled in by ingest script after fetch
        ingested_at=datetime.now(timezone.utc).isoformat(),
        parsed_sections=sections,
    )
    return filing


def _extract_sections_edgartools(filing_obj) -> dict[str, str]:
    """Extract item sections via edgartools dict-style item access.

    `tenk[item_label]` returns the full item *text* (e.g. 'Item 8' ~260k chars),
    unlike semantic attributes such as `tenk.financials` which return summary
    objects whose str() is a one-line stub. Use the canonical labels the filing
    reports in `tenk.items`.
    """
    tenk = filing_obj.obj()
    sections = {}
    for label in TARGET_ITEMS:
        try:
            val = tenk[label]
        except Exception:
            val = None
        if val is None:
            continue
        text = _normalize_text(str(val))
        if len(text) >= MIN_SECTION_CHARS and text.lower() != "none":
            sections[label] = text
    return sections


def _extract_sections_fallback(filing_obj) -> dict[str, str]:
    from bs4 import BeautifulSoup
    import re

    try:
        html = filing_obj.html() or ""
    except Exception:
        html = ""

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    patterns = {
        "Item 1": r"(?i)item\s*1[\.\s]+business",
        "Item 1A": r"(?i)item\s*1a[\.\s]+risk\s+factors",
        "Item 7": r"(?i)item\s*7[\.\s]+management",
        "Item 7A": r"(?i)item\s*7a",
        "Item 8": r"(?i)item\s*8[\.\s]+financial\s+statements",
        "Item 9A": r"(?i)item\s*9a",
    }

    sections = {}
    items = list(patterns.items())
    for i, (label, pat) in enumerate(items):
        m = re.search(pat, text)
        if not m:
            continue
        start = m.start()
        if i + 1 < len(items):
            next_pat = items[i + 1][1]
            m2 = re.search(next_pat, text[start + 1:])
            end = start + 1 + m2.start() if m2 else len(text)
        else:
            end = len(text)
        sections[label] = text[start:end].strip()

    return sections


def _validate_mandatory_sections(ticker: str, sections: dict):
    missing = MANDATORY_SECTIONS - set(sections.keys())
    if missing:
        raise ValueError(f"Mandatory sections missing for {ticker}: {missing}")
    # Present-but-tiny is a parse failure (truncation or stub), not success.
    truncated = {
        s: len(sections[s])
        for s in MANDATORY_SECTIONS
        if len(sections[s]) < MIN_MANDATORY_CHARS
    }
    if truncated:
        raise ValueError(
            f"Mandatory sections truncated for {ticker} (< {MIN_MANDATORY_CHARS} chars): {truncated}"
        )
    for opt in OPTIONAL_SECTIONS:
        if opt not in sections:
            log.warning("optional_section_missing", ticker=ticker, section=opt)


def _parse_fiscal_year(filing_obj) -> int:
    period = str(getattr(filing_obj, "period_of_report", ""))
    if period and len(period) >= 4:
        try:
            return int(period[:4])
        except ValueError:
            pass
    return datetime.now().year


def save_filing(filing: Filing):
    out_dir = PROCESSED_DIR / filing.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{filing.accession_number.replace('/', '_')}.json"
    path.write_text(filing.model_dump_json(indent=2))
    log.info("filing_saved", ticker=filing.ticker, path=str(path))
    return path
