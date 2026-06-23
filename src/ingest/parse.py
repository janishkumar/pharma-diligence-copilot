import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT, cfg
from src.logging_setup import get_logger
from src.schemas import Filing

log = get_logger("parse")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MANDATORY_SECTIONS = {"Item 1", "Item 1A", "Item 7"}
OPTIONAL_SECTIONS = {"Item 7A", "Item 8", "Item 9A"}


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
        source_url=getattr(filing_obj, "filing_index", ""),
        file_sha256="",  # filled in by ingest script after fetch
        ingested_at=datetime.now(timezone.utc).isoformat(),
        parsed_sections=sections,
    )
    return filing


def _extract_sections_edgartools(filing_obj) -> dict[str, str]:
    tenk = filing_obj.obj()
    sections = {}
    item_map = {
        "Item 1": ["business"],
        "Item 1A": ["risk_factors"],
        "Item 7": ["management_discussion"],
        "Item 7A": ["market_risk"],
        "Item 8": ["financials"],
        "Item 9A": ["controls_and_procedures"],
    }
    for label, attrs in item_map.items():
        for attr in attrs:
            val = getattr(tenk, attr, None)
            if val is not None:
                text = str(val).strip()
                if text:
                    sections[label] = text
                    break
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
