"""Data integrity tests — PRD Section 19.1.

Filing/section-level checks that apply after M1 (ingestion). Chunk-level §19.1
checks (deduplication, chunk-size distribution, per-chunk metadata correctness)
require the index and are exercised at M2 in test_retrieval.py.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

MANDATORY_SECTIONS = {"Item 1", "Item 1A", "Item 7"}
MIN_MANDATORY_CHARS = 500
REPLACEMENT_CHAR = "�"
MOJIBAKE_MARKERS = ["Ã©", "Ã¨", "â€™", "â€œ", "â€\x9d", "Ã ", "Ã¢"]

# Plausible name fragment per ticker (case-insensitive substring of company_name)
EXPECTED_NAME = {
    "PFE": "pfizer", "MRK": "merck", "JNJ": "johnson", "ABBV": "abbvie",
    "BMY": "bristol", "LLY": "lilly", "AMGN": "amgen", "GILD": "gilead",
    "REGN": "regeneron", "VRTX": "vertex", "MRNA": "moderna", "BIIB": "biogen",
}


def get_all_filings():
    return list(PROCESSED_DIR.rglob("*.json"))


def load_all():
    return [json.loads(p.read_text()) for p in get_all_filings()]


def config_tickers():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return cfg["corpus"]["tickers"]


def test_processed_dir_exists():
    assert PROCESSED_DIR.exists(), "data/processed/ must exist after ingestion"


def test_all_config_companies_ingested():
    """Corpus completeness: every ticker in config produced a filing."""
    ingested = {d["ticker"] for d in load_all()}
    missing = set(config_tickers()) - ingested
    assert not missing, f"Config tickers not ingested: {missing}"


def test_mandatory_sections_present():
    for d in load_all():
        sections = set(d.get("parsed_sections", {}).keys())
        missing = MANDATORY_SECTIONS - sections
        assert not missing, f"{d['ticker']}: missing mandatory sections {missing}"


def test_mandatory_sections_substantive():
    """Present-but-tiny mandatory section = truncation/stub. Regression guard
    for the Item 8 `str(Financials_object)` stub bug and the LLY truncation."""
    for d in load_all():
        for sec in MANDATORY_SECTIONS:
            text = d["parsed_sections"].get(sec, "")
            assert len(text) >= MIN_MANDATORY_CHARS, (
                f"{d['ticker']} {sec}: only {len(text)} chars "
                f"(< {MIN_MANDATORY_CHARS}); likely truncated or a stub"
            )


def test_no_empty_sections():
    for d in load_all():
        for section, text in d.get("parsed_sections", {}).items():
            assert text and text.strip(), f"{d['ticker']}: section '{section}' is empty"


def test_encoding_integrity():
    """No Unicode replacement characters or common mojibake sequences."""
    for d in load_all():
        full = " ".join(d["parsed_sections"].values())
        assert REPLACEMENT_CHAR not in full, f"{d['ticker']}: U+FFFD replacement char present"
        for marker in MOJIBAKE_MARKERS:
            assert marker not in full, f"{d['ticker']}: mojibake marker {marker!r} present"


def test_checksum_is_valid_sha256():
    for d in load_all():
        sha = d.get("file_sha256", "")
        assert re.fullmatch(r"[0-9a-f]{64}", sha), f"{d['ticker']}: invalid sha256 {sha!r}"


def test_metadata_correctness():
    """Company name plausibly matches ticker, CIK numeric, fiscal year sane."""
    for d in load_all():
        ticker = d["ticker"]
        expected = EXPECTED_NAME.get(ticker)
        if expected:
            assert expected in d["company_name"].lower(), (
                f"{ticker}: company_name {d['company_name']!r} does not match expected '{expected}'"
            )
        assert d["cik"].isdigit(), f"{ticker}: cik {d['cik']!r} not numeric"
        assert isinstance(d["fiscal_year"], int) and 2000 <= d["fiscal_year"] <= 2030, (
            f"{ticker}: implausible fiscal_year {d['fiscal_year']}"
        )


def test_fiscal_year_matches_period():
    """fiscal_year must equal the year of fiscal_year_end_date."""
    for d in load_all():
        end = d.get("fiscal_year_end_date", "")
        if end and len(end) >= 4 and end[:4].isdigit():
            assert d["fiscal_year"] == int(end[:4]), (
                f"{d['ticker']}: fiscal_year {d['fiscal_year']} != period end {end}"
            )


def test_section_junk_ratio():
    """Junk detection: non-alphanumeric/space ratio stays under 15% per section."""
    for d in load_all():
        for section, text in d["parsed_sections"].items():
            if len(text) < 200:  # short cross-references exempt
                continue
            nonalnum = sum(1 for c in text if not (c.isalnum() or c.isspace()))
            ratio = nonalnum / len(text)
            assert ratio < 0.15, f"{d['ticker']} {section}: junk ratio {ratio:.0%} >= 15%"


def test_source_url_present():
    """Section 8: source_url is mandatory and must be a real https EDGAR URL.
    Regression guard for the filing_index->filing_url bug."""
    for d in load_all():
        url = d.get("source_url", "")
        assert url.startswith("https://"), f"{d['ticker']}: source_url not https: {url!r}"
        assert "sec.gov" in url, f"{d['ticker']}: source_url not an SEC URL: {url!r}"


def test_raw_cache_exists_and_checksum_stable():
    """Section 8/9.1/19.1: raw filing cached under data/raw/, and recomputing its
    sha256 on reload matches the stored file_sha256 (checksum stability)."""
    import hashlib
    raw_dir = ROOT / "data" / "raw"
    for d in load_all():
        acc = d["accession_number"].replace("/", "_")
        raw_path = raw_dir / d["ticker"] / f"{acc}.txt"
        assert raw_path.exists(), f"{d['ticker']}: raw cache missing at {raw_path}"
        actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert actual == d["file_sha256"], (
            f"{d['ticker']}: checksum mismatch on reload (tamper/corruption)"
        )


def test_no_boilerplate_prefix():
    """Page furniture (Table of Contents nav, leading page numbers) is stripped."""
    for d in load_all():
        for section, text in d["parsed_sections"].items():
            head = text[:40]
            assert not head.lower().startswith("table of contents"), (
                f"{d['ticker']} {section}: leading 'Table of Contents' boilerplate"
            )
            assert not re.match(r"^\d{1,4}\s", head), (
                f"{d['ticker']} {section}: stray leading page number: {head!r}"
            )
