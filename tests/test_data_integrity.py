"""Data integrity tests — Section 19.1"""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def get_all_filings():
    return list(PROCESSED_DIR.rglob("*.json"))


def test_processed_dir_exists():
    assert PROCESSED_DIR.exists(), "data/processed/ must exist after ingestion"


def test_mandatory_sections_present():
    mandatory = {"Item 1", "Item 1A", "Item 7"}
    for path in get_all_filings():
        data = json.loads(path.read_text())
        sections = set(data.get("parsed_sections", {}).keys())
        missing = mandatory - sections
        assert not missing, f"{path.name}: missing mandatory sections {missing}"


def test_no_empty_sections():
    for path in get_all_filings():
        data = json.loads(path.read_text())
        for section, text in data.get("parsed_sections", {}).items():
            assert text and text.strip(), f"{path.name}: section '{section}' is empty"


def test_sha256_present():
    for path in get_all_filings():
        data = json.loads(path.read_text())
        assert data.get("file_sha256"), f"{path.name}: missing file_sha256"
