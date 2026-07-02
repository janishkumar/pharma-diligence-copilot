"""Tests for deterministic company auto-detection / query scoping."""
from src.retrieve.company_filter import detect_ticker, detect_tickers, apply_auto_company_filter


def test_detect_single_company_by_name():
    assert detect_ticker("What are Vertex's main products?") == "VRTX"
    assert detect_ticker("Eli Lilly's principal risk factors") == "LLY"
    assert detect_ticker("How is Moderna's mRNA platform?") == "MRNA"


def test_detect_aliases_and_abbreviations():
    assert detect_ticker("J&J revenue") == "JNJ"
    assert detect_ticker("BMS key products") == "BMY"
    assert detect_ticker("bristol myers squibb pipeline") == "BMY"
    assert detect_ticker("lilly diabetes franchise") == "LLY"


def test_no_company_named_returns_none():
    assert detect_ticker("Which companies mention climate risk?") is None
    assert detect_ticker("What are the biggest industry-wide risks?") is None


def test_multiple_companies_returns_none():
    # ambiguous -> do not scope, let retrieval span the corpus
    assert detect_ticker("Compare Pfizer and Merck revenue") is None
    assert set(detect_tickers("Pfizer, Merck, and Moderna")) == {"PFE", "MRK", "MRNA"}


def test_no_substring_false_positives():
    # 'lilly' must not match inside another word; 'merck' is whole-token only
    assert detect_ticker("the merckle family office") != "MRK" or detect_ticker("the merckle family office") is None


def test_apply_filter_adds_ticker_when_unscoped():
    filters, auto = apply_auto_company_filter("Vertex products", None)
    assert auto == "VRTX"
    assert filters == {"ticker": {"$eq": "VRTX"}}


def test_apply_filter_respects_explicit_ticker():
    explicit = {"ticker": {"$eq": "PFE"}}
    filters, auto = apply_auto_company_filter("Vertex products", explicit)
    assert auto is None  # explicit wins, no override
    assert filters == explicit


def test_apply_filter_merges_with_year_filter():
    filters, auto = apply_auto_company_filter("Moderna 2025 risks", {"fiscal_year": {"$eq": 2025}})
    assert auto == "MRNA"
    assert filters["ticker"] == {"$eq": "MRNA"}
    assert filters["fiscal_year"] == {"$eq": 2025}


def test_apply_filter_noop_when_no_company():
    filters, auto = apply_auto_company_filter("industry-wide pricing pressure", None)
    assert auto is None
    assert filters is None
