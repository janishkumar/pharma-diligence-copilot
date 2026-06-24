#!/usr/bin/env python3
"""M1: Fetch and parse pharma 10-K filings from SEC EDGAR."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import cfg
from src.ingest.edgar_client import (
    MAX_FILINGS,
    cache_raw_filing,
    is_cached,
    verify_cached_filing,
)
from src.ingest.parse import MANDATORY_SECTIONS, parse_filing, save_filing
from src.logging_setup import get_logger

log = get_logger("01_ingest")

PROCESSED_DIR = ROOT / "data" / "processed"


def _processed_path(ticker: str, accession: str) -> Path:
    return PROCESSED_DIR / ticker / f"{accession.replace('/', '_')}.json"


def run(verify: bool = False):
    tickers = cfg["corpus"]["tickers"]
    num_years = cfg["corpus"]["num_fiscal_years"]
    prefer_amended = cfg["edgar"]["prefer_amended"]

    # Guardrail counts the actual filings to fetch (tickers x years), not just tickers.
    planned = len(tickers) * max(num_years, 1)
    if planned > MAX_FILINGS:
        raise ValueError(
            f"Planned filings ({planned} = {len(tickers)} tickers x {num_years} years) "
            f"exceeds max_filings_per_run ({MAX_FILINGS})"
        )

    try:
        import edgar as et
    except ImportError:
        log.error("edgartools_not_installed", hint="pip install edgartools")
        sys.exit(1)

    et.set_identity(cfg["edgar"]["user_agent"])

    results = {"ok": [], "failed": []}

    for ticker in tickers:
        log.info("processing", ticker=ticker)
        try:
            company = et.Company(ticker)
            latest_10k = company.get_filings(form="10-K").latest(1)
            filing_obj = latest_10k

            # prefer amended only if it covers the same fiscal period as the latest 10-K
            if prefer_amended:
                try:
                    amended = company.get_filings(form="10-K/A").latest(1)
                    if amended and amended.period_of_report == latest_10k.period_of_report:
                        log.info("using_amended", ticker=ticker, accession=amended.accession_number)
                        filing_obj = amended
                except Exception:
                    pass

            accession = getattr(filing_obj, "accession_number", "") or getattr(filing_obj, "accession_no", "")

            # Cache check: if raw is cached, processed JSON exists, and the stored
            # checksum still matches the cached raw bytes, skip re-fetch (Section 8).
            proc_path = _processed_path(ticker, accession)
            if is_cached(ticker, accession) and proc_path.exists():
                stored = json.loads(proc_path.read_text())
                if verify_cached_filing(ticker, accession, stored.get("file_sha256", "")):
                    log.info("cache_hit_skip", ticker=ticker, accession=accession)
                    results["ok"].append(ticker)
                    continue
                log.warning("cache_invalid_refetch", ticker=ticker, accession=accession)

            filing = parse_filing(ticker, filing_obj)

            # Persist raw filing text under data/raw/ and derive checksum from the
            # persisted bytes so it can be re-verified later (Section 8, 9.1).
            raw_text = filing_obj.text() or ""
            _, sha = cache_raw_filing(ticker, accession, raw_text)
            filing.file_sha256 = sha

            save_filing(filing)
            results["ok"].append(ticker)
            log.info("done", ticker=ticker, fiscal_year=filing.fiscal_year, sections=list(filing.parsed_sections.keys()))

        except Exception as e:
            log.error("failed", ticker=ticker, error=str(e))
            results["failed"].append(ticker)

    print(f"\nIngestion complete: {len(results['ok'])} ok, {len(results['failed'])} failed")
    if results["failed"]:
        print(f"  Failed: {results['failed']}")

    if verify:
        ok = _verify(tickers)
        if not ok:
            sys.exit(1)

    return results


def _verify(expected_tickers: list[str]) -> bool:
    """M1 acceptance gate (PRD Section 6): every config ticker fetched, parsed,
    normalized JSON exists, source_url + valid checksum recorded, raw cached and
    its checksum re-verifies."""
    print("\n=== M1 verification (PRD Section 6) ===")
    files = list(PROCESSED_DIR.rglob("*.json"))
    by_ticker = {}
    for f in files:
        d = json.loads(f.read_text())
        by_ticker[d["ticker"]] = d

    problems = []
    for tk in expected_tickers:
        d = by_ticker.get(tk)
        if d is None:
            problems.append(f"{tk}: no processed filing")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", d.get("file_sha256", "")):
            problems.append(f"{tk}: file_sha256 not a valid 64-hex checksum")
        if not d.get("source_url", "").startswith("http"):
            problems.append(f"{tk}: source_url missing/invalid")
        missing = MANDATORY_SECTIONS - set(d.get("parsed_sections", {}).keys())
        if missing:
            problems.append(f"{tk}: missing mandatory sections {missing}")
        if not verify_cached_filing(tk, d["accession_number"], d["file_sha256"]):
            problems.append(f"{tk}: raw cache missing or checksum mismatch")

    print(f"  Companies expected: {len(expected_tickers)}, parsed: {len(by_ticker)}")
    if problems:
        print("  FAILURES:")
        for p in problems:
            print(f"    - {p}")
        return False
    print("  All checks passed: filings, sections, source_url, checksum, raw cache verified.")
    print("  M1 verification passed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(verify=args.verify)
