#!/usr/bin/env python3
"""M1: Fetch and parse pharma 10-K filings from SEC EDGAR."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import cfg
from src.ingest.edgar_client import MAX_FILINGS
from src.ingest.parse import save_filing
from src.logging_setup import get_logger

log = get_logger("01_ingest")


def run(verify: bool = False):
    tickers = cfg["corpus"]["tickers"]
    num_years = cfg["corpus"]["num_fiscal_years"]
    prefer_amended = cfg["edgar"]["prefer_amended"]

    if len(tickers) > MAX_FILINGS:
        raise ValueError(f"Ticker list ({len(tickers)}) exceeds max_filings_per_run ({MAX_FILINGS})")

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

            from src.ingest.parse import parse_filing
            from src.ingest.edgar_client import sha256_of_bytes

            filing = parse_filing(ticker, filing_obj)

            # compute sha256 from raw text
            try:
                raw_text = filing_obj.text() or ""
                filing.file_sha256 = sha256_of_bytes(raw_text.encode())
            except Exception:
                filing.file_sha256 = "unavailable"

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
        processed = list((ROOT / "data" / "processed").rglob("*.json"))
        print(f"\nVerification: {len(processed)} filing JSON(s) found under data/processed/")
        if len(processed) < len(results["ok"]):
            print("  WARNING: fewer files than expected")
            sys.exit(1)
        print("  M1 verification passed.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(verify=args.verify)
