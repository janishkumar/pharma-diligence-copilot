"""EDGAR raw-filing caching and checksum verification.

Acquisition itself goes through `edgartools` (PRD Section 8 mandates it as the
acquisition method). edgartools enforces SEC compliance on every request:
<=8 req/s rate limiting (stricter than the 10 req/s cap), exponential backoff
with jitter, and 429/503 handling. This module owns the parts edgartools does
not: persisting raw filings under data/raw/ so re-runs do not re-hit EDGAR
(Section 8), and recomputing file_sha256 on load to refuse tampered/changed
cached files (Section 9.1, Section 16).
"""
import hashlib
from datetime import datetime, timezone

from src.config import cfg, PROJECT_ROOT
from src.logging_setup import get_logger

log = get_logger("edgar_client")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILINGS = cfg["corpus"]["max_filings_per_run"]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw_path(ticker: str, accession_number: str):
    safe = accession_number.replace("/", "_")
    p = RAW_DIR / ticker / f"{safe}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_raw_filing(ticker: str, accession_number: str, text: str) -> tuple[str, str]:
    """Persist raw filing text under data/raw/ and return (path, sha256).

    The sha256 is computed from the persisted bytes so it can be re-verified
    later against a real artifact (not a transient in-memory string).
    """
    path = _raw_path(ticker, accession_number)
    data = text.encode("utf-8")
    path.write_bytes(data)
    sha = sha256_of_bytes(data)
    log.info("raw_cached", ticker=ticker, accession=accession_number, bytes=len(data), sha256=sha[:16])
    return str(path), sha


def is_cached(ticker: str, accession_number: str) -> bool:
    return _raw_path(ticker, accession_number).exists()


def verify_cached_filing(ticker: str, accession_number: str, expected_sha256: str) -> bool:
    """Recompute sha256 of the cached raw file and compare (Section 9.1).

    Returns True if the cached file exists and matches. On mismatch, logs and
    returns False so the caller refuses the cached file and re-fetches.
    """
    path = _raw_path(ticker, accession_number)
    if not path.exists():
        return False
    actual = sha256_of_bytes(path.read_bytes())
    if actual != expected_sha256:
        log.warning(
            "checksum_mismatch", ticker=ticker, accession=accession_number,
            expected=expected_sha256[:16], actual=actual[:16],
        )
        return False
    return True
