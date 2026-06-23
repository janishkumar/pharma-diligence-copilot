import hashlib
import json
import time
import random
from pathlib import Path
from datetime import datetime, timezone

import httpx
from src.config import cfg, PROJECT_ROOT
from src.logging_setup import get_logger

log = get_logger("edgar_client")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = cfg["edgar"]["user_agent"]
RATE_LIMIT = cfg["edgar"]["rate_limit_per_sec"]
BACKOFF_BASE = cfg["edgar"]["backoff_base_sec"]
BACKOFF_MAX = cfg["edgar"]["backoff_max_sec"]
MAX_RETRIES = cfg["edgar"]["backoff_max_retries"]
MAX_FILINGS = cfg["corpus"]["max_filings_per_run"]

_last_request_time = 0.0


def _throttle():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    gap = 1.0 / RATE_LIMIT
    if elapsed < gap:
        time.sleep(gap - elapsed)
    _last_request_time = time.monotonic()


def _backoff_get(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    consecutive_failures = 0
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES + 1):
        _throttle()
        try:
            r = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            if r.status_code in (429, 503):
                consecutive_failures += 1
                if consecutive_failures >= MAX_RETRIES:
                    raise RuntimeError(
                        f"BLOCKER: EDGAR returned {r.status_code} {MAX_RETRIES} times in a row on {url}. "
                        "Check your network, wait a few minutes, then retry."
                    )
                jitter = random.uniform(0, delay)
                wait = min(delay + jitter, BACKOFF_MAX)
                log.warning("rate_limited", status=r.status_code, wait_sec=round(wait, 1), attempt=attempt + 1)
                time.sleep(wait)
                delay = min(delay * 2, BACKOFF_MAX)
                continue
            r.raise_for_status()
            return r.content
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error fetching {url}: {e}") from e
    raise RuntimeError(f"Exhausted retries for {url}")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_filing_raw(ticker: str, accession_number: str, url: str) -> tuple[bytes, str]:
    """Fetch raw filing bytes, using local cache if present. Returns (bytes, sha256)."""
    cache_path = RAW_DIR / ticker / f"{accession_number.replace('/', '_')}.raw"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists():
        data = cache_path.read_bytes()
        meta = json.loads(meta_path.read_text())
        current_sha = sha256_of_bytes(data)
        if current_sha != meta["file_sha256"]:
            log.warning("checksum_mismatch", ticker=ticker, cached_sha=meta["file_sha256"], actual_sha=current_sha)
            log.info("re_fetching", ticker=ticker, url=url)
        else:
            log.info("cache_hit", ticker=ticker, accession=accession_number)
            return data, current_sha

    log.info("fetching", ticker=ticker, url=url)
    data = _backoff_get(url)
    sha = sha256_of_bytes(data)
    cache_path.write_bytes(data)
    meta_path.write_text(json.dumps({"file_sha256": sha, "url": url, "fetched_at": datetime.now(timezone.utc).isoformat()}))
    return data, sha
