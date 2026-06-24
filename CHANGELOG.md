# Changelog

## Session 1 — 2026-06-22

**Decisions locked:** D1=Anthropic/claude-sonnet-4-6, D2=Qwen2.5-7B-Instruct, D3=Azure A100, D4=TensorBoard, D5=Streamlit 8501, D6=OpenAI/gpt-4o

**Repository scaffolded.** Full directory structure, config.yaml, .env.example, requirements.in, all source stubs, and scripts/00_check_env.py created.

**Environment check passed** (Python 3.13.5, 251 GB free, Anthropic + OpenAI keys present). Network reachability checks to EDGAR/Anthropic were false negatives (5s timeout); confirmed reachable via curl.

### M1 — Ingestion ✅ (2026-06-23)

**Verification:** `python scripts/01_ingest.py --verify` → 12 ok, 0 failed. `pytest tests/test_data_integrity.py` → 10/10 pass.

**Numbers:**
- 12/12 pharma tickers ingested (PFE, MRK, JNJ, ABBV, BMY, LLY, AMGN, GILD, REGN, VRTX, MRNA, BIIB), all fiscal year 2025.
- All 6 target sections per filing: Item 1, 1A, 7, 7A, 8, 9A.
- Section sizes: Item 1 ~35k–165k chars, Item 1A ~43k–213k, Item 7 ~47k–141k. Item 8 full text for 8 companies; 4 (AMGN, BIIB, REGN, VRTX) carry the legitimate "incorporated by reference to Item 15" cross-reference.
- 0 encoding issues, 0 metadata failures, all `file_sha256` valid 64-hex.

**Bugs found and fixed during M1 QA:**
1. **Item 8 stub** — parser used `str(tenk.financials)`, a summary object whose repr is a one-line stub (`"Financials(... • N facts)"`). Switched to dict-style item access `tenk["Item 8"]`, which returns full item text. This also recovered Item 7A and 9A (previously logged missing). Added a 500-char floor on mandatory sections to reject present-but-truncated parses.
2. **Amendment selection** — `prefer_amended` was pulling decade-old 10-K/A filings. Now only prefers a 10-K/A whose `period_of_report` matches the latest 10-K.
3. **store.py import crash** — `chromadb.PersistentClient | None` annotation threw `TypeError` at import (factory function, not a type). Would have blocked M2. Fixed with `Optional[Any]`.

**Tests added:** expanded `tests/test_data_integrity.py` to 10 checks (§19.1): corpus completeness, mandatory-section presence + substantive length, encoding integrity, valid sha256, metadata correctness, fiscal-year consistency, junk ratio.

#### M1 adversarial QA audit (27 agents, 3 lenses × verify) — 22/24 findings confirmed

Ran an independent multi-agent audit (data-quality, code-correctness, PRD-compliance lenses, each finding adversarially verified). Fixed all HIGH/MEDIUM findings:

- **HIGH `source_url` empty on all 12 filings** — parser read non-existent `filing_index` attr → always `""`, breaking the §9.7/§14.2 citation contract. Fixed to use `filing_url` (fallback `homepage_url`/`url`). Now valid https EDGAR URLs.
- **HIGH no raw cache / no checksum verification** — `data/raw/` was empty; `file_sha256` was computed from a transient in-memory string with no artifact to re-verify, so §9.1 checksum-on-load, §16 tamper detection, and the §19.1 checksum-stability test were unsatisfiable. Now persists raw filing text under `data/raw/{ticker}/`, derives the checksum from persisted bytes, re-verifies on load, and skips re-fetch on a valid cache hit.
- **HIGH dead `edgar_client` HTTP layer** — the custom fetch/throttle/backoff code was never called by the active edgartools path, falsely implying §8 controls were enforced. Removed it and repurposed the module for raw caching + checksum verification (the parts edgartools doesn't do). edgartools enforces SEC rate limits (≤8 req/s, backoff, 429 handling) on the active path; documented.
- **MEDIUM weak `--verify` gate** — only counted JSON files. Now asserts every config ticker is parsed, mandatory sections present, `source_url` is https, `file_sha256` is valid 64-hex, and the raw cache re-verifies.
- **LOW data cleaning** — strip recurring page furniture ("…2025 Form 10-K" headers, bare page-number lines, "Table of Contents" nav, stray leading page numbers) in `parse.py`, preserving legitimate inline prose. Removed unused imports. Guardrail now counts tickers × years, not just tickers.

**Deferred (documented):**
- Table/financial-statement structure is flattened on parse (numbers fuse to labels). PRD-sanctioned v1 limitation; structured table extraction is a Phase 3 item (§9.2, §16).
- `sec-edgar-downloader` acquisition fallback (§8) not implemented — edgartools is the PRD-mandated primary and fetched all 12 successfully; a parse-level HTML/BeautifulSoup fallback exists. Acquisition fallback deferred.

**Tests added:** `test_source_url_present`, `test_raw_cache_exists_and_checksum_stable`, `test_no_boilerplate_prefix`. Full suite: **21 passed, 1 skipped** (Phase 2).

Next: M2 Index — chunk, embed, store in ChromaDB.
