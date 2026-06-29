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

### M2 — Index ✅ (2026-06-29)

**Verification:** `python scripts/02_index.py --verify` → passed (drop/stale/embedding-norm/idempotency). `pytest tests/` → **34 passed, 1 skipped** (Phase 2).

**Numbers:**
- 1,582 chunks across all 12 filings, stored in ChromaDB (cosine), 85 MB on disk.
- Chunk size 800 tokens / 120 overlap (PRD-mandated), avg ~well within the model limit, 0 chunks truncated.
- Embeddings: nomic-embed-text-v1.5, 768-dim, unit-norm, no NaN.
- Idempotent: re-derived chunk ids identical across passes; re-upsert count delta 0.
- Relevance smoke test: "principal risk factors" → top hit Regeneron Item 1A.

**Decision (user-approved PRD deviation):** the PRD mandates 800-token chunks, but
`bge-small-en-v1.5` caps at 512, so 800-token chunks would be silently truncated at
embed time. Switched the embedder to **nomic-embed-text-v1.5** (8192-token context,
768-dim) to honor the literal 800/120. Adds `search_document:`/`search_query:` task
prefixes (applied at embed time only; stored text stays verbatim) and `trust_remote_code`.
Collection renamed `pharma_10k_v768_nomic_v1_5`.

**Critical chunker bugs found and fixed before indexing:**
1. **Verbatim-text corruption** — chunks were built via `tokenizer.decode(token_ids)`
   on the UNCASED bge tokenizer, which lowercased and mangled punctuation
   ("U.S." → "u. s.", "12%" → "12 %"). Since stored chunk text is shown in
   citations and fed to the LLM, this poisoned everything downstream. Rewrote
   chunking to slice the ORIGINAL text by sentence boundaries, using the tokenizer
   only to MEASURE token counts.
2. **800 > 512 truncation** — see decision above.

**M2 adversarial QA audit (3 lenses × verify) — 11/12 findings confirmed, all fixed:**
- **HIGH reranker silently bypassed** — used `signal.SIGALRM`, which only works on the
  main thread, so it raised `ValueError` (caught as a generic error) under FastAPI/
  Streamlit worker threads. Replaced with a `ThreadPoolExecutor` timeout; genuine
  failures now logged distinctly from timeouts.
- **HIGH multi-key filter 500** — a `{ticker, fiscal_year}` ChromaDB filter (exactly
  what the UI builds) crashed with HTTP 500; ChromaDB needs an explicit `$and`.
  `store.query` now wraps multi-condition filters in `$and` and degrades to empty on
  any query error instead of propagating a 500.
- **MEDIUM overlap silently dropped to 0** — the reverse-walk carry broke before adding
  any sentence when a boundary sentence exceeded the overlap budget (~18% of in-section
  pairs had zero overlap). Added a token-tail fallback; zero-overlap pairs dropped to ~6%
  (residual are hard-split boundaries).
- **MEDIUM degenerate chunks** — 3–21-token fragments were embedded standalone. Added a
  min-chunk-size merge (also fixes the orphan-before-hard-split LOW finding).
- **MEDIUM `--verify` was a no-op** — re-upserting the same objects could never change the
  count. Replaced with real invariants: drop/collision detection, stale-chunk detection,
  embedding dim/norm/NaN sanity, and cross-pass idempotency (re-derive ids from disk).
- **LOW** BM25 `ZeroDivisionError` on an empty corpus (guarded); RRF returned an uncapped
  union to the reranker (now capped at `max(top_k_dense, bm25_top_k)`).
- **Rejected (1):** cross-encoder `score_threshold=-2.0` flagged as arbitrary — verifier
  judged it acceptable; left as-is.

**Infra note:** nomic on Apple **MPS hangs** on long-sequence batches (froze at 0% CPU
mid-embed). Added an `EMBED_DEVICE` env override and ran the index build on **CPU**
(reliable). Runtime single-query embedding still uses the configured device.

**Tests added:** chunk-level §19.1 checks in `test_retrieval.py` (dedup, size
distribution, metadata completeness, all-companies/sections indexed, verbatim-text
guard, relevance) + `test_m2_regressions.py` (rerank off-main-thread, no-signal,
multi-key `$and` filter end-to-end).

Next: M3 — RAG answer pipeline (retrieve → rerank → generate with citations).
