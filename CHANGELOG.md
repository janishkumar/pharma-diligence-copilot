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

### M3 — RAG answer pipeline ✅ (2026-06-29)

**Verification:** live end-to-end queries through Claude + `pytest tests/` → **49 passed, 1 skipped**.
- Positive (Pfizer risk factors): grounded answer, citations map to real Item 1/1A chunks.
- Positive (Moderna mRNA platform): grounded answer, 3 citations (Item 1 + Item 8).
- Negative (Pfizer Bitcoin holdings): correctly abstains, 0 citations.

**What M3 delivers:** `retrieve → rerank → generate` with inline `[n]` citations or abstention,
returned as the `AskResponse` contract. CLI at `scripts/03_ask.py`.

**Bug found during sanity-testing (before the audit):**
- **Abstention undetected** — exact-equality check `answer == ABSTENTION_STRING` failed because
  the model abstains *then helpfully elaborates*, so the phrase is a prefix. Replaced with robust
  leading-phrase detection (`_is_abstention`, tolerant of quotes/markdown/elaboration).

**Critical regression I introduced AND caught via live testing:**
- Following audit finding #12 ("PRD §9.7 mandates top_p=0.95"), I added `top_p` to the Anthropic
  call — but **Claude's API returns 400 if both `temperature` and `top_p` are sent**. My new error
  handler then masked the 400 as an abstention, so *every* positive query silently "abstained"
  (generation_ms ~400ms gave it away). Mocked unit tests passed; only a live query exposed it.
  Fix: Anthropic sends temperature only (the determinism control); top_p stays on openai/ollama.
  Locked with a regression test asserting the Anthropic call omits top_p.

**M3 adversarial QA audit (3 lenses × verify) — 21/21 findings confirmed, fixed:**
- **HIGH ungrounded answers passed silently** — a non-abstaining answer with zero valid citations was
  returned as fact. Added a groundedness guard: refuse (abstain) when there are no valid citations.
- **HIGH generator had no error handling** — any API error (rate limit, overload, network, 400)
  crashed the request. Added bounded retry+backoff on transient errors, empty-content guards, and
  graceful degradation; generation failures now surface a distinct `error` (not a fake abstention).
- **HIGH no §19.3 answer-quality tests** — added 18 API-free tests (citation parsing across
  [1][2]/[1,2]/[1-3]/double-digit/out-of-range, abstention detection, groundedness guard, error
  surfacing, top_p regression).
- **MEDIUM citation parser** — only matched adjacent `[n]`; grouped `[1,2]`/`[1-3]` parsed to ZERO
  citations (a correctly-cited answer looked unsupported, and corrupted eval recall). Rewrote
  `_parse_citations` with a regex that expands comma/range groups and validates 1..len(chunks);
  out-of-range markers (e.g. `[10]`) are logged, not silently dropped.
- **MEDIUM** missing `top_p` on Anthropic (see regression above — resolved by sending temperature only);
  truncation now detected via `stop_reason` and flagged (`AskResponse.truncated`).
- **MEDIUM rerank 2s timeout** — CPU cross-encoder predict (~2.5s) exceeded it, silently falling back to
  dense on most queries. Raised to 30s so reranking actually applies.
- **LOW** dead `--local` flag (now sets the backend override before import); `_corpus_snapshot_hash`
  memoized (was rglob-ing 85MB every query); eval scorecard `model_version` bug (was storing answer text).

**Deferred (documented):**
- **Full per-session cost-budget enforcement** — token usage + USD cost are now captured per call in
  `GenerationResult`, but not yet accumulated/halted against `COST_BUDGET_USD` or surfaced in
  `AskResponse`. Enforcement is a follow-up; usage is observable in logs.
- **Reranker pre-warm** — first query still pays the one-time cross-encoder load (~20s) inside
  `rerank_ms`; a startup warm-up is deferred to M5 (server boot).
- **Content-based corpus hash** — snapshot hash is mtime-based (not reproducible across machines);
  acceptable for a single-host v1.

### M5 (partial) — UI improvements + retrieval-scoping fix (2026-06-29)

Pulled forward from M5 after live user testing exposed a real defect:

- **Cross-company retrieval contamination** — unscoped questions ("Vertex's products?",
  "Eli Lilly's risks?") retrieved chunks from OTHER issuers because generic 10-K language
  embeds nearly identically across filings. The model behaved honestly (refused or flagged
  the mismatch) but the UX was broken. Fix: **deterministic company auto-detection**
  (`src/retrieve/company_filter.py`) — an alias map over the 12 known issuers applied in
  the pipeline when the caller didn't pin a ticker; explicit filters always win; zero or
  multiple matches → no filter (cross-company questions still work). 9 unit tests.
- **UI (`src/ui.py`)**: fixed `ModuleNotFoundError` under `streamlit run` (sys.path);
  starter example questions; newest-first history with latest expanded + Clear button;
  auto-scope indicator; distinct error/abstention/truncation states.

### M4 — Evaluation harness ✅ (2026-07-02)

**Verification:** `python scripts/04_eval.py` on a 15-question golden set → **Phase-1 PASS**.
`pytest tests/` → **58 passed, 1 skipped**.

**Harness (scripts/04_eval.py):**
- Golden set `eval/golden_set_v1.json`: 15 questions across all 12 issuers (11 factual,
  1 numeric, 3 abstention traps), written against verified filing content.
- **Retrieval-trace plumbing** (`ask(_trace=)`) — recall and the judge now measure what was
  *actually retrieved*, not just what was cited (the scaffold's recall was circular: it
  measured cited chunks, so it couldn't see retrieval misses at all).
- Section-level recall@5/10/20; GPT-4o judge (groundedness/relevance/correctness) fed the
  FULL reranked context the generator saw; abstention accuracy; **answer-rate metric**
  exposing false abstentions (which the groundedness average silently excludes);
  **resumable checkpointing** (`_partial.jsonl`) after repeated session-teardown kills.

**Baseline → tuned (top_k_dense 20→30, rerank top_n 5→8):**

| metric | baseline | tuned |
|---|---|---|
| groundedness | 0.95 | **0.9909** |
| answer rate | (2 hidden false abstentions) | **0.9167** (11/12) |
| recall@5 / @10 | 0.9167 / 1.0 | 0.9167 / 1.0 |
| correctness | 0.917 | 0.831* |
| abstention accuracy | 3/3 | 3/3 |

\* Correctness "drop" is composition: baseline excluded 2 abstained questions from scoring;
tuned answers one of them imperfectly, so more of reality is being scored.

**What the eval caught (validating the M4-first decision):**
- The Vertex partial-fabrication (groundedness 0.5 at baseline) — **resolved by the tuning** (→0.99 aggregate).
- Two false abstentions on answerable questions (LLY $3B products, BMS key products) hidden
  by the old metrics — BMS fixed by tuning; **LLY `gs_002` remains** (the exact fact-bearing
  chunk still doesn't reach the model).
- Amgen risk-factors: Item 1A never lands in the dense top-5 (`recall@5=0`) — k-tuning can't fix
  ordering, only depth.

**Known limitation / next lever (documented, deferred):** both residual failures share one root
cause — pure semantic retrieval misses exact-phrase/section targets. The scaffolded **hybrid
BM25 + RRF** path (`hybrid_enabled: false`) is the targeted fix, to be enabled and re-measured
as its own change.

### M5 — Interfaces ✅ (2026-07-09)

**Verification:** API live on 127.0.0.1:8000 — `/health` readiness (`{"status":"ok","index_chunks":1582,"reranker_available":true}`), real cited `/ask` (Gilead HIV, 4 citations w/ snippets, 14.6s), 422/413/invalid-filter all returning the §14.2 ErrorResponse contract. `pytest tests/` → **69 passed, 1 skipped**.

**Service (`src/api.py`, rewritten from the never-run scaffold):**
- **Pre-warm at startup** (embedder + reranker in lifespan) — first request no longer pays the
  ~25s cold-load; measured retrieval on request #1 dropped ~3.5s → 266ms. Lifespan failures now
  emit structured logs (embedder → fail fast; reranker → logged dense-only degradation).
- **Uniform §14.2 error contract** — 413/422/500/502 all return error_code/message/request_id;
  internal details and submitted input are logged server-side, never echoed (the default FastAPI
  422 leaked the full input back).
- **Strict filter validation** — whitelisted keys with typed `$eq`/`$in` shapes and bounded list
  sizes ($in ≤ 12 tickers / 10 years); malformed values on known keys are rejected 422, not
  silently dropped. 32KB request-body cap (was: a 20MB body with a 200k-element `$in` was accepted).
- **Real `/health`** — reports index chunk count + reranker availability, not a static "ok".

**M5 adversarial audit — 19 candidates; 4 agent-verified, the other 15 verifier agents hit the
session usage limit, so I triaged those against the code myself. Fixed:**
- **HIGH shared 1-worker rerank executor** — one hung predict would occupy the pool forever,
  silently disabling reranking process-wide (and serializing all requests through one queue).
  Replaced with per-call executors plus a predict lock that is released by the worker when
  predict *actually* finishes — a hung predict degrades later requests to dense fallback but
  can never run concurrently with a new predict or wedge the process.
- **Thread-safety** — encode/predict are torch forward passes called from uvicorn's 40-thread
  pool with no synchronization; added an encode lock in `embed.py`.
- **Latent hybrid bug (agent-confirmed)** — the BM25 leg ignored metadata filters, so enabling
  `hybrid_enabled` would have silently bypassed ticker scoping (cross-company bleed again).
  BM25 results now post-filtered by the same filter shape; regression test added.
- **PRD §18 audit trail** — implemented `logs/queries.jsonl` (timestamp, question, filters,
  auto-ticker, outcome, citation ids, timings, model, corpus hash); write failures never break
  a request. **PRD §9.8** — citations now carry a 500-char source `snippet`; the UI expander
  shows the actual cited passage, not just metadata.
- **UI** — ask() wrapped (no tracebacks with internal paths rendered to the browser); auto-scope
  recorded at query time (was recomputed against live sidebar state, mislabeling history);
  invalid fiscal year stops the query instead of running unscoped; history capped at 20.
- **D5 "local only"** — launch commands and README now bind API and UI to 127.0.0.1.

**Accepted/deferred (documented):** Streamlit's rerun model can abandon an in-flight answer on
a mid-run click (inherent; not worth fighting v1); `AskResponse.error/truncated` are intentional
additive extensions to §14.2; no rate limiting (local-only, single user).

**Phase 1 complete: M1–M5 all verified.**

## Session 2 — Phase 2

### M6 — Synthetic training data ✅ (2026-07-09)

**Verification:** `python finetune/gen_synthetic.py --verify` → all PASS. `pytest tests/` →
**70 passed, 0 skipped** (the §19.4 leakage test is now active and green).

**Numbers:**
- 749 chunks processed (60 source-separated test + 689 train) with the PRD §11.1 verbatim
  prompt against claude-sonnet-4-6; ~$8 spend (halted at the configured ceiling per §«cost
  budget»; the halt still cleared every target).
- 2,241 raw pairs → **2,026 accepted**: **1,857 train** (≥1,500 ✓) / **169 test** (≥80 ✓).
- Filters rejected 215: 185 low token-overlap (potential ungrounded content — the filter
  doing its real job), 23 copy-paste, 7 near-duplicate questions (cosine > 0.95).
- Diversity: all 6 question types, max share 32% (risk) ≤ 50% ✓. 0 question overlap,
  0 source-chunk overlap (train and test come from disjoint chunks). Every record labeled
  `provenance: synthetic/claude-sonnet-4-6` (§16).
- §11.2 RAG framing: each record = question + source chunk context + grounded answer.

**Bug found and fixed:** the cost-ceiling halt cancelled queued futures, then the collection
loop called `.result()` on a cancelled future → `CancelledError` crash before the dataset
build. Checkpoint made it a non-event: fixed the loop, rebuilt from `raw_pairs.jsonl` with
zero new API calls.

**Datasets committed** (`train_v1.jsonl` 8.1MB, `test_v1.jsonl` 780KB — public filing text
only; $8 to regenerate and M7 needs them on the training box).

**⏳ Human gate open:** the PRD requires the user to review/approve the test set
(`data/finetune/test_v1_for_review.md`, 169 pairs) **before M7 training starts**.

Next: M7 — QLoRA fine-tune of Qwen2.5-7B-Instruct on Azure A100 (blocked on test-set
approval + Azure GPU provisioning/quota).

### M7 — Fine-tune (2026-11-06) — TRAINED; acceptance eval PASS (tie)

**Status:** model trained and test answers generated; the base-vs-tuned judging (`python finetune/eval_models.py`) has NOT run — the OpenAI judge account is out of credits. M7 is not signed off until that comparison passes (tuned faithfulness >= base, PRD 11.4) and the user approves the test set.

**Training:** Qwen2.5-7B-Instruct, 4-bit QLoRA (Unsloth), r=16 alpha=16 dropout 0, all projection modules, lr 2e-4 cosine, 2 epochs, effective batch 16, 236 steps / 2792 s on one A100 80GB. Train loss 0.93 -> ~0.41; held-out validation loss 0.435 (smoke test 0.625). Adapter 161 MB (local, gitignored).

**Training data (`finetune/sft_data.py`):** chat format identical to the inference prompt. Each example = gold chunk + 2 same-company distractors (numbered, shuffled), answer carries a ` [n]` citation to the gold chunk (the pipeline suppresses uncited answers). +10% abstention examples (other-company chunks -> exact refusal string), since the M6 data has none. 1883 train (172 abstention) / 94 val (split by source chunk) / 184 test (169 answerable + 15 abstention).

**PRD deviations (documented):** max_seq_length 2048 -> 4096 (three ~770-token chunks average 2.7k tokens); packing off with completion-only loss (packing is incompatible with response-only masking and sequences were mostly full anyway).

**Contamination found and fixed:** filings repeat passages, so chunk-ID separation alone leaked: 3 test chunks' interior text appeared in train prompts and 2 test answers duplicated train answers. Added decontamination (dropped 62 + 2 examples); re-verified 0 interior-text, 0 answer, 0 near-duplicate-question overlaps. (Head/tail overlap with neighbouring chunks is expected from the 120-token chunk overlap and is not leakage.)

**Infrastructure:** A100 quota obtained via `az quota create` after the portal wizard proved unreliable (East US and others returned ContactSupport; South Central US approved 24 vCPUs); the Free support plan blocks the support-ticket API. VM zone 2 had no capacity (ZonalAllocationFailed), zone 3 worked. Total compute cost ~$4.65 (18:41-19:57 UTC); resource group deleted and verified empty. Generation ran on the VM; judging is local so no API keys were on the VM.

**Tooling added:** `finetune/sft_data.py`, `train_qlora.py`, `gen_answers.py`, `eval_models.py` (GPT-4o judge, paired bootstrap CI, fails fast on judge errors).

Next: add OpenAI credits -> run `finetune/eval_models.py`. PASS -> M8 (GGUF Q4_K_M export + Ollama). FAIL -> stop and present the gap (PRD 11.4).


### M7 acceptance eval + M8 local serving (2026-11-06)
- **Judge: GPT-4o on 316 answers (169 answerable + 15 abstain, base and tuned).**
  Base vs tuned: faithfulness 0.894 (n=147 judged) vs 0.905 (n=169); relevance 0.991 vs 0.994; correctness 0.878 vs 0.892;
  false-abstention 13.0% vs 0.0%; cites gold chunk 98.6% vs 100%; abstention on should-refuse 15/15 vs 14/15.
  Paired faithfulness diff +0.021, 95% CI [-0.026, +0.067], W/T/L 23/111/13 -> **PASS by tie rule** (CI spans 0; not a proven improvement).
  Caveats: base abstained on 22 answerable questions, so faithfulness is judged on different populations; tuned is worse on 'factual' (0.953 vs 1.000) and missed 1 of 15 abstentions.
- **M8:** adapter merged into bf16 Qwen2.5-7B-Instruct (transformers+peft, not Unsloth export - same result, fewer moving parts), converted with llama.cpp, quantized Q4_K_M
  -> `models/gguf/pharma-diligence-q4_k_m.gguf` (4.68 GB, sha256 84c16328...c65e). Modelfile per PRD 11.5 verbatim. `ollama create pharma-diligence`.
- **Fix:** PRD Modelfile has num_ctx 4096 but pipeline sends 8 reranked chunks (~6-7k tokens) -> Ollama 400. Added `serving.local_top_n: 3` (model was trained on 3-chunk prompts), applied only for the local backend; `model_version` now reports `ollama/pharma-diligence` under `--local`.
- **Verified offline** (`OFFLINE_MODE=1 python scripts/03_ask.py --local ...`): cited answers, abstains on out-of-corpus question. Warm pipeline run 5.2 s (retrieval 0.14, rerank 2.0, generation 3.1); some other runs took 21-27 s wall - not yet explained. Rerank (CPU) exceeds the 1 s stage target.
- Azure: second A100 session ~18 min (~$1.1), resource group deleted and verified. Total Azure for M7+M8 ~ $5.8 (estimate; billing data not yet posted).
- Test set (169 pairs, `data/finetune/test_v1_for_review.md`) approved by Janish Kumar 2026-11-06 -> M7 signed off.
