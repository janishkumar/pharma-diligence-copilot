# Baseline Metrics

Populated at M4 (first eval runs). Regressions are measured against the **tuned** column.
Reports: `eval/reports/report_20260630_042455.json` (baseline), `report_20260702_034358.json` (tuned).

| Metric | Baseline (2026-06-30) | Tuned (2026-07-02) | Notes |
|--------|----------------------|--------------------|-------|
| Index size (disk) | 85 MB | 85 MB | ChromaDB, `.chroma/` |
| Chunk count | 1,582 | 1,582 | 800-token chunks / 120 overlap |
| recall@5 (section-level) | 0.9167 | 0.9167 | golden set N=12 answerable |
| recall@10 | 1.0 | 1.0 | |
| Groundedness (GPT-4o judge) | 0.95 | **0.9909** | full retrieved context judged |
| Relevance | 1.0 | 1.0 | |
| Correctness (vs reference) | 0.917 | 0.831 | tuned scores more questions (see CHANGELOG) |
| Answer rate (answerable) | 10/12 | **11/12 (0.9167)** | residual false abstention: gs_002 |
| Abstention accuracy | 3/3 | 3/3 | |
| Latency p50 (per query, CPU) | ~10 s | ~10 s | first query ~26 s (model load) |
| Embedding model | nomic-embed-text-v1.5 (768d) | same | `EMBED_DEVICE=cpu` for batch work |
| Generator | anthropic/claude-sonnet-4-6 | same | temperature 0.1 |
| Reranker | BAAI/bge-reranker-base | same | top_n 5 → **8**, timeout 30 s |
| Retrieval | dense top_k 20 | dense top_k **30** | hybrid BM25 scaffolded, disabled |
| Judge | openai/gpt-4o | same | seed 42, temperature 0 |
| Eval config | seed 42 | seed 42 | golden_set_v1.json (15 q) |
