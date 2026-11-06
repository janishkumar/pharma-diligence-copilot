# Pharma Diligence Copilot

A local, private Q&A copilot over public pharma SEC 10-K filings: RAG pipeline, evaluation harness, and a synthetic-data fine-tuning track toward a fully offline model.

## The problem

Diligence analysts answer the same questions about pharma companies over and over: risk factors, litigation exposure, segment revenue, pipeline concentration. The answers exist in 10-K filings, but each one is hundreds of pages, and pasting confidential deal context into a hosted chatbot is not an option. This project answers those questions from the filings themselves, cites its sources, runs on a laptop, and refuses to answer when the filings do not contain the answer.

## What it does

- Ingests and parses 10-K filings straight from SEC EDGAR
- Chunks, embeds (nomic-embed-text), and indexes them in ChromaDB
- Answers questions with retrieval, cross-encoder reranking (bge-reranker), company filtering, and Claude generation, with source citations
- Abstains explicitly when the answer is not in the corpus, instead of guessing
- Scores itself with an eval harness: section-level recall@k, groundedness, relevance, and correctness via an LLM judge, plus abstention accuracy
- Generates synthetic Q&A training pairs from the corpus for fine-tuning a local model

## How it works

```
EDGAR fetch -> parse -> chunk (800 tok / 120 overlap) -> embed -> ChromaDB
                                                                    |
question -> retrieve (dense top_k 30) -> company filter -> rerank (top 8)
         -> generate (Claude, temp 0.1, cite-or-abstain prompt) -> answer + sources
                                                                    |
eval harness: 15-question golden set -> recall@k + GPT-4o judge -> scorecard
```

## Design decisions

- Chose local-first (ChromaDB, local embedder, localhost-only API with no auth) over a cloud stack, because the target use case is diligence work where the queries themselves are confidential
- Chose an explicit abstention string over letting the model improvise refusals, because the eval harness needs to measure false answers and false abstentions separately
- Chose section-level retrieval ground truth over chunk IDs for the golden set, because chunk boundaries change with every re-index and section labels do not
- Chose a two-phase plan (Claude API now, fine-tuned Qwen2.5-7B via Ollama later) over local-only from day one, so answer quality had a strong baseline before any training run
- Chose a separate LLM judge (GPT-4o, seed 42, temperature 0) from the generator (Claude), to avoid a model grading its own homework

## Status and results

Milestones M1 through M6 complete; fine-tuning (M7) and local serving (M8) are in progress. Measured numbers, from `BASELINE.md` and the eval reports in `eval/reports/`:

| Metric (15-question golden set) | Tuned config (2026-07-02) |
|---|---|
| Retrieval recall@5 / @10 (section-level) | 0.92 / 1.0 |
| Groundedness (GPT-4o judge) | 0.99 |
| Answer rate on answerable questions | 11/12 |
| Abstention accuracy on trap questions | 3/3 |
| Latency p50 (CPU) | ~10 s |

The golden set is small (15 questions); these numbers show the harness works, not that the system is production-ready.

## Running it

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY and OPENAI_API_KEY (judge)

pip install -r requirements.txt

python scripts/00_check_env.py   # validate environment first
python scripts/01_ingest.py      # fetch and parse filings
EMBED_DEVICE=cpu python scripts/02_index.py --verify
EMBED_DEVICE=cpu python scripts/03_ask.py "What are Pfizer's principal risk factors?"
EMBED_DEVICE=cpu python scripts/04_eval.py   # evaluation scorecard

# Interfaces, localhost only by design
EMBED_DEVICE=cpu uvicorn src.api:app --host 127.0.0.1 --port 8000
EMBED_DEVICE=cpu streamlit run src/ui.py --server.address 127.0.0.1
```

`EMBED_DEVICE=cpu` is required on Apple Silicon: the nomic embedder hangs on MPS for long-sequence batches. Single queries are still fast on CPU.

## Milestones

| # | Milestone | Verification |
|---|-----------|-------------|
| M1 | Ingestion | `python scripts/01_ingest.py --verify` |
| M2 | Index | `python scripts/02_index.py --verify` |
| M3 | RAG answer | `python scripts/03_ask.py "test question"` |
| M4 | Eval harness | `python scripts/04_eval.py` |
| M5 | Interfaces | API + Streamlit both answering |
| M6 | Synthetic data | `python finetune/gen_synthetic.py --verify` |
| M7 | Fine-tune | `python finetune/eval_models.py` |
| M8 | Local serve | `ollama run pharma-diligence "test question"` |

## Stack

Python, ChromaDB, nomic-embed-text-v1.5, BAAI/bge-reranker-base, Anthropic Claude (generation), OpenAI GPT-4o (judge), FastAPI, Streamlit, pytest.

## Limitations and next steps

- Golden set is 15 questions across a handful of tickers; scaling it is cheaper than any model change and is the next eval investment
- Hybrid BM25 retrieval is scaffolded but disabled; dense-only retrieval already hits recall@10 = 1.0 on the current set, so it stays off until the eval says otherwise
- One residual false abstention (gs_002) survives tuning; tracked in `CHANGELOG.md`
- Correctness score dipped from 0.917 to 0.831 after tuning because the tuned config answers more questions; see `CHANGELOG.md` for the tradeoff
- M7/M8 (fine-tuned local model via Ollama) not yet delivered; until then the generator requires an Anthropic API key
