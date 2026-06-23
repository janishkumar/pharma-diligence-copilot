# Pharma Diligence Copilot

A local, private Q&A copilot over public pharma SEC 10-K filings.

**Owner:** Janish Pranesh Kumar
**PRD:** v1.2 — see `../PRD_Pharma_Diligence_Copilot_v1.2.md`

## Quickstart

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY

pip install -r requirements.txt

python scripts/00_check_env.py   # validate environment before anything else
python scripts/01_ingest.py      # M1: fetch and parse filings
python scripts/02_index.py       # M2: chunk, embed, store
python scripts/03_ask.py "What are Pfizer's principal risk factors?"  # M3: RAG answer
python scripts/04_eval.py        # M4: evaluation scorecard
uvicorn src.api:app --reload     # M5: API
streamlit run src/ui.py          # M5: demo UI
```

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

## Decisions

| Decision | Choice |
|----------|--------|
| Generator (Phase 1) | Anthropic `claude-sonnet-4-6` |
| Fine-tune base model | Qwen2.5-7B-Instruct |
| GPU (Phase 2) | Azure A100 |
| Training observability | TensorBoard |
| Demo UI | Streamlit port 8501 |
| Judge model | OpenAI `gpt-4o` |
