#!/usr/bin/env python3
"""M6: Generate synthetic Q&A training data from the indexed corpus (PRD §11.1).

- Verbatim PRD generation prompt against claude-sonnet-4-6 (D1).
- Source separation: a held-out set of chunks is used EXCLUSIVELY for test
  pairs, so test questions come from chunks never seen in training data.
- Quality filters applied in PRD order; embedding dedup (cosine > 0.95).
- Checkpointed per chunk (raw_pairs.jsonl) — safe to kill and resume.
- Hard cost ceiling; halts cleanly when reached.

Usage:
  EMBED_DEVICE=cpu python finetune/gen_synthetic.py            # generate (resumable)
  EMBED_DEVICE=cpu python finetune/gen_synthetic.py --verify   # M6 acceptance gate
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import cfg, ANTHROPIC_API_KEY  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

log = get_logger("gen_synthetic")

OUT_DIR = ROOT / "data" / "finetune"
RAW_PATH = OUT_DIR / "raw_pairs.jsonl"          # checkpoint: one record per processed chunk
TRAIN_PATH = OUT_DIR / "train_v1.jsonl"
TEST_PATH = OUT_DIR / "test_v1.jsonl"
REVIEW_PATH = OUT_DIR / "test_v1_for_review.md"
STATS_PATH = OUT_DIR / "gen_stats.json"

SEED = 42
GEN_MODEL = cfg["generation"]["model"]           # D1 hosted model per PRD §11.1
N_TEST_CHUNKS = 60                                # held out exclusively for test pairs
MAX_TRAIN_CHUNKS = 950                            # budget-driven cap (~$7 at ~$0.007/call)
COST_CEILING_USD = 8.0
CONCURRENCY = 6
PRICE_IN, PRICE_OUT = 3.0, 15.0                   # $/1M tokens, claude-sonnet-4-6

QUESTION_TYPES = {"factual", "financial", "risk", "concentration", "comparative", "summarization"}

# PRD §11.1 generation prompt — VERBATIM.
SYSTEM_PROMPT = """You generate diligence-style question-answer pairs strictly grounded in the provided text. Output only valid JSON.

Rules:
- Generate 2 to 3 Q&A pairs from the CHUNK.
- Each ANSWER must be fully supported by the CHUNK. Do not introduce facts not present.
- Vary question types across: factual, financial, risk, concentration, comparative-within-text, summarization.
- Avoid trivial questions whose answer is a single named entity from the first sentence.
- Avoid questions whose answer is "yes" or "no" without a substantive supporting clause.
- Keep questions natural-sounding, as an analyst would ask."""

USER_TEMPLATE = """CHUNK (from {company} {fiscal_year} 10-K, {section}):
{chunk_text}

Return JSON in this exact shape:
{{
  "pairs": [
    {{"question": "...", "answer": "...", "question_type": "factual|financial|risk|concentration|comparative|summarization"}}
  ]
}}"""

_STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with
which their they them we our us not no than then over under more most other such may can could would should about""".split())

_cost_lock = threading.Lock()
_total_cost = 0.0


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def load_chunks() -> list[dict]:
    from src.index import store
    col = store.get_collection()
    data = col.get(include=["documents", "metadatas"])
    chunks = [
        {"chunk_id": cid, "text": doc, "meta": meta}
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    chunks.sort(key=lambda c: c["chunk_id"])  # deterministic before shuffle
    random.Random(SEED).shuffle(chunks)
    return chunks


def generate_for_chunk(chunk: dict) -> dict:
    """One API call -> raw record (checkpointable). Never raises."""
    global _total_cost
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    meta = chunk["meta"]
    user_msg = USER_TEMPLATE.format(
        company=meta.get("company", ""), fiscal_year=meta.get("fiscal_year", ""),
        section=meta.get("section", ""), chunk_text=chunk["text"],
    )
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=GEN_MODEL, max_tokens=1000, temperature=0.4,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            cost = resp.usage.input_tokens / 1e6 * PRICE_IN + resp.usage.output_tokens / 1e6 * PRICE_OUT
            with _cost_lock:
                _total_cost += cost
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            m = re.search(r"\{.*\}", text, re.DOTALL)
            pairs = json.loads(m.group()) ["pairs"] if m else []
            return {"chunk_id": chunk["chunk_id"], "pairs": pairs, "cost": round(cost, 5)}
        except Exception as e:
            if attempt == 2:
                log.warning("chunk_failed", chunk_id=chunk["chunk_id"], error=str(e)[:120])
                return {"chunk_id": chunk["chunk_id"], "pairs": [], "cost": 0.0, "error": str(e)[:120]}
            time.sleep(2 ** attempt)


def run_generation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    test_chunks = chunks[:N_TEST_CHUNKS]
    train_chunks = chunks[N_TEST_CHUNKS:N_TEST_CHUNKS + MAX_TRAIN_CHUNKS]
    test_ids = {c["chunk_id"] for c in test_chunks}

    done_ids = set()
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["chunk_id"])
    todo = [c for c in test_chunks + train_chunks if c["chunk_id"] not in done_ids]
    print(f"Chunks: {len(test_chunks)} test / {len(train_chunks)} train | done {len(done_ids)} | todo {len(todo)}")

    halted = False
    written = 0
    with open(RAW_PATH, "a") as f, ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(generate_for_chunk, c): c for c in todo}
        for fut in as_completed(futures):
            if fut.cancelled():   # ceiling halt cancels queued calls
                continue
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{len(todo)} chunks | cost so far ${_total_cost:.2f}", flush=True)
            if _total_cost >= COST_CEILING_USD and not halted:
                halted = True
                print(f"COST CEILING ${COST_CEILING_USD} reached — cancelling remaining calls", flush=True)
                for other in futures:
                    other.cancel()
    print(f"Generation pass done: {written} new chunks, session cost ${_total_cost:.2f}"
          + (" (HALTED at ceiling — rerun to continue under a fresh budget)" if halted else ""))
    build_datasets(test_ids)


def build_datasets(test_ids: set[str] | None = None):
    """Apply PRD §11.1 filters in order, dedup, diversity check, and write
    source-separated train/test JSONL."""
    if test_ids is None:
        chunks = load_chunks()
        test_ids = {c["chunk_id"] for c in chunks[:N_TEST_CHUNKS]}

    # chunk_id -> text (for filters 2 & 3)
    chunk_text = {c["chunk_id"]: c["text"] for c in load_chunks()}
    chunk_meta = {c["chunk_id"]: c["meta"] for c in load_chunks()}

    raw = []
    for line in RAW_PATH.read_text().splitlines():
        if line.strip():
            raw.append(json.loads(line))

    candidates = []
    rejects = {"short_answer": 0, "copy_paste": 0, "low_overlap": 0, "short_question": 0, "bad_type": 0, "dup": 0}
    for rec in raw:
        ctext = chunk_text.get(rec["chunk_id"], "")
        if not ctext:
            continue
        for p in rec.get("pairs", []):
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            qtype = (p.get("question_type") or "").strip().lower()
            if qtype not in QUESTION_TYPES:
                rejects["bad_type"] += 1
                continue
            if len(a) < 15:                                   # filter 1
                rejects["short_answer"] += 1
                continue
            if len(a) > 0.5 * len(ctext):                     # filter 2
                rejects["copy_paste"] += 1
                continue
            a_tok = _tokens(a)                                # filter 3
            if a_tok and len(a_tok & _tokens(ctext)) / len(a_tok) < 0.5:
                rejects["low_overlap"] += 1
                continue
            if len(q.split()) < 6:                            # filter 4
                rejects["short_question"] += 1
                continue
            candidates.append({"question": q, "answer": a, "question_type": qtype,
                               "chunk_id": rec["chunk_id"]})

    # Filter 5: embedding dedup (cosine > 0.95 vs any PRIOR accepted question).
    import numpy as np
    from src.index import embed
    model = embed.get_model()
    prefix = cfg["embeddings"]["query_instruction"]
    print(f"Embedding {len(candidates)} candidate questions for dedup…", flush=True)
    vecs = model.encode([prefix + c["question"] for c in candidates],
                        batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    accepted_idx: list[int] = []
    kept = np.zeros((0, vecs.shape[1]), dtype=vecs.dtype)
    for i, v in enumerate(vecs):
        if kept.shape[0] and float((kept @ v).max()) > 0.95:
            rejects["dup"] += 1
            continue
        accepted_idx.append(i)
        kept = np.vstack([kept, v])
    accepted = [candidates[i] for i in accepted_idx]

    train = [p for p in accepted if p["chunk_id"] not in test_ids]
    test = [p for p in accepted if p["chunk_id"] in test_ids]

    def to_record(p):
        meta = chunk_meta.get(p["chunk_id"], {})
        return {
            "question": p["question"],
            "context": chunk_text.get(p["chunk_id"], ""),   # §11.2: RAG-framed
            "answer": p["answer"],
            "question_type": p["question_type"],
            "source": {"chunk_id": p["chunk_id"], "ticker": meta.get("ticker"),
                       "section": meta.get("section"), "fiscal_year": meta.get("fiscal_year")},
            "provenance": "synthetic/" + GEN_MODEL,          # §16: labeled as model-generated
        }

    with open(TRAIN_PATH, "w") as f:
        for p in train:
            f.write(json.dumps(to_record(p)) + "\n")
    with open(TEST_PATH, "w") as f:
        for p in test:
            f.write(json.dumps(to_record(p)) + "\n")

    # Human-review file for the test set (PRD: user approves before training).
    lines = ["# Test set for human review (approve before M7 training)\n"]
    for i, p in enumerate(test, 1):
        meta = chunk_meta.get(p["chunk_id"], {})
        lines.append(f"### {i}. [{p['question_type']}] {meta.get('ticker')} {meta.get('section')}")
        lines.append(f"**Q:** {p['question']}")
        lines.append(f"**A:** {p['answer']}\n")
    REVIEW_PATH.write_text("\n".join(lines))

    from collections import Counter
    type_counts = Counter(p["question_type"] for p in accepted)
    stats = {
        "raw_pairs": sum(len(r.get("pairs", [])) for r in raw),
        "accepted": len(accepted), "train": len(train), "test": len(test),
        "rejects": rejects, "type_counts": dict(type_counts),
        "session_cost_usd": round(_total_cost, 2),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def verify():
    """M6 acceptance gate (PRD §6): >=1500 train, >=80 test, 0 overlap,
    source separation, diversity."""
    ok = True
    train = [json.loads(l) for l in TRAIN_PATH.read_text().splitlines() if l.strip()]
    test = [json.loads(l) for l in TEST_PATH.read_text().splitlines() if l.strip()]

    print(f"train pairs: {len(train)} (need >= 1500): {'PASS' if len(train) >= 1500 else 'FAIL'}")
    print(f"test pairs:  {len(test)} (need >= 80): {'PASS' if len(test) >= 80 else 'FAIL'}")
    ok &= len(train) >= 1500 and len(test) >= 80

    train_qs = {p["question"] for p in train}
    overlap = sum(1 for p in test if p["question"] in train_qs)
    print(f"question overlap: {overlap} (need 0): {'PASS' if overlap == 0 else 'FAIL'}")
    ok &= overlap == 0

    train_chunks = {p["source"]["chunk_id"] for p in train}
    chunk_overlap = sum(1 for p in test if p["source"]["chunk_id"] in train_chunks)
    print(f"source-chunk overlap: {chunk_overlap} (need 0): {'PASS' if chunk_overlap == 0 else 'FAIL'}")
    ok &= chunk_overlap == 0

    from collections import Counter
    types = Counter(p["question_type"] for p in train + test)
    total = sum(types.values())
    max_share = max(types.values()) / total if total else 1
    print(f"question types: {dict(types)}")
    print(f"diversity: {len(types)} types (need >= 4), max share {max_share:.0%} (need <= 50%): "
          f"{'PASS' if len(types) >= 4 and max_share <= 0.5 else 'FAIL'}")
    ok &= len(types) >= 4 and max_share <= 0.5

    labeled = all(p.get("provenance", "").startswith("synthetic/") for p in train + test)
    print(f"provenance labeled: {'PASS' if labeled else 'FAIL'}")
    ok &= labeled

    print("\nM6 verification " + ("PASSED" if ok else "FAILED"))
    print("NOTE: the test set additionally requires HUMAN review/approval before M7 "
          f"training starts — see {REVIEW_PATH}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="re-run filters/split from the raw checkpoint without new API calls")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.rebuild:
        build_datasets()
    else:
        run_generation()
