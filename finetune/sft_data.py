#!/usr/bin/env python3
"""Build chat-format SFT data (M7) from the M6 synthetic pairs.

Matches the INFERENCE prompt exactly (src.generate.prompt) so the fine-tuned
model learns the real task: numbered context chunks -> grounded answer with an
[n] citation, or the exact abstention string.

Per answerable example: gold chunk + 2 same-company distractors, shuffled; the
answer gets " [gold_position]" appended (synthetic answers carry no markers,
and the pipeline suppresses uncited answers).
~10% abstention examples: question + chunks from a DIFFERENT company -> the
exact abstention string (M6 data contains no refusals).

Leakage control: test chunks are never used as distractors in train/val, val is
split by source chunk, and adjacent/overlapping chunks of the gold chunk are
excluded as distractors (they could contain the answer).

Outputs (data/finetune/): train_sft.jsonl, val_sft.jsonl, test_sft.jsonl
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generate.prompt import ABSTENTION_STRING, SYSTEM_PROMPT, build_user_message  # noqa: E402
from src.index import store  # noqa: E402

OUT = ROOT / "data" / "finetune"
SEED = 42
N_DISTRACTORS = 2
ABSTAIN_FRAC = 0.10
VAL_CHUNK_FRAC = 0.05
N_TEST_ABSTAIN = 15


def load_pool() -> dict[str, dict]:
    col = store.get_collection()
    d = col.get(include=["documents", "metadatas"])
    return {
        cid: {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(d["ids"], d["documents"], d["metadatas"])
    }


def _adjacent(a: dict, b: dict) -> bool:
    ma, mb = a["metadata"], b["metadata"]
    return (
        ma["ticker"] == mb["ticker"]
        and ma["section"] == mb["section"]
        and abs(int(ma["chunk_index_in_section"]) - int(mb["chunk_index_in_section"])) <= 1
    )


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]


def make_answerable(row, pool, distractor_ids_by_ticker, rng):
    gold = pool[row["source"]["chunk_id"]]
    cands = [pool[c] for c in distractor_ids_by_ticker[gold["metadata"]["ticker"]]
             if c != gold["chunk_id"] and not _adjacent(pool[c], gold)]
    chunks = rng.sample(cands, N_DISTRACTORS) + [gold]
    rng.shuffle(chunks)
    pos = next(i for i, c in enumerate(chunks, 1) if c["chunk_id"] == gold["chunk_id"])
    msgs = build_prompt(row["question"], chunks)
    msgs.append({"role": "assistant", "content": f"{row['answer'].rstrip()} [{pos}]"})
    return {"messages": msgs, "meta": {"abstain": False, "gold_chunk_id": gold["chunk_id"],
            "gold_position": pos, "ticker": gold["metadata"]["ticker"],
            "question_type": row["question_type"], "reference": row["answer"]}}


def make_abstain(row, pool, distractor_ids_by_ticker, rng):
    ticker = row["source"]["ticker"]
    other = rng.choice([t for t in distractor_ids_by_ticker if t != ticker])
    chunks = [pool[c] for c in rng.sample(distractor_ids_by_ticker[other], N_DISTRACTORS + 1)]
    msgs = build_prompt(row["question"], chunks)
    msgs.append({"role": "assistant", "content": ABSTENTION_STRING})
    return {"messages": msgs, "meta": {"abstain": True, "ticker": ticker,
            "question_type": row["question_type"], "reference": ABSTENTION_STRING}}


def main():
    rng = random.Random(SEED)
    pool = load_pool()
    train_rows = [json.loads(l) for l in open(OUT / "train_v1.jsonl")]
    test_rows = [json.loads(l) for l in open(OUT / "test_v1.jsonl")]
    test_chunk_ids = {r["source"]["chunk_id"] for r in test_rows}

    # Distractor pool: never a test chunk.
    by_ticker: dict[str, list[str]] = {}
    for cid, c in pool.items():
        if cid not in test_chunk_ids:
            by_ticker.setdefault(c["metadata"]["ticker"], []).append(cid)
    for t in by_ticker:
        by_ticker[t].sort()

    # Split train/val by SOURCE CHUNK so val questions come from unseen chunks.
    chunk_ids = sorted({r["source"]["chunk_id"] for r in train_rows})
    rng.shuffle(chunk_ids)
    val_chunks = set(chunk_ids[: max(1, int(len(chunk_ids) * VAL_CHUNK_FRAC))])
    tr = [r for r in train_rows if r["source"]["chunk_id"] not in val_chunks]
    va = [r for r in train_rows if r["source"]["chunk_id"] in val_chunks]

    def build(rows, abstain_n):
        out = [make_answerable(r, pool, by_ticker, rng) for r in rows]
        out += [make_abstain(r, pool, by_ticker, rng) for r in rng.sample(rows, abstain_n)]
        rng.shuffle(out)
        return out

    train = build(tr, int(len(tr) * ABSTAIN_FRAC))
    val = build(va, int(len(va) * ABSTAIN_FRAC))

    # Decontaminate against the test set. Filings repeat passages (and the same
    # boilerplate recurs across companies), so chunk-id separation is not enough.
    # Probes sit at 25/50/75% of each test chunk — safely inside it, so the
    # 120-token overlap with neighbouring chunks can't trigger them.
    probes = []
    for c in {r["context"] for r in test_rows}:
        if len(c) >= 1600:
            probes += [c[int(len(c) * f): int(len(c) * f) + 200] for f in (0.25, 0.5, 0.75)]
    test_answers = {r["answer"][:120] for r in test_rows}
    test_q_words = [set(r["question"].lower().split()) for r in test_rows]

    def contaminated(rec) -> str | None:
        user, ans = rec["messages"][1]["content"], rec["messages"][2]["content"]
        if any(p in user for p in probes):
            return "test_chunk_text"
        if any(a in ans for a in test_answers):
            return "test_answer"
        q = set(user.split("QUESTION:\n")[1].split("\n")[0].lower().split())
        if any(len(q & t) / max(len(q | t), 1) > 0.7 for t in test_q_words):
            return "test_question"
        return None

    dropped = {"test_chunk_text": 0, "test_answer": 0, "test_question": 0}
    def clean(data):
        kept = []
        for rec in data:
            why = contaminated(rec)
            if why:
                dropped[why] += 1
            else:
                kept.append(rec)
        return kept
    train, val = clean(train), clean(val)
    print("decontamination dropped (train+val):", dropped)

    # Test: gold = test chunk; distractors from the non-test pool (adjacency still excluded).
    pool_with_test = pool
    test = []
    for r in test_rows:
        gold_id = r["source"]["chunk_id"]
        # allow the gold test chunk itself through the helper
        by_ticker_t = {t: ids + ([gold_id] if t == pool[gold_id]["metadata"]["ticker"] else [])
                       for t, ids in by_ticker.items()}
        test.append(make_answerable(r, pool_with_test, by_ticker_t, rng))
    test += [make_abstain(r, pool, by_ticker, rng) for r in rng.sample(test_rows, N_TEST_ABSTAIN)]

    for name, data in (("train_sft", train), ("val_sft", val), ("test_sft", test)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")
    n_abs = lambda d: sum(1 for x in d if x["meta"]["abstain"])
    print(f"train {len(train)} ({n_abs(train)} abstain) | val {len(val)} ({n_abs(val)}) | test {len(test)} ({n_abs(test)})")
    print(f"val chunks held out: {len(val_chunks)}")


if __name__ == "__main__":
    main()
