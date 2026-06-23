#!/usr/bin/env python3
"""M4: Run the evaluation harness and generate a scorecard."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import cfg
from src.pipeline import ask
from src.logging_setup import get_logger

log = get_logger("04_eval")

GOLDEN_PATH = ROOT / cfg["eval"]["golden_set_path"]
REPORTS_DIR = ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = cfg["eval"]["seed"]
ABSTENTION_STRING = cfg["generation"]["abstention_string"]


def load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        print(f"Golden set not found at {GOLDEN_PATH}. Create it before running eval.")
        sys.exit(1)
    return json.loads(GOLDEN_PATH.read_text())


def recall_at_k(retrieved_ids: list[str], expected_sources: list[dict], k: int) -> float:
    top_k_ids = set(retrieved_ids[:k])
    for src in expected_sources:
        chunk_id = src.get("chunk_id")
        if chunk_id and chunk_id in top_k_ids:
            return 1.0
    return 0.0


def judge_answer(question: str, context: str, reference: str, system_answer: str) -> dict:
    judge_model = cfg["eval"]["judge_model"]
    from openai import OpenAI
    from src.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""You are an impartial grader. Given a QUESTION, the CONTEXT used to answer it, the REFERENCE answer, and the SYSTEM ANSWER, score the SYSTEM ANSWER on:

1. Groundedness (0-1): every claim in the SYSTEM ANSWER must be supported by the CONTEXT. Penalize unsupported claims.
2. Relevance (0-1): the SYSTEM ANSWER directly addresses the QUESTION.
3. Correctness (0-1): the SYSTEM ANSWER aligns with the REFERENCE answer.

Return strictly JSON: {{"groundedness": <float>, "relevance": <float>, "correctness": <float>, "notes": "<short rationale>"}}.

QUESTION: {question}
CONTEXT: {context}
REFERENCE: {reference}
SYSTEM ANSWER: {system_answer}"""

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=SEED,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def run():
    golden = load_golden()
    print(f"Loaded {len(golden)} golden questions from {GOLDEN_PATH}")

    results = []
    recall_k5, recall_k10, recall_k20 = [], [], []
    groundedness_scores, relevance_scores = [], []
    abstain_correct = 0
    abstain_total = 0

    for item in golden:
        q = item["question"]
        ref = item.get("reference_answer", "")
        expected = item.get("expected_sources", [])
        should_abstain = item.get("should_abstain", False)

        print(f"  [{item['id']}] {q[:80]}...", end=" ", flush=True)

        result = ask(q)
        answer = result.answer
        retrieved_ids = [c.chunk_id for c in result.citations] if result.citations else []

        # retrieval metrics (skip for abstain items)
        if not should_abstain and expected:
            recall_k5.append(recall_at_k(retrieved_ids, expected, 5))
            recall_k10.append(recall_at_k(retrieved_ids, expected, 10))
            recall_k20.append(recall_at_k(retrieved_ids, expected, 20))

        # abstention check
        if should_abstain:
            abstain_total += 1
            if result.abstained:
                abstain_correct += 1

        # judge scoring (skip abstain items and items without reference)
        judge_scores = {}
        if not should_abstain and ref and not result.abstained:
            context_text = " | ".join(c.text[:200] for c in (result.citations or []))
            try:
                judge_scores = judge_answer(q, context_text, ref, answer)
                groundedness_scores.append(judge_scores.get("groundedness", 0))
                relevance_scores.append(judge_scores.get("relevance", 0))
            except Exception as e:
                log.warning("judge_failed", question_id=item["id"], error=str(e))

        results.append({
            "id": item["id"],
            "question": q,
            "answer": answer,
            "abstained": result.abstained,
            "should_abstain": should_abstain,
            "retrieved_chunk_ids": retrieved_ids,
            "judge": judge_scores,
            "timing_ms": result.timing.total_ms,
        })
        print("done")

    # aggregate
    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    scorecard = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": results[0]["answer"] if results else "",
        "seed": SEED,
        "n_questions": len(golden),
        "retrieval": {
            "recall@5": round(avg(recall_k5), 4),
            "recall@10": round(avg(recall_k10), 4),
            "recall@20": round(avg(recall_k20), 4),
        },
        "generation": {
            "groundedness": round(avg(groundedness_scores), 4),
            "relevance": round(avg(relevance_scores), 4),
        },
        "abstention": {
            "correct": abstain_correct,
            "total": abstain_total,
            "accuracy": round(abstain_correct / abstain_total, 4) if abstain_total else None,
        },
        "per_question": results,
        "phase1_pass": avg(recall_k10) >= 0.85 and avg(groundedness_scores) >= 0.90,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"report_{ts}.json"
    md_path = REPORTS_DIR / f"report_{ts}.md"
    latest_md = REPORTS_DIR / "latest.md"

    json_path.write_text(json.dumps(scorecard, indent=2))

    md_lines = [
        f"# Eval Report — {ts}",
        f"\n**Model:** {scorecard['model_version']}  **Seed:** {SEED}  **N:** {len(golden)}",
        "\n## Retrieval",
        f"- recall@5: {scorecard['retrieval']['recall@5']}",
        f"- recall@10: {scorecard['retrieval']['recall@10']} {'✓ PASS' if scorecard['retrieval']['recall@10'] >= 0.85 else '✗ FAIL (need >= 0.85)'}",
        f"- recall@20: {scorecard['retrieval']['recall@20']}",
        "\n## Generation",
        f"- Groundedness: {scorecard['generation']['groundedness']} {'✓ PASS' if scorecard['generation']['groundedness'] >= 0.90 else '✗ FAIL (need >= 0.90)'}",
        f"- Relevance: {scorecard['generation']['relevance']}",
        "\n## Abstention",
        f"- Accuracy: {scorecard['abstention']['accuracy']} ({abstain_correct}/{abstain_total})",
        f"\n**Phase 1 acceptance:** {'PASS ✓' if scorecard['phase1_pass'] else 'FAIL ✗'}",
    ]
    md_path.write_text("\n".join(md_lines))
    if latest_md.is_symlink():
        latest_md.unlink()
    latest_md.symlink_to(md_path.name)

    print(f"\nScorecard written to {json_path}")
    print(f"Latest: {latest_md}")
    print(f"\nrecall@10: {scorecard['retrieval']['recall@10']}  |  groundedness: {scorecard['generation']['groundedness']}")
    print(f"Phase 1 acceptance: {'PASS' if scorecard['phase1_pass'] else 'FAIL'}")
    return scorecard


if __name__ == "__main__":
    run()
