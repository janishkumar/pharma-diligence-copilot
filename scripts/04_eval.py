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


def section_recall_at_k(candidates: list[dict], ticker: str, expected_sections: list[str], k: int) -> float:
    """Did the top-k RETRIEVED candidates include a chunk from the expected
    company+section? Section-level ground truth (we don't know exact chunk ids
    a priori). candidates come from the pipeline trace, in retrieval order."""
    if not expected_sections:
        return None  # not applicable (e.g. abstention questions)
    want = set(expected_sections)
    for c in candidates[:k]:
        if (not ticker or c.get("ticker") == ticker) and c.get("section") in want:
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

    # Resumable checkpoint: each completed question is appended here, so a killed
    # run resumes instead of restarting (the run kept getting torn down).
    partial_path = REPORTS_DIR / "_partial.jsonl"
    done = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"Resuming: {len(done)}/{len(golden)} already completed")

    results = []
    pf = partial_path.open("a")
    for item in golden:
        qid = item["id"]
        if qid in done:
            results.append(done[qid])
            print(f"  [{qid}] cached")
            continue

        q = item["question"]
        ref = item.get("reference_answer", "")
        ticker = item.get("ticker")
        expected = item.get("expected_sections", [])
        should_abstain = item.get("should_abstain", False)
        print(f"  [{qid}] {q[:70]}...", end=" ", flush=True)

        trace = {}
        result = ask(q, _trace=trace)
        answer = result.answer
        candidates = trace.get("candidates", [])

        recall = None
        if not should_abstain and expected:
            recall = {
                "r5": section_recall_at_k(candidates, ticker, expected, 5),
                "r10": section_recall_at_k(candidates, ticker, expected, 10),
                "r20": section_recall_at_k(candidates, ticker, expected, 20),
            }

        judge_scores = {}
        if not should_abstain and not result.abstained:
            context_text = "\n\n".join(
                f"[{i}] {c['text']}" for i, c in enumerate(trace.get("reranked", []), 1)
            )
            try:
                judge_scores = judge_answer(q, context_text, ref, answer)
            except Exception as e:
                log.warning("judge_failed", question_id=qid, error=str(e))

        rec = {
            "id": qid, "question": q, "answer": answer[:500],
            "abstained": result.abstained, "should_abstain": should_abstain,
            "ref_present": bool(ref), "recall": recall, "judge": judge_scores,
            "retrieved_sections": [c.get("section") for c in candidates[:5]],
            "timing_ms": result.timing.total_ms,
        }
        results.append(rec)
        pf.write(json.dumps(rec) + "\n"); pf.flush()
        print(f"done ({result.timing.total_ms}ms)")
    pf.close()

    # aggregate from per-question results (resilient to resume)
    recall_k5 = [r["recall"]["r5"] for r in results if r.get("recall")]
    recall_k10 = [r["recall"]["r10"] for r in results if r.get("recall")]
    recall_k20 = [r["recall"]["r20"] for r in results if r.get("recall")]
    groundedness_scores = [r["judge"]["groundedness"] for r in results if r.get("judge") and "groundedness" in r["judge"]]
    relevance_scores = [r["judge"]["relevance"] for r in results if r.get("judge") and "relevance" in r["judge"]]
    correctness_scores = [r["judge"]["correctness"] for r in results if r.get("judge") and r.get("ref_present") and "correctness" in r["judge"]]
    abstain_total = sum(1 for r in results if r["should_abstain"])
    abstain_correct = sum(1 for r in results if r["should_abstain"] and r["abstained"])

    # Answer rate on ANSWERABLE questions — exposes false abstentions that the
    # groundedness average hides (abstained answers are not judged/scored).
    answerable = [r for r in results if not r["should_abstain"]]
    answered = [r for r in answerable if not r["abstained"]]
    answer_rate = round(len(answered) / len(answerable), 4) if answerable else None
    false_abstentions = [r["id"] for r in answerable if r["abstained"]]

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    scorecard = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": f"{cfg['generation']['backend']}/{cfg['generation']['model']}",
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
            "correctness": round(avg(correctness_scores), 4),
        },
        "abstention": {
            "correct": abstain_correct,
            "total": abstain_total,
            "accuracy": round(abstain_correct / abstain_total, 4) if abstain_total else None,
        },
        "answering": {
            "answer_rate": answer_rate,
            "answered": len(answered),
            "answerable": len(answerable),
            "false_abstentions": false_abstentions,
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
        f"- Correctness: {scorecard['generation']['correctness']}",
        "\n## Answering (answerable questions)",
        f"- Answer rate: {answer_rate} ({len(answered)}/{len(answerable)})",
        f"- False abstentions: {false_abstentions or 'none'}",
        "\n## Abstention (negative questions)",
        f"- Accuracy: {scorecard['abstention']['accuracy']} ({abstain_correct}/{abstain_total})",
        f"\n**Phase 1 acceptance:** {'PASS ✓' if scorecard['phase1_pass'] else 'FAIL ✗'}",
    ]
    md_path.write_text("\n".join(md_lines))
    if latest_md.is_symlink():
        latest_md.unlink()
    latest_md.symlink_to(md_path.name)

    partial_path.unlink(missing_ok=True)  # scorecard written; clear the checkpoint

    print(f"\nScorecard written to {json_path}")
    print(f"Latest: {latest_md}")
    print(f"\nrecall@10: {scorecard['retrieval']['recall@10']}  |  groundedness: {scorecard['generation']['groundedness']}")
    print(f"Phase 1 acceptance: {'PASS' if scorecard['phase1_pass'] else 'FAIL'}")
    partial_path.unlink(missing_ok=True)  # scorecard committed; clear the checkpoint
    return scorecard


if __name__ == "__main__":
    run()
