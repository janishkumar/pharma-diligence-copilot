#!/usr/bin/env python3
"""M7 acceptance (PRD 6 / 11.4): BASE vs FINE-TUNED on the held-out test set.

Reads the answers generated on the GPU VM (finetune/gen_answers.py) and judges
them locally with GPT-4o (a different model family from the Claude-generated
training data and from the Qwen candidates — judge independence per PRD 19).
No API key ever touches the VM.

Pass rule: tuned faithfulness >= base. A statistical tie (diff within -0.02 and
the 95% CI spanning 0) counts as "match". Anything worse FAILS and per PRD 11.4
we stop before quantization.

Usage: python finetune/eval_models.py [--answers-dir data/finetune/answers]
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OPENAI_API_KEY, cfg  # noqa: E402
from src.generate.prompt import ABSTENTION_STRING  # noqa: E402

DATA = ROOT / "data" / "finetune"
REPORTS = ROOT / "eval" / "reports"
CACHE = REPORTS / "finetune_judge_cache.jsonl"
JUDGE_MODEL = cfg["eval"]["judge_model"]
SEED = cfg["eval"]["seed"]

JUDGE_PROMPT = """You are an impartial grader. Given a QUESTION, the CONTEXT the answer was written from, a REFERENCE answer, and the SYSTEM ANSWER, score the SYSTEM ANSWER:

1. groundedness (0-1): every claim (especially every number, name and date) must be supported by the CONTEXT. Penalize any unsupported or invented claim. Citation markers like [2] are not claims.
2. relevance (0-1): the answer directly addresses the QUESTION.
3. correctness (0-1): the answer agrees with the REFERENCE answer.

Return strictly JSON: {{"groundedness": <float>, "relevance": <float>, "correctness": <float>, "notes": "<short>"}}.

QUESTION: {q}
CONTEXT: {ctx}
REFERENCE: {ref}
SYSTEM ANSWER: {ans}"""


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def split_prompt(user_msg: str) -> tuple[str, str]:
    q = user_msg.split("QUESTION:\n", 1)[1].split("\n\nCONTEXT:", 1)[0].strip()
    ctx = user_msg.split("CONTEXT:\n", 1)[1].rsplit("\n\nANSWER:", 1)[0]
    return q, ctx


def is_abstention(text: str) -> bool:
    norm = text.strip().lstrip('>#*-").“”\' ').lower()
    return norm.startswith(ABSTENTION_STRING.rstrip(".").lower())


def cited(text: str) -> set[int]:
    return {int(n) for g in re.findall(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]", text)
            for n in re.findall(r"\d+", g)}


def judge(client, q, ctx, ref, ans) -> dict:
    r = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, seed=SEED, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, ctx=ctx, ref=ref, ans=ans)}],
    )
    return json.loads(r.choices[0].message.content)


def bootstrap_ci(diffs: list[float], n=5000):
    rng = random.Random(SEED)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers-dir", default=str(DATA / "answers"))
    args = ap.parse_args()
    adir = Path(args.answers_dir)

    test = load_jsonl(DATA / "test_sft.jsonl")
    ans = {tag: {r["i"]: r for r in load_jsonl(adir / f"answers_{tag}.jsonl")} for tag in ("base", "tuned")}
    for tag in ans:
        assert len(ans[tag]) == len(test), f"{tag}: {len(ans[tag])} answers for {len(test)} prompts"

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            if l.strip():
                c = json.loads(l); cache[(c["tag"], c["i"])] = c["j"]

    jobs = []  # answerable items where the model actually answered
    for i, rec in enumerate(test):
        if rec["meta"]["abstain"]:
            continue
        for tag in ("base", "tuned"):
            if not is_abstention(ans[tag][i]["answer"]) and (tag, i) not in cache:
                jobs.append((tag, i))
    print(f"judging {len(jobs)} answers ({len(cache)} cached) with {JUDGE_MODEL}")

    def run(job):
        tag, i = job
        q, ctx = split_prompt(test[i]["messages"][1]["content"])
        try:
            return tag, i, judge(client, q, ctx, test[i]["meta"]["reference"], ans[tag][i]["answer"])
        except Exception as e:
            return tag, i, {"error": str(e)[:120]}

    REPORTS.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with ThreadPoolExecutor(6) as ex, open(CACHE, "a") as f:
        for n, (tag, i, j) in enumerate(ex.map(run, jobs), 1):
            if "error" not in j:
                cache[(tag, i)] = j
                f.write(json.dumps({"tag": tag, "i": i, "j": j}) + "\n"); f.flush()
            else:
                errors.append(j["error"])
                if len(errors) >= 5 and not cache:
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(f"ABORT: judge failing on every call ({len(errors)} errors so far). First error: {errors[0]}\n"
                             "Nothing was judged; fix the cause (e.g. OpenAI credits/key) and re-run — cached results resume.")
            if n % 40 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    # ---- aggregate ----
    answerable = [i for i, r in enumerate(test) if not r["meta"]["abstain"]]
    abstain_items = [i for i, r in enumerate(test) if r["meta"]["abstain"]]
    res = {}
    for tag in ("base", "tuned"):
        judged = [i for i in answerable if (tag, i) in cache]
        g = [cache[(tag, i)]["groundedness"] for i in judged]
        corr = [cache[(tag, i)]["correctness"] for i in judged]
        rel = [cache[(tag, i)]["relevance"] for i in judged]
        mean = lambda x: round(sum(x) / len(x), 4) if x else None
        gold_cited = sum(1 for i in judged if test[i]["meta"]["gold_position"] in cited(ans[tag][i]["answer"]))
        any_cite = sum(1 for i in judged if cited(ans[tag][i]["answer"]))
        n_ctx = 3
        bad_cite = sum(1 for i in judged if any(n < 1 or n > n_ctx for n in cited(ans[tag][i]["answer"])))
        res[tag] = {
            "n_answerable": len(answerable), "n_judged": len(judged),
            "false_abstention_rate": round(1 - len(judged) / len(answerable), 4),
            "faithfulness": mean(g), "relevance": mean(rel), "correctness": mean(corr),
            "cites_gold_chunk": round(gold_cited / len(judged), 4) if judged else None,
            "has_any_citation": round(any_cite / len(judged), 4) if judged else None,
            "out_of_range_citation": round(bad_cite / len(judged), 4) if judged else None,
            "abstain_correct": f"{sum(1 for i in abstain_items if is_abstention(ans[tag][i]['answer']))}/{len(abstain_items)}",
        }

    # paired faithfulness difference on items BOTH models answered
    both = [i for i in answerable if ("base", i) in cache and ("tuned", i) in cache]
    if not both:
        sys.exit(f"ABORT: no answer pairs were judged ({len(errors)} judge errors). First: {errors[:1]}")
    diffs = [cache[("tuned", i)]["groundedness"] - cache[("base", i)]["groundedness"] for i in both]
    d = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)
    wins, ties, losses = sum(x > 0 for x in diffs), sum(x == 0 for x in diffs), sum(x < 0 for x in diffs)
    if d >= 0:
        verdict = "PASS (tuned >= base)"
    elif d >= -0.02 and lo <= 0 <= hi:
        verdict = "PASS (statistical tie / match)"
    else:
        verdict = "FAIL (tuned worse than base) — per PRD 11.4 do NOT quantize; present gap to user"

    # by question type
    bytype = {}
    for i in both:
        t = test[i]["meta"]["question_type"]
        bytype.setdefault(t, []).append((cache[("base", i)]["groundedness"], cache[("tuned", i)]["groundedness"]))
    bytype = {t: {"n": len(v), "base": round(sum(b for b, _ in v) / len(v), 3), "tuned": round(sum(t_ for _, t_ in v) / len(v), 3)}
              for t, v in sorted(bytype.items())}

    report = {"timestamp": datetime.utcnow().isoformat(), "judge": JUDGE_MODEL, "seed": SEED,
              "models": res, "paired_faithfulness": {"n": len(both), "mean_diff_tuned_minus_base": round(d, 4),
              "ci95": [round(lo, 4), round(hi, 4)], "tuned_wins": wins, "ties": ties, "tuned_losses": losses},
              "faithfulness_by_type": bytype, "verdict": verdict}
    out = REPORTS / f"finetune_eval_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    sys.exit(0 if verdict.startswith("PASS") else 1)


if __name__ == "__main__":
    main()
