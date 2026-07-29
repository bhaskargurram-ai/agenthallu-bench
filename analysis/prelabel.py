"""AI pre-labeling for the 100-sample human annotation task.

Uses gpt-4o to generate YES/NO/UNCERTAIN suggestions + rationale for the
deterministic 100-sample stratified subset that human_label.py presents.
Output format matches what human_label_review.py expects.

Usage:
    python analysis/prelabel.py --output data/results/prelabel_gpt4o.csv

After this, run analysis/human_label_review.py --annotator yoshitha
to review/confirm each suggestion in ~15 min.
"""

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.model_router import ModelRouter
from analysis.human_label import stratified_sample, load_task_queries

PROMPT = """You are judging whether an AI agent's answer is substantively correct.

QUERY: {query}

REFERENCE ANSWER (ground truth): {reference}

CANDIDATE ANSWER (agent's response): {candidate}

Decide: does the CANDIDATE substantively match the REFERENCE?

Rules:
- YES if the candidate conveys the same key facts as the reference (wording can differ, extra harmless detail is OK)
- NO if the candidate contradicts the reference, is a refusal/error, is a non-answer ("I will look this up"), or fabricates facts
- UNCERTAIN only if the reference is too vague to compare (e.g. "task complete" with no specifics) AND the candidate's correctness is not obvious

Respond in this exact format:
LABEL: <YES|NO|UNCERTAIN>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASON: <one short sentence>
"""


def judge_one(row, queries, model_key="gpt4o"):
    router = ModelRouter(model_key)
    tid = row["task_id"]
    query = queries.get(tid, "")
    ref = row.get("reference", "")[:600]
    cand = row.get("candidate", "")[:600]
    msg = [{"role": "user", "content": PROMPT.format(query=query, reference=ref, candidate=cand)}]
    try:
        res = router.complete(msg, temperature=0, max_tokens=150)
        text = res["text"]
    except Exception as e:
        text = f"LABEL: UNCERTAIN\nCONFIDENCE: LOW\nREASON: API error {e}"

    # Parse
    label = "UNCERTAIN"; conf = "LOW"; reason = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("LABEL:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ("YES", "NO", "UNCERTAIN"):
                label = v
        elif line.startswith("CONFIDENCE:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ("HIGH", "MEDIUM", "LOW"):
                conf = v
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {
        "session_id": row["session_id"],
        "task_id": tid,
        "model": row.get("model", ""),
        "query": query,
        "reference": ref,
        "candidate": cand,
        "ai_label": label,
        "ai_confidence": conf,
        "ai_reason": reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/ensemble_judge.csv")
    parser.add_argument("--output", default="data/results/prelabel_gpt4o.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--hard-fraction", type=float, default=0.7)
    parser.add_argument("--model", default="gpt4o")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(base, args.input)) as f:
        rows = list(csv.DictReader(f))
    sample = stratified_sample(rows, args.n, args.hard_fraction)
    queries = load_task_queries(base)

    print(f"Pre-labeling {len(sample)} samples with {args.model} ({args.workers} workers)...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, r, queries, args.model): r for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 10 == 0:
                print(f"  {i}/{len(sample)}")

    # Preserve deterministic order matching sample order
    order = {r["session_id"]: i for i, r in enumerate(sample)}
    results.sort(key=lambda x: order.get(x["session_id"], 1e9))

    out_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    # Summary
    from collections import Counter
    print("\nLabel distribution:", Counter(r["ai_label"] for r in results))
    print("Confidence:", Counter(r["ai_confidence"] for r in results))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
