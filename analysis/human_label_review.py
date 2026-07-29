"""Fast review CLI for AI-pre-labeled samples.

Shows each AI suggestion with rationale; user confirms with ENTER or overrides.
Low-confidence items get the full 4-field prompt (like human_label.py).

Usage:
    python analysis/human_label_review.py --annotator yoshitha

Produces:  data/results/human_labels_<annotator>.csv
Resumable — already-labeled rows are skipped.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.human_label import load_existing, save_row


def review(r: dict) -> str | None:
    """Return 'YES' / 'NO' / None (skip). None means user pressed s."""
    ai = r.get("ai_label", "UNCERTAIN").upper()
    conf = r.get("ai_confidence", "LOW").upper()
    reason = r.get("ai_reason", "")

    print(f"\nQUERY     : {r['query'][:400]}")
    print(f"REFERENCE : {r['reference'][:400]}")
    print(f"CANDIDATE : {r['candidate'][:400]}")
    print(f"\nAI suggestion: {ai}  (confidence={conf})")
    print(f"AI reason   : {reason}")

    if ai == "UNCERTAIN" or conf == "LOW":
        prompt = "  YOUR verdict [y/n/s/q]: "
    elif ai == "YES":
        prompt = "  Accept YES? [ENTER=yes, n=flip to NO, s=skip, q=quit]: "
    else:
        prompt = "  Accept NO? [ENTER=no, y=flip to YES, s=skip, q=quit]: "

    while True:
        c = input(prompt).strip().lower()
        if c == "":
            if ai == "YES":
                return "YES"
            if ai == "NO":
                return "NO"
            # UNCERTAIN / LOW -> force explicit
            print("  → please type y / n / s / q")
            continue
        if c in ("y", "n", "s", "q"):
            break
        print("  → please type y / n / s / q (or ENTER to accept the AI suggestion)")

    if c == "q":
        return "QUIT"
    if c == "s":
        return None
    return "YES" if c == "y" else "NO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--prelabel", default="data/results/prelabel_gpt4o.csv")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    pre_path = os.path.join(base, args.prelabel)
    if not os.path.exists(pre_path):
        print(f"ERROR: {pre_path} not found. Run analysis/prelabel.py first.")
        sys.exit(1)

    out_path = os.path.join(base, "data/results", f"human_labels_{args.annotator}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(pre_path) as f:
        items = list(csv.DictReader(f))

    done = load_existing(out_path)

    from collections import Counter
    conf_dist = Counter(r["ai_confidence"].upper() for r in items)

    print(f"\n{'='*70}")
    print(f"  ANNOTATOR: {args.annotator}")
    print(f"  Total: {len(items)}   Already done: {len(done)}   Remaining: {len(items)-len(done)}")
    print(f"  AI confidence: HIGH={conf_dist.get('HIGH',0)}  MED={conf_dist.get('MEDIUM',0)}  LOW={conf_dist.get('LOW',0)}")
    print(f"  Output: {out_path}")
    print(f"{'='*70}")
    print("  HIGH/MED: press ENTER to accept AI suggestion, or flip (y/n)")
    print("  LOW/UNCERTAIN: type y or n yourself")
    print("  s=skip  q=save and quit")
    print(f"{'='*70}")

    labeled_this_session = 0
    for i, r in enumerate(items, 1):
        sid = r["session_id"]
        if sid in done:
            continue
        print(f"\n─── [{i}/{len(items)}]  session_id={sid}  model={r.get('model','?')} ───")
        verdict = review(r)
        if verdict == "QUIT":
            print("Saving and exiting.")
            break
        if verdict is None:
            continue
        save_row(out_path, {
            "session_id": sid,
            "task_id": r.get("task_id", ""),
            "model": r.get("model", ""),
            "annotator": args.annotator,
            "label": verdict,
            "ai_label": r.get("ai_label", ""),
            "ai_confidence": r.get("ai_confidence", ""),
        })
        labeled_this_session += 1
        done[sid] = verdict

    print(f"\n{'='*70}")
    print(f"  Labeled this session: {labeled_this_session}.  Total: {len(done)}/{len(items)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
