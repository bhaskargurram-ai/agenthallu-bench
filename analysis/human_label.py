"""Fix 3: Human labeling CLI for 3-annotator inter-rater agreement study.

Each annotator runs:
  python analysis/human_label.py --annotator <NAME> --n 100

The script:
  1. Stratifies 100 samples from ensemble_judge.csv (70 hard cases + 30 easy controls)
  2. Shows query / reference / candidate and prompts [y]es / [n]o / [s]kip / [q]uit
  3. Saves incrementally to data/results/human_labels_<annotator>.csv
  4. Is resumable — already-labeled rows skipped.

After all annotators finish, run analysis/compute_human_kappa.py
to compute:
  - κ between each annotator and ensemble majority
  - Fleiss κ among the 3 annotators
  - κ between annotators and heuristic
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ENSEMBLE_CSV = "data/results/ensemble_judge.csv"
TASKS_GLOB = "data/tasks/*_tasks.json"
OUTPUT_DIR = "data/results"


def load_task_queries(base: str) -> dict[str, str]:
    """Build task_id -> query map from all tasks JSON files."""
    mapping = {}
    for path in glob.glob(os.path.join(base, TASKS_GLOB)):
        try:
            with open(path) as f:
                for t in json.load(f):
                    tid = t.get("task_id")
                    q = t.get("query") or t.get("question") or ""
                    if tid:
                        mapping[tid] = q
        except Exception:
            continue
    return mapping


def stratified_sample(rows: list[dict], n: int, hard_fraction: float = 0.7) -> list[dict]:
    """Sample n rows: `hard_fraction` from 2-1 splits (hard cases), rest unanimous."""
    rng = random.Random(42)

    hard = []
    easy = []
    for r in rows:
        verdicts = []
        for k in ["gpt4o_verdict", "gemini_25_flash_verdict", "gpt4o_mini_verdict"]:
            v = str(r.get(k, "")).strip().lower()
            verdicts.append(v in ("true", "1", "yes"))
        s = sum(verdicts)
        if s in (1, 2):  # split — hard case
            hard.append(r)
        elif s in (0, 3):
            easy.append(r)
    rng.shuffle(hard)
    rng.shuffle(easy)

    n_hard = int(n * hard_fraction)
    n_easy = n - n_hard
    pick = hard[:n_hard] + easy[:n_easy]
    rng.shuffle(pick)
    return pick[:n]


def load_existing(path: str) -> dict[str, str]:
    """Return {session_id: label} of already-labeled rows."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["session_id"]] = row["label"]
    return out


def save_row(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True, help="Annotator identifier (e.g. A1, A2, yourname)")
    parser.add_argument("--n", type=int, default=100, help="Target number of labels")
    parser.add_argument("--hard-fraction", type=float, default=0.7)
    parser.add_argument("--input", default=ENSEMBLE_CSV)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    in_path = os.path.join(base, args.input)
    out_path = os.path.join(base, OUTPUT_DIR, f"human_labels_{args.annotator}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run M1 first.")
        sys.exit(1)

    with open(in_path) as f:
        rows = list(csv.DictReader(f))

    # Deterministic stratified sample — same set across annotators (crucial for κ).
    sample = stratified_sample(rows, args.n, args.hard_fraction)
    done = load_existing(out_path)

    # Look up the real query text for each task_id
    queries = load_task_queries(base)

    print(f"\n{'='*70}")
    print(f"  ANNOTATOR: {args.annotator}")
    print(f"  Total samples: {len(sample)}   Already labeled: {len(done)}")
    print(f"  Output: {out_path}")
    print(f"{'='*70}")
    print("  Judge whether the AGENT ANSWER is substantively correct vs reference.")
    print("  Keys: [y]=YES correct  [n]=NO incorrect  [s]=skip  [q]=save and quit")
    print(f"{'='*70}\n")

    labeled = 0
    for i, r in enumerate(sample, 1):
        sid = r["session_id"]
        if sid in done:
            continue
        tid = r.get("task_id", "?")
        query_text = queries.get(tid, "(query text not found)")
        print(f"\n─── [{i}/{len(sample)}]  session_id={sid}  model={r.get('model','?')}  task={tid} ───")
        print(f"QUERY     : {query_text[:500]}")
        print(f"REFERENCE : {r.get('reference','')[:500]}")
        print(f"CANDIDATE : {r.get('candidate','')[:500]}")
        # Show ensemble votes to help annotator understand hard cases
        votes = []
        for k, label in [("gpt4o_verdict","gpt4o"),
                         ("gemini_25_flash_verdict","gemini"),
                         ("gpt4o_mini_verdict","gpt4om")]:
            v = str(r.get(k,"")).strip().lower()
            votes.append(f"{label}={'Y' if v in ('true','1','yes') else 'N'}")
        print(f"(LLM judges: {' '.join(votes)}; heuristic={'Y' if str(r.get('heuristic_correct','')).lower() in ('true','1') else 'N'})")

        while True:
            choice = input("YOUR VERDICT [y/n/s/q]: ").strip().lower()
            if choice in ("y", "n", "s", "q"):
                break
            print("Please type y, n, s, or q.")

        if choice == "q":
            print("Saving and exiting.")
            break
        if choice == "s":
            continue

        label = "YES" if choice == "y" else "NO"
        save_row(out_path, {
            "session_id": sid,
            "task_id": r.get("task_id", ""),
            "model": r.get("model", ""),
            "annotator": args.annotator,
            "label": label,
            "ensemble_majority": r.get("majority", ""),
            "heuristic_correct": r.get("heuristic_correct", ""),
        })
        labeled += 1
        done[sid] = label

    print(f"\n{'='*70}")
    print(f"  Done. Labeled this session: {labeled}.  Total in file: {len(done)}/{len(sample)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
