"""Compute inter-annotator agreement and agreement vs ensemble/heuristic.

Metrics:
  - Fleiss κ among the N annotators (tests consistency of humans)
  - Cohen κ of each annotator vs ensemble majority
  - Cohen κ of each annotator vs heuristic
  - Cohen κ of ensemble majority vs heuristic (the headline: if this is low
    AND humans agree with ensemble, the story writes itself)

Output:
  analysis/human_kappa.md  (human-readable)
  data/results/human_kappa.json  (machine-readable)
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_y = sum(a) / n
    pb_y = sum(b) / n
    pe = pa_y * pb_y + (1 - pa_y) * (1 - pb_y)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """Fleiss κ for N subjects × k categories of ratings.

    matrix[i][j] = number of raters who assigned subject i to category j.
    Each row sums to N (number of raters).
    """
    if not matrix:
        return 0.0
    N = len(matrix)
    n = sum(matrix[0])  # raters per subject
    if n < 2:
        return 0.0
    k = len(matrix[0])

    # Marginal category proportions
    p = [sum(row[j] for row in matrix) / (N * n) for j in range(k)]

    # Per-subject agreement
    P_i = []
    for row in matrix:
        s = sum(x * (x - 1) for x in row) / (n * (n - 1))
        P_i.append(s)

    P_bar = sum(P_i) / N
    P_e = sum(pj ** 2 for pj in p)
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    return (P_bar - P_e) / (1 - P_e)


def to_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default="data/results")
    parser.add_argument("--output-md", default="analysis/human_kappa.md")
    parser.add_argument("--output-json", default="data/results/human_kappa.json")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    label_files = sorted(glob.glob(os.path.join(base, args.labels_dir, "human_labels_*.csv")))
    if not label_files:
        print("ERROR: no human_labels_*.csv files found in", args.labels_dir)
        sys.exit(1)

    # Load each annotator's labels: annotator -> {session_id: YES/NO}
    annotators = {}
    for path in label_files:
        name = os.path.basename(path).replace("human_labels_", "").replace(".csv", "")
        with open(path) as f:
            r = csv.DictReader(f)
            annotators[name] = {row["session_id"]: row["label"] for row in r}

    # Intersection of session IDs labeled by ALL annotators
    common = set.intersection(*[set(d.keys()) for d in annotators.values()])
    common = sorted(common)
    print(f"Annotators: {list(annotators.keys())}")
    print(f"Sessions labeled by all: {len(common)}")
    if len(common) < 10:
        print("WARNING: fewer than 10 common labels — κ may be unstable.")

    # Also load ensemble + heuristic from ensemble_judge.csv
    ens_path = os.path.join(base, "data/results/ensemble_judge.csv")
    ens = {}
    if os.path.exists(ens_path):
        with open(ens_path) as f:
            r = csv.DictReader(f)
            for row in r:
                ens[row["session_id"]] = {
                    "majority": to_bool(row.get("majority", "")),
                    "heuristic": to_bool(row.get("heuristic_correct", "")),
                }

    # Restrict to sessions with ensemble info
    common = [s for s in common if s in ens]
    print(f"Common ∩ ensemble: {len(common)}")

    # Build Fleiss matrix: each session, count YES votes across annotators (binary Y/N)
    ann_names = sorted(annotators.keys())
    fleiss_mat = []
    for s in common:
        yes = sum(1 for a in ann_names if annotators[a][s] == "YES")
        no = len(ann_names) - yes
        fleiss_mat.append([yes, no])
    f_kappa = fleiss_kappa(fleiss_mat)

    # Per-annotator κ vs ensemble majority and vs heuristic
    metrics = {
        "n_common": len(common),
        "n_annotators": len(ann_names),
        "fleiss_kappa_humans": round(f_kappa, 4),
        "per_annotator": {},
    }

    for a in ann_names:
        a_labels = [annotators[a][s] == "YES" for s in common]
        e_labels = [ens[s]["majority"] for s in common]
        h_labels = [ens[s]["heuristic"] for s in common]
        metrics["per_annotator"][a] = {
            "n": len(common),
            "kappa_vs_ensemble": round(cohens_kappa(a_labels, e_labels), 4),
            "kappa_vs_heuristic": round(cohens_kappa(a_labels, h_labels), 4),
            "agreement_vs_ensemble": round(sum(1 for x,y in zip(a_labels,e_labels) if x==y) / len(common), 4),
        }

    # Meta: ensemble vs heuristic on just these samples
    e_labels = [ens[s]["majority"] for s in common]
    h_labels = [ens[s]["heuristic"] for s in common]
    metrics["ensemble_vs_heuristic_kappa"] = round(cohens_kappa(e_labels, h_labels), 4)

    os.makedirs(os.path.dirname(os.path.join(base, args.output_json)), exist_ok=True)
    with open(os.path.join(base, args.output_json), "w") as f:
        json.dump(metrics, f, indent=2)

    md = ["# Human-Annotator Agreement Study (Fix 3)\n",
          f"- Annotators: {', '.join(ann_names)}",
          f"- Samples labeled by all: {len(common)}",
          f"- **Fleiss κ among humans: {f_kappa:.3f}**",
          f"- Ensemble vs heuristic κ (on these samples): {metrics['ensemble_vs_heuristic_kappa']:.3f}",
          "",
          "| Annotator | κ vs ensemble | κ vs heuristic | agreement vs ensemble |",
          "|-----------|---------------|-----------------|------------------------|"]
    for a in ann_names:
        m = metrics["per_annotator"][a]
        md.append(f"| {a} | {m['kappa_vs_ensemble']:.3f} | {m['kappa_vs_heuristic']:.3f} | {m['agreement_vs_ensemble']*100:.1f}% |")
    md_str = "\n".join(md)

    with open(os.path.join(base, args.output_md), "w") as f:
        f.write(md_str)

    print(md_str)
    print(f"\nSaved: {args.output_md}, {args.output_json}")


if __name__ == "__main__":
    main()
