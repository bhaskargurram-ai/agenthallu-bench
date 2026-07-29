"""Merge locally-served open-weight model results into the rejudged CSVs.

Idempotent: reads the original closed-model rejudged files plus any
data/results/{p2,baseline}_<localmodel>_rejudged.csv that exist, aligns
columns, and writes *_v2.csv files that carry both closed and open models.
Downstream analysis (hop_analysis, table/figure generators) points at the
_v2 files. Originals are never mutated.
"""
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "results")
# Main tables use only clean zero-shot ReAct tool-users (tool-call rate ~99%):
# the Qwen2.5 3B/7B/14B scale ladder. Mistral-7B-v0.2 (~20% tool-call rate) and
# Phi-3.5-mini (0%) did not reliably follow the protocol, so they are reported
# separately as a protocol-adherence observation (see analysis/tool_call_rates).
LOCAL = ["qwen25_3b", "qwen25_7b", "qwen25_14b"]


def _base(name: str) -> str:
    """Prefer the closed-only backup as the merge base so finalize is idempotent
    (avoids double-counting open models if the canonical file was already swapped)."""
    closed = os.path.join(BASE, f"{name}.closed.csv")
    return closed if os.path.exists(closed) else os.path.join(BASE, f"{name}.csv")


def merge(kind: str):
    orig = pd.read_csv(_base(f"{kind}_all_rejudged"))
    frames = [orig]
    added = []
    for m in LOCAL:
        f = os.path.join(BASE, f"{kind}_{m}_rejudged.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            for c in orig.columns:
                if c not in d.columns:
                    d[c] = pd.NA
            frames.append(d[orig.columns])
            added.append(f"{m}({len(d)})")
    merged = pd.concat(frames, ignore_index=True)
    out = os.path.join(BASE, f"{kind}_all_rejudged_v2.csv")
    merged.to_csv(out, index=False)
    print(f"{kind}: {len(orig)} closed + open[{', '.join(added) or 'none'}] "
          f"= {len(merged)} rows, {merged['model'].nunique()} models -> {os.path.basename(out)}")
    return merged


def merge_baseline_heuristic():
    """Merge the heuristic baseline (baseline_all.csv + baseline_<m>.csv)."""
    orig = pd.read_csv(_base("baseline_all"))
    frames = [orig]
    added = []
    for m in LOCAL:
        f = os.path.join(BASE, f"baseline_{m}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            for c in orig.columns:
                if c not in d.columns:
                    d[c] = pd.NA
            frames.append(d[orig.columns])
            added.append(f"{m}({len(d)})")
    merged = pd.concat(frames, ignore_index=True)
    out = os.path.join(BASE, "baseline_all_v2.csv")
    merged.to_csv(out, index=False)
    print(f"baseline_heuristic: {len(orig)} + open[{', '.join(added) or 'none'}] "
          f"= {len(merged)} rows, {merged['model'].nunique()} models -> {os.path.basename(out)}")


if __name__ == "__main__":
    for k in ("p2", "baseline"):
        merge(k)
    merge_baseline_heuristic()
