"""Protocol-adherence observation: per-model tool-call rate under P2.

The propagation analysis conditions on a tool call being emitted (so an injection
can be applied). A model that shortcuts to a final answer without calling a tool
yields no injection point. We report, per model, the fraction of P2 semantic_wrong
traces that contain a parseable tool call. Closed models come from p2_all.csv;
open models from the per-model p2_<key>.csv files.
"""
import os
import glob
import json
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "results")


def rate(df):
    df = df[df["error_type"] == "semantic_wrong"]
    if len(df) == 0:
        return None, 0
    empty = (df["tool_selected"].isna() | (df["tool_selected"].astype(str).str.strip() == "")).sum()
    return round(100 * (1 - empty / len(df)), 1), len(df)


rows = {}
# closed models (aggregate file)
closed = pd.read_csv(os.path.join(BASE, "p2_all.csv"))
for m in sorted(closed["model"].unique()):
    r, n = rate(closed[closed["model"] == m])
    rows[m] = {"tool_call_rate_pct": r, "n": n, "kind": "proprietary"}
# open models (per-model files)
for f in sorted(glob.glob(os.path.join(BASE, "p2_*.csv"))):
    base = os.path.basename(f)
    if base.startswith("p2_all") or base == "p2_retail.csv":
        continue
    if base.endswith("_rejudged.csv") or base.endswith(".closed.csv") or "_v2" in base:
        continue
    key = base[len("p2_"):-len(".csv")]
    try:
        r, n = rate(pd.read_csv(f))
        rows[key] = {"tool_call_rate_pct": r, "n": n, "kind": "open-weight"}
    except Exception:
        pass

out = os.path.join(BASE, "tool_call_rates.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"{'model':22s} {'kind':12s} {'tool_call_rate':>14s}  n")
for m, v in sorted(rows.items(), key=lambda kv: (kv[1]["kind"], -(kv[1]["tool_call_rate_pct"] or 0))):
    print(f"{m:22s} {v['kind']:12s} {str(v['tool_call_rate_pct'])+'%':>14s}  {v['n']}")
print(f"\nsaved {out}")
