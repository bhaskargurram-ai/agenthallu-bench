"""Shared loaders for A_* analyses (AgentProp-Bench revision, propagation stats)."""
import csv, os, sys
csv.field_size_limit(sys.maxsize)
REPO = "/Users/challagullayoshitha/Downloads/agentprop_repo/src"
RES = os.path.join(REPO, "data", "results")
OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/A_propagation"
P2_FILES = ["p2_all_rejudged.csv", "p2_qwen25_3b_rejudged.csv", "p2_qwen25_7b_rejudged.csv",
            "p2_qwen25_14b_rejudged.csv", "p2_hermes3_llama8b_rejudged.csv"]
BL_FILES = ["baseline_all_rejudged.csv", "baseline_qwen25_3b_rejudged.csv", "baseline_qwen25_7b_rejudged.csv",
            "baseline_qwen25_14b_rejudged.csv", "baseline_hermes3_llama8b_rejudged.csv"]

def read(fn):
    with open(os.path.join(RES, fn), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_file"] = fn
    return rows

def tf(v):
    """Parse a boolean-ish CSV cell. Returns True/False/None."""
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"): return True
    if s in ("false", "0", "0.0", "no"): return False
    return None

def dedup(rows, key):
    seen, out, dups = set(), [], 0
    for r in rows:
        k = key(r)
        if k in seen:
            dups += 1; continue
        seen.add(k); out.append(r)
    return out, dups
