"""C2_05: spot-trace arXiv v2 (paper_arxiv_v2, original-only) numbers to data files. Data only; pandas not used."""
import csv, json, sys, os, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import ORIG, REPO, OUT
csv.field_size_limit(sys.maxsize)
R = f"{ORIG}/data/results"
out = {}
# tool-call rate, Hermes (paper 37.0) using tool_call_rates.py rule
for key in ["hermes3_llama8b", "qwen25_7b", "qwen25_14b", "qwen25_3b"]:
    rows = [r for r in csv.DictReader(open(f"{R}/p2_{key}.csv")) if r["error_type"] == "semantic_wrong"]
    empty = sum(1 for r in rows if not r.get("tool_selected", "").strip() or r["tool_selected"].strip().lower() == "nan")
    out[f"toolcall_{key}"] = {"n": len(rows), "rate_pct": round(100 * (1 - empty / len(rows)), 1) if rows else None}
tc = json.load(open(f"{R}/tool_call_rates.json"))
out["tool_call_rates.json_has_hermes"] = "hermes3_llama8b" in tc
hop = json.load(open(f"{R}/hop_rates_v2.json"))
out["hop_v2_models"] = sorted(hop)
out["inj_to_err_open"] = {k: round(hop[k]["injection_to_error"]["point"], 3) for k in hop if k.startswith(("qwen", "hermes"))}
# pooled r23
s2 = sum(v["n_S2"] for v in hop.values()); s23 = sum(round(v["r_2_3"]["point"] * v["n_S2"]) for v in hop.values() if v["n_S2"])
out["pooled_r23_hop_v2"] = {"S2": s2, "S3|S2": s23, "models": len(hop)}
# released trace count
for p in [f"{ORIG}/hf_dataset", f"{REPO}/hf_dataset"]:
    for f in glob.glob(f"{p}/*.csv"):
        with open(f, newline="") as fh:
            out[f"rows:{f.replace(os.path.expanduser('~'), '~')}"] = sum(1 for _ in csv.reader(fh)) - 1
# open-weight interceptor json vs csv recomputation
oj = json.load(open(f"{ORIG}/analysis/interceptor_open_delta.json"))["models"]
out["interceptor_open_delta.json"] = oj
# judge_agreement_dual.json heuristic kappa vs each annotator
out["judge_agreement_dual"] = json.load(open(f"{R}/judge_agreement_dual.json"))["judges"]["heuristic"]
json.dump(out, open(f"{OUT}/v2_trace_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
