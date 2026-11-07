"""C2_07: rescore all five interceptor models with V_validator/labels_all.csv (if present).
Joins on session_id (both arms carry unique session ids). Label column auto-detected; a row's label = True means CORRECT.
Abstained interceptor-arm rows are never relabelled (abstention != hallucination). NOT yet run: labels file absent at analysis time."""
import csv, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import *  # noqa
from C2_01_open_interceptor import analyse, arms
LAB = "/Users/challagullayoshitha/Downloads/agentprop_revision/V_validator/labels_all.csv"
if not os.path.exists(LAB):
    sys.exit("labels_all.csv not found")
rows = list(csv.DictReader(open(LAB, newline="", encoding="utf-8")))
cols = rows[0].keys()
lc = next(c for c in ["correct", "is_correct", "validator_correct", "label_correct", "label"] if c in cols)
lab = {r["session_id"]: str(r[lc]).strip().lower() in ("true", "1", "yes", "correct") for r in rows if r.get(lc, "") != ""}
out = {}


def run(C, I, name):
    keep = [t for t in C if C[t]["session_id"] in lab and (tf(I[t]["abstained"]) or I[t]["session_id"] in lab)]
    C2 = {t: C[t] for t in keep}; I2 = {t: I[t] for t in keep}
    fn = lambda r: (not lab[r["session_id"]]) if r["session_id"] in lab else True
    out[name] = {"coverage_of_labels": len(keep), "validator": analyse(C2, I2, label_fn=fn), "heuristic_same_subset": analyse(C2, I2)}


for m in OPEN:
    for var, fn in [("full", f"interceptor_{m}.csv"), ("schemaonly", f"interceptor_{m}_schemaonly.csv")]:
        C, I = arms(read_orig(fn), m); run(C, I, f"{m}/{var}")
ctrl = read_orig("interceptor_control_600.csv"); intr = read_orig("interceptor_heldout_600.csv")
for m in ["gpt4o_mini", "gemini_20_flash"]:
    run({r["task_id"]: r for r in ctrl if r["model"] == m}, {r["task_id"]: r for r in intr if r["model"] == m}, m)
json.dump(out, open(f"{OUT}/rescored_validator_labels.json", "w"), indent=1, default=str)
for k, v in out.items():
    a, b = v["validator"], v["heuristic_same_subset"]
    print(k, v["coverage_of_labels"], "validator d=%.1f P=%.2f base=%.2f int-rand=%.1f p=%.3f" % (
        a["delta_pp(int-ctrl)"], a["precision"] or 0, a["base_rate"], a["random_abstention"]["int_minus_random_pp"],
        a["random_abstention"]["p_sim_one_sided(saved>=obs)"]), "| heuristic d=%.1f P=%.2f" % (b["delta_pp(int-ctrl)"], b["precision"] or 0))
