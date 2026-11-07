"""C2_02: categorise L1 abstention reasons (open + proprietary) and the halueval/empty-reference artefact. Data only."""
import json, re, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import *  # noqa


def l1_cat(r):
    if "tool returned error" in r:
        return "tool_returned_error"
    if "Unknown tool" in r:
        return "unknown_tool(hallucinated tool name)"
    if "'param'" in r:
        return "placeholder_param(harness format example)"
    if "not in allowed values" in r:
        return "enum_violation"
    if "must be" in r:
        return "type_error"
    if "Missing required parameter" in r or "Unknown parameter" in r:
        return "missing/unknown_param_name"
    return "other"


def run(C, I):
    out = collections.defaultdict(lambda: [0, 0])
    for t in I:
        if tf(I[t]["abstained"]) and layer(I[t]["abstain_reason"]) == "L1":
            k = l1_cat(I[t]["abstain_reason"])
            out[k][0 if not tf(C[t]["final_correct"]) else 1] += 1
    return {k: {"saved(ctrl_wrong)": v[0], "broken(ctrl_right)": v[1]} for k, v in sorted(out.items())}


res = {}
for m in OPEN:
    for var, fn in [("full", f"interceptor_{m}.csv"), ("schemaonly", f"interceptor_{m}_schemaonly.csv")]:
        rows = read_orig(fn)
        C = {r["task_id"]: r for r in rows if r["condition"] == "no_interceptor"}
        I = {r["task_id"]: r for r in rows if r["condition"] == "early_interceptor"}
        res[f"{m}/{var}"] = run(C, I)
ctrl = read_orig("interceptor_control_600.csv"); intr = read_orig("interceptor_heldout_600.csv")
for m in ["gpt4o_mini", "gemini_20_flash"]:
    C = {r["task_id"]: r for r in ctrl if r["model"] == m}; I = {r["task_id"]: r for r in intr if r["model"] == m}
    res[f"{m}/full"] = run(C, I)
# empty-reference check (halueval) in the 500-task file
T = json.load(open(f"{ORIG}/data/tasks/full_500_tasks.json"))
emp = collections.Counter(src(t["task_id"]) for t in T if t.get("split", "test") == "test" and not t.get("correct_final_answer"))
res["empty_reference_test_tasks_by_source(full_500)"] = dict(emp)
json.dump(res, open(f"{OUT}/l1_categories.json", "w"), indent=1)
for k, v in res.items():
    print(k, v)
