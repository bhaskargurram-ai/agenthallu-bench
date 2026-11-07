"""Verify empirically the EPS decomposition derived from metrics/eps_scorer.py + experiments/run_p2_experiment.py.

Code-derived claim (see REPORT.md, Sec. 3):
  applied  = first emitted Action names a tool in ToolExecutor.schemas (only then is the injector invoked/logged)
  T        = some Observation in the trace carries validator errors or a tool result dict with key 'error'
  W_h      = runtime heuristic judged the answer wrong (final_correct == False == heuristic_correct False)
  EPS      = applied * (T + W_h)   in {0,1,2}
  reached_output = applied * W_h
Checks every row of the P2 semantic_wrong data against these identities.
"""
import sys, json; sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from A_common import *
from collections import Counter, defaultdict
TOOLS = {'get_weather','get_forecast','get_historical','create_event','get_events','delete_event',
         'get_patient_record','check_drug_interaction','schedule_appointment','search_products','get_product','place_order'}
p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
out = {}
viol = Counter()
per_model = defaultdict(Counter)
for r in p2:
    eps = int(r["eps"]); H = tf(r["heuristic_correct"]); F = tf(r["final_correct"]); ro = tf(r["reached_output"])
    applied = r["tool_selected"] in TOOLS
    if F != H: viol["final_correct!=heuristic_correct"] += 1
    if not applied and eps != 0: viol["eps>0 but no valid first tool"] += 1
    if applied and (not H) and eps == 0: viol["applied & heuristic wrong but eps==0"] += 1
    if eps == 2 and H: viol["eps==2 but heuristic correct"] += 1
    if ro != (applied and not H): viol["reached_output != applied & W_h"] += 1
    if ro and eps == 0: viol["reached_output but eps 0"] += 1
    # inferred tool-error indicator
    T = (eps - (0 if H else 1)) if applied else None
    m = r["model"]; c = per_model[m]
    c["n"] += 1; c["applied"] += applied; c["no_tool"] += (r["tool_selected"] == "")
    c["invalid_tool_name"] += (r["tool_selected"] not in TOOLS and r["tool_selected"] != "")
    c["S1"] += eps >= 1; c["S2"] += eps >= 2
    c["W_h"] += (not H); c["W_ens"] += (not tf(r["ensemble_correct"]))
    if applied:
        c["T"] += (T == 1); c["T_invalid"] += (T not in (0, 1))
        c["applied_Wh"] += (not H)
    # S1 composition
    if eps >= 1:
        c["S1_via_Wh_only"] += (T == 0); c["S1_via_T_only"] += (T == 1 and H); c["S1_via_both"] += (T == 1 and not H)
print("identity violations:", dict(viol) or "none")
tab = {}
for m in sorted(per_model):
    c = per_model[m]; n = c["n"]
    tab[m] = dict(c)
    print(f"{m:16s} n={n:3d} noTool={c['no_tool']:3d} badName={c['invalid_tool_name']:3d} applied={c['applied']:3d} "
          f"S1={c['S1']:3d} S2={c['S2']:3d} T|applied={c['T']:3d} Wh={c['W_h']:3d} Wens={c['W_ens']:3d} "
          f"S1: Wh-only={c['S1_via_Wh_only']} T-only={c['S1_via_T_only']} both={c['S1_via_both']}")
tot = Counter()
for c in per_model.values(): tot.update(c)
print("TOTAL", dict(tot))
# tool_selected detail for low-S1 models
for m in ["gemini_20_flash", "deepseek_v3", "gpt4o_mini"]:
    print(m, Counter(r["tool_selected"] for r in p2 if r["model"] == m).most_common(8))
# what do no-tool answers look like
ex = [r["final_answer"][:100] for r in p2 if r["tool_selected"] == ""][:12]
print("no-tool final answers sample:", ex)
print("no-tool by model x heuristic_correct:", Counter((r["model"], r["heuristic_correct"]) for r in p2 if r["tool_selected"]==""))
json.dump({"identity_violations": dict(viol), "per_model_decomposition": tab, "total": dict(tot)},
          open(os.path.join(OUT, "eps_mechanism.json"), "w"), indent=2)
