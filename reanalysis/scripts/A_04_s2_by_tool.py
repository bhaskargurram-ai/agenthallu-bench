"""Tool-error indicator T (inferred: T = eps - [heuristic wrong], valid on applied traces) by first tool,
to locate where EPS>=2 comes from. Also: which semantic_wrong substitution each first tool receives (from code)."""
import sys, json; sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from A_common import *
from collections import Counter, defaultdict
TOOLS = {'get_weather','get_forecast','get_historical','create_event','get_events','delete_event',
         'get_patient_record','check_drug_interaction','schedule_appointment','search_products','get_product','place_order'}
# from injector/parameter_injector.py::_semantic_wrong + agent/tool_executor.py executors
SUBST = {
 'get_weather': ("city -> alternative city (fallback 'Reykjavik')", "never returns an error dict"),
 'get_forecast': ("city -> alternative city", "never returns an error dict"),
 'get_historical': ("city -> alternative city (city checked before dates)", "no error unless the model's own dates are malformed/reversed"),
 'create_event': ("date + 180 days", "never returns an error dict"),
 'get_events': ("date + 180 days", "never returns an error dict (mock events generated)"),
 'delete_event': ("event_id -> 'EVT_9999'", "returns error 'not found' -- but so does the un-injected id, store is empty"),
 'get_patient_record': ("patient_id P00k -> P00(k+1) (else 'P999')", "error only if ID outside P001-P005"),
 'check_drug_interaction': ("drug_a -> alternative drug", "never returns an error dict"),
 'schedule_appointment': ("date + 180 days", "never returns an error dict"),
}
p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
agg = defaultdict(Counter)
for r in p2:
    if r["tool_selected"] not in TOOLS: continue
    H = tf(r["heuristic_correct"]); eps = int(r["eps"]); T = eps - (0 if H else 1)
    a = agg[r["tool_selected"]]; a["n_applied"] += 1; a["T"] += T; a["S2"] += eps >= 2; a["W_h"] += (not H)
    a["W_ens"] += (not tf(r["ensemble_correct"]))
res = {}
for t, a in sorted(agg.items(), key=lambda x: -x[1]["n_applied"]):
    res[t] = dict(a); res[t]["T_rate"] = round(a["T"]/a["n_applied"], 3)
    res[t]["injection"], res[t]["injected_call_error_behaviour"] = SUBST.get(t, ("?", "?"))
    print(f"{t:24s} n={a['n_applied']:4d} T={a['T']:4d} ({a['T']/a['n_applied']:.2f}) S2={a['S2']:3d} Wh={a['W_h']:3d} Wens={a['W_ens']:3d} | {SUBST.get(t)}")
json.dump(res, open(os.path.join(OUT, "s2_by_first_tool.json"), "w"), indent=2)
