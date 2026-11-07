import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""Corroborate the EPS/S2 mechanism on full-trace re-runs produced by workstream C
(C_interceptor/traces/p2sem_r1.jsonl: GPT-4o-mini, same agent loop + semantic_wrong injector as run_p2_experiment.py,
full step logging). These are NOT the paper's P2 traces (the original trace DB p2_day6.db is not in the repo);
they are used only to observe the mechanism. Snapshot is whatever is on disk when this script runs."""
import sys, json, os
from collections import Counter, defaultdict
OUT = _REPO + "/reanalysis/A_propagation"
TR = _REPO + "/reanalysis/C_interceptor/traces"
TOOLS = {'get_weather','get_forecast','get_historical','create_event','get_events','delete_event',
         'get_patient_record','check_drug_interaction','schedule_appointment','search_products','get_product','place_order'}
res = {}
for run in ["p2sem_r1", "clean_r1"]:
    fn = os.path.join(TR, run + ".jsonl")
    if not os.path.exists(fn): continue
    rows = []
    for line in open(fn):
        try: rows.append(json.loads(line))
        except Exception: pass
    c = Counter(); by_tool = defaultdict(Counter)
    for r in rows:
        steps = r.get("steps", [])
        c["n"] += 1
        first = steps[0] if steps else {}
        txt = first.get("llm_text", "")
        if "Action:" in txt and "Final Answer:" in txt and "tool_name" not in first:
            c["action_and_final_in_same_completion_no_exec"] += 1
        execd = [s for s in steps if "tool_name" in s]
        if not execd:
            c["no_tool_executed"] += 1; continue
        f = execd[0]; t = f["tool_name"]; bt = by_tool[t]
        bt["n"] += 1; c["tool_executed"] += 1
        if "param" in (f.get("params") or {}): bt["template_key_param"] += 1; c["template_key_param"] += 1
        if f.get("validation_errors"): bt["first_call_validation_error"] += 1; c["first_call_validation_error"] += 1
        if isinstance(f.get("tool_result"), dict) and "error" in f["tool_result"]: bt["first_call_error_dict"] += 1; c["first_call_error_dict"] += 1
        anyerr = any(s.get("validation_errors") or (isinstance(s.get("tool_result"), dict) and "error" in s["tool_result"]) for s in execd)
        c["T_any_error"] += anyerr; bt["T_any_error"] += anyerr
        later_clean_same_tool = any(s["tool_name"] == t and not s.get("validation_errors") for s in execd[1:])
        if f.get("validation_errors") and later_clean_same_tool:
            c["error_then_clean_retry_same_tool"] += 1
        inj = r.get("injection")
        if inj:
            c["injection_logged"] += 1
            # did the injection change keys or value types? (semantic_wrong should not)
            o, i = inj.get("orig") or {}, inj.get("injected") or {}
            if set(o) != set(i) or any(type(o[k]) != type(i.get(k)) for k in o): c["injection_changed_keys_or_types"] += 1
            if o == i: c["injection_noop"] += 1
    res[run] = {"summary": dict(c), "by_first_tool": {k: dict(v) for k, v in by_tool.items()}}
    print(run, dict(c))
    for k, v in sorted(by_tool.items()): print("   ", k, dict(v))
json.dump(res, open(os.path.join(OUT, "rerun_trace_mechanism.json"), "w"), indent=2)
