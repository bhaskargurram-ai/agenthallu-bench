"""C2_03: failure-type breakdown from the ORIGINAL trace DBs (read-only sqlite).
Only GPT-4o-mini and Gemini-2.0-Flash have traces (interceptor_control.db = control arm, interceptor_day8.db = interceptor arm);
the open-weight runs used a NullTracer (run_open_interceptor_local.py), so no open-weight traces exist."""
import sqlite3, json, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import *  # noqa
TR = f"{ORIG}/data/traces"


def ftypes(db):
    con = sqlite3.connect(f"file:{TR}/{db}?mode=ro", uri=True)
    out = {}
    sess = [r[0] for r in con.execute("select session_id from sessions")]
    st = collections.defaultdict(list)
    for sid, typ, pe, tr in con.execute("select session_id, step_type, param_errors, tool_result from steps order by session_id, step_number"):
        st[sid].append((typ, pe, tr))
    for s in sess:
        steps = st.get(s, [])
        obs = [x for x in steps if x[0] == "observation"]
        has_final = any(x[0] == "final" for x in steps)
        if not obs:
            k = "no_tool_executed"
        elif any("Unknown tool" in (x[2] or "") for x in obs):
            k = "unknown_tool"
        elif any(x[1] and x[1] not in ("[]", "null") for x in obs):
            k = "schema_error"
        elif any('"error"' in (x[2] or "") for x in obs):
            k = "tool_error_other"
        else:
            k = "tools_ok"
        out[s] = (k, has_final, len(obs))
    # check for open-weight models
    models = [r[0] for r in con.execute("select distinct model from sessions")]
    return out, models


res = {}
cft, cm = ftypes("interceptor_control.db")
ift, im = ftypes("interceptor_day8.db")
res["models_in_trace_dbs"] = {"interceptor_control.db": cm, "interceptor_day8.db": im}
ctrl = read_orig("interceptor_control_600.csv"); intr = read_orig("interceptor_heldout_600.csv")
for m in ["gpt4o_mini", "gemini_20_flash"]:
    C = {r["task_id"]: r for r in ctrl if r["model"] == m}; I = {r["task_id"]: r for r in intr if r["model"] == m}
    tab = collections.defaultdict(collections.Counter)
    for t in C:
        k = cft[C[t]["session_id"]][0]
        w = not tf(C[t]["final_correct"])
        ab = tf(I[t]["abstained"]); L = layer(I[t]["abstain_reason"]) if ab else None
        tab[k]["n"] += 1; tab[k]["ctrl_wrong"] += w
        tab[k]["int_abst"] += ab; tab[k]["abst_ctrl_wrong(saved)"] += ab and w; tab[k]["abst_ctrl_right(broken)"] += ab and not w
        if L: tab[k][f"abst_{L}"] += 1
    # interceptor arm's own trace type vs layer
    tab2 = collections.defaultdict(collections.Counter)
    for t in I:
        k = ift[I[t]["session_id"]][0]
        ab = tf(I[t]["abstained"]); L = layer(I[t]["abstain_reason"]) if ab else "answered"
        tab2[k][L] += 1
    res[m] = {"ctrl_trace_type_x_interceptor": {k: dict(v) for k, v in tab.items()},
              "int_trace_type_x_layer": {k: dict(v) for k, v in tab2.items()}}
json.dump(res, open(f"{OUT}/trace_failure_types.json", "w"), indent=1)
print(json.dumps(res, indent=1))
