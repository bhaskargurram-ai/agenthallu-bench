"""Reviewer 1, point 1: causal (paired) effect of semantic_wrong injection on final-answer error.

Pairing: every P2 semantic_wrong trace (model m, task t) is matched to THE baseline trace of the same
(task_id, model). Each (task, model) has exactly one baseline run and one P2 run (verified in A_01), so the
pairing is 1:1; no averaging over baseline repeats is needed. Unpairable: gpt4o, gemini_25_flash (no baseline
run exists), and 9/50 o3_mini tasks.

Estimand: RD = P(wrong | injected) - P(wrong | not injected) on paired tasks.
CIs: Newcombe (1998) method-10 hybrid score CI for paired proportions (per model and pooled);
pooled additionally with a task-cluster bootstrap (tasks recur across models). McNemar exact test.

Strata (all defined from the P2 trace; see REPORT.md Sec. 3 for derivations from code):
  all           : all paired traces
  S1            : eps>=1 (paper's S1)
  S2            : eps>=2 (paper's S2)
  no_tool       : P2 trace emitted no tool call -> injector never invoked
  placebo_both_no_tool : neither the P2 nor the baseline trace executed a tool (cleanest negative control:
                  the injection cannot have acted; any RD here is run-to-run / judge-run noise or bias)
  all_p2_query_nonempty: drops 170 P2 rows whose 'query' cell is empty (the P2 re-judge passed an empty
                  QUESTION to the judges for these rows; baseline judges saw the query)
  applied       : first Action named a valid tool -> injector invoked
  applied_T1    : applied and the trace contains a tool/validator error (T=1, inferred T = eps - [heuristic wrong])
  applied_T0    : applied and no error anywhere -> the corrupted call validated and EXECUTED silently
Judges: ensemble_correct (primary), heuristic_correct, gpt4o_mini_verdict (+ gpt4o, gemini_25_flash).
"""
import sys, json, re
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from A_common import *
from A_stats import *

TOOLS = {'get_weather','get_forecast','get_historical','create_event','get_events','delete_event',
         'get_patient_record','check_drug_interaction','schedule_appointment','search_products','get_product','place_order'}
JUDGES = {"ensemble": "ensemble_correct", "heuristic": "heuristic_correct", "gpt4o_mini": "gpt4o_mini_verdict",
          "gpt4o": "gpt4o_verdict", "gemini_25_flash": "gemini_25_flash_verdict"}
MODEL_ORDER = ["deepseek_v3", "gpt35_turbo", "gpt41_mini", "gpt41_nano", "gpt4o_mini", "gemini_20_flash",
               "o3_mini", "qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b", "gpt4o", "gemini_25_flash"]

# ---- injected-value candidates (judge-free "verbatim propagation" check), from injector/parameter_injector.py
CITY_ALT = {"Paris": "Lyon", "London": "Manchester", "Tokyo": "Osaka", "New York": "Los Angeles",
            "São Paulo": "Rio de Janeiro", "Berlin": "Munich", "Sydney": "Melbourne", "Toronto": "Vancouver",
            "Chicago": "Houston", "Mumbai": "Delhi", "Lagos": "Abuja"}
CITIES = list(CITY_ALT) + ["Dubai", "Singapore", "Istanbul", "Beijing", "Moscow", "Seoul", "Mexico City",
                           "Bangkok", "Rome", "Cairo", "Madrid", "Los Angeles"]
PAT_ALT = {"P001": "P002", "P002": "P003", "P003": "P004", "P004": "P005", "P005": "P001"}
DRUG_ALT = {"metformin": "glipizide", "lisinopril": "losartan", "ibuprofen": "naproxen", "sertraline": "fluoxetine",
            "sumatriptan": "rizatriptan", "albuterol": "ipratropium", "aspirin": "acetaminophen", "warfarin": "heparin",
            "atorvastatin": "simvastatin", "cetirizine": "loratadine", "fluticasone": "budesonide"}

def injected_tokens(query):
    toks = set(); q = query or ""
    for c in CITIES:
        if c.lower() in q.lower(): toks.add(CITY_ALT.get(c, "Reykjavik"))
    for p in re.findall(r"P00\d", q): toks.add(PAT_ALT.get(p, "P999"))
    for d in re.findall(r"\d{4}-\d{2}-\d{2}", q):
        try: toks.add((datetime.strptime(d, "%Y-%m-%d") + timedelta(days=180)).strftime("%Y-%m-%d"))
        except ValueError: pass
    for dname, alt in DRUG_ALT.items():
        if dname in q.lower(): toks.add(alt)
    if re.search(r"EVT_\d+", q): toks.add("EVT_9999")
    toks.add("_WRONG")
    return {t for t in toks if t.lower() not in q.lower()}

def mentions(ans, toks):
    a = (ans or "").lower()
    return int(any(t.lower() in a for t in toks))

def build_pairs():
    p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
    bl = {(r["task_id"], r["model"]): r for r in read("baseline_all_rejudged.csv")}
    pairs, unpaired = [], defaultdict(int)
    for r in p2:
        b = bl.get((r["task_id"], r["model"]))
        if b is None: unpaired[r["model"]] += 1; continue
        eps = int(r["eps"]); H = tf(r["heuristic_correct"]); applied = r["tool_selected"] in TOOLS
        T = (eps - (0 if H else 1)) if applied else None
        q = b["query"] or r["query"]; toks = injected_tokens(q)
        d = dict(task=r["task_id"], model=r["model"], domain=r["domain"], eps=eps, applied=applied, T=T,
                 no_tool=(r["tool_selected"] == ""), bl_no_tool=(b["tool_selected"] == ""),
                 same_first_tool=(r["tool_selected"] == b["tool_selected"]), p2_query_empty=(r["query"].strip() == ""),
                 tok_inj=mentions(r["final_answer"], toks), tok_bl=mentions(b["final_answer"], toks))
        for j, col in JUDGES.items():
            d["inj_wrong_" + j] = int(not tf(r[col])); d["bl_wrong_" + j] = int(not tf(b[col]))
        pairs.append(d)
    return pairs, dict(unpaired)

STRATA = {
    "all": lambda d: True, "S1": lambda d: d["eps"] >= 1, "S2": lambda d: d["eps"] >= 2,
    "no_tool": lambda d: d["no_tool"], "applied": lambda d: d["applied"],
    "applied_T1": lambda d: d["applied"] and d["T"] == 1, "applied_T0": lambda d: d["applied"] and d["T"] == 0,
    "applied_T0_same_first_tool": lambda d: d["applied"] and d["T"] == 0 and d["same_first_tool"],
    "placebo_both_no_tool": lambda d: d["no_tool"] and d["bl_no_tool"],
    "all_p2_query_nonempty": lambda d: not d["p2_query_empty"],
    "placebo_both_no_tool_query_nonempty": lambda d: d["no_tool"] and d["bl_no_tool"] and not d["p2_query_empty"],
}

def paired_summary(rows, xk, yk, cluster=False):
    a = sum(1 for d in rows if d[xk] and d[yk]); b = sum(1 for d in rows if d[xk] and not d[yk])
    c = sum(1 for d in rows if not d[xk] and d[yk]); dd = sum(1 for d in rows if not d[xk] and not d[yk])
    n = a+b+c+dd
    if n == 0: return {"n": 0}
    th, lo, hi = newcombe_paired(a, b, c, dd)
    out = {"n": n, "p_wrong_inj": (a+b)/n, "p_wrong_base": (a+c)/n, "RD": th, "RD_ci_newcombe": [lo, hi],
           "both_wrong": a, "inj_wrong_base_right": b, "inj_right_base_wrong": c, "both_right": dd,
           "mcnemar_exact_p": mcnemar_exact(b, c), "mcnemar_cc_chi2_p": list(mcnemar_chi2_cc(b, c))}
    if cluster:
        out["RD_ci_task_cluster_boot"] = list(cluster_boot_rd([(d["task"], d[xk], d[yk]) for d in rows], "task"))
    return out

def main():
    pairs, unpaired = build_pairs()
    res = {"n_pairs": len(pairs), "unpaired_p2_traces_by_model": unpaired,
           "pairing_rule": "1:1 on (task_id, model); exactly one baseline run per (task_id, model) exists",
           "pooled": {}, "per_model": {}, "verbatim_injected_token": {}}
    for j in JUDGES:
        res["pooled"][j] = {s: paired_summary([d for d in pairs if f(d)], "inj_wrong_" + j, "bl_wrong_" + j, cluster=True)
                            for s, f in STRATA.items()}
    models = [m for m in MODEL_ORDER if any(d["model"] == m for d in pairs)]
    for m in models:
        sub = [d for d in pairs if d["model"] == m]
        res["per_model"][m] = {j: {s: paired_summary([d for d in sub if f(d)], "inj_wrong_" + j, "bl_wrong_" + j)
                                   for s, f in STRATA.items()} for j in JUDGES}
    # judge-free check: does the final answer contain the injected substitute value?
    for s, f in STRATA.items():
        res["verbatim_injected_token"][s] = paired_summary([d for d in pairs if f(d)], "tok_inj", "tok_bl", cluster=True)
    res["verbatim_injected_token_per_model_all"] = {m: paired_summary([d for d in pairs if d["model"] == m], "tok_inj", "tok_bl") for m in models}
    res["stratum_sizes"] = {s: sum(1 for d in pairs if f(d)) for s, f in STRATA.items()}
    res["same_first_tool_rate"] = sum(d["same_first_tool"] for d in pairs) / len(pairs)
    json.dump(res, open(os.path.join(OUT, "paired_rd.json"), "w"), indent=2)
    # console
    print("pairs", len(pairs), "unpaired", unpaired, "same first tool %.3f" % res["same_first_tool_rate"])
    print("stratum sizes", res["stratum_sizes"])
    for j in ["ensemble", "heuristic", "gpt4o_mini"]:
        print("== judge", j)
        for s, v in res["pooled"][j].items():
            if v["n"]:
                print(f"  {s:28s} n={v['n']:4d} Pinj={v['p_wrong_inj']:.3f} Pbl={v['p_wrong_base']:.3f} RD={v['RD']:+.3f} "
                      f"NC[{v['RD_ci_newcombe'][0]:+.3f},{v['RD_ci_newcombe'][1]:+.3f}] CB[{v['RD_ci_task_cluster_boot'][0]:+.3f},{v['RD_ci_task_cluster_boot'][1]:+.3f}] "
                      f"b={v['inj_wrong_base_right']} c={v['inj_right_base_wrong']} p={v['mcnemar_exact_p']:.2g}")
    print("== verbatim injected token in final answer")
    for s, v in res["verbatim_injected_token"].items():
        if v["n"]: print(f"  {s:28s} n={v['n']} Pinj={v['p_wrong_inj']:.3f} Pbl={v['p_wrong_base']:.3f} RD={v['RD']:+.3f} p={v['mcnemar_exact_p']:.2g}")
    print("== per model, ensemble, all / applied_T0 / S2")
    for m in models:
        line = m
        for s in ["all", "applied_T0", "S2", "no_tool"]:
            v = res["per_model"][m]["ensemble"][s]
            line += f" | {s}: n={v['n']}" + (f" RD={v['RD']:+.3f}[{v['RD_ci_newcombe'][0]:+.2f},{v['RD_ci_newcombe'][1]:+.2f}] p={v['mcnemar_exact_p']:.2g}" if v["n"] else "")
        print(line)

if __name__ == "__main__":
    main()
