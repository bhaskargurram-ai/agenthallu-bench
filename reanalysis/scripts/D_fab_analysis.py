"""Fabricated-tool-use analysis for all 13 P2 models.
Usage: python3 -I D_fab_analysis.py <repo_results_dir> <D_fabrication_dir>
Inputs: p2_all_rejudged.csv, fabricated_tooluse.json, tool_call_rates.json, native_toolcall_*.json (repo);
        per_trace_fabrication.csv, labeling/llm_judge2_gpt-4o_cache.jsonl, labeling/fabrication_labeling_KEY.csv,
        db_structural_features.csv (D_fabrication)."""
import csv, json, os, sys, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D_fab_rule import PLACEHOLDER_ONLY
from D_stats import wilson, fisher_exact_2x2, cohen_kappa, spearman
csv.field_size_limit(10**9)
R, D = sys.argv[1], sys.argv[2]
P = {x["session_id"]: x for x in csv.DictReader(open(os.path.join(R, "p2_all_rejudged.csv")))}
pub = json.load(open(os.path.join(R, "fabricated_tooluse.json")))
T = {x["session_id"]: x for x in csv.DictReader(open(os.path.join(D, "per_trace_fabrication.csv")))}
J2 = {}
for l in open(os.path.join(D, "labeling", "llm_judge2_gpt-4o_cache.jsonl")):
    d = json.loads(l); J2[d["session_id"]] = d
DB = {x["session_id"]: x for x in csv.DictReader(open(os.path.join(D, "db_structural_features.csv")))}
OPEN = {"qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b"}
FAMILY = {"gpt4o": "OpenAI-GPT", "gpt4o_mini": "OpenAI-GPT", "gpt41_mini": "OpenAI-GPT", "gpt41_nano": "OpenAI-GPT",
          "gpt35_turbo": "OpenAI-GPT", "o3_mini": "OpenAI-o-series", "gemini_20_flash": "Google-Gemini",
          "gemini_25_flash": "Google-Gemini", "deepseek_v3": "DeepSeek", "qwen25_3b": "Qwen2.5", "qwen25_7b": "Qwen2.5",
          "qwen25_14b": "Qwen2.5", "hermes3_llama8b": "Hermes3-Llama3.1"}
NAMES = {"gpt4o": "GPT-4o", "gpt4o_mini": "GPT-4o-mini", "gpt41_mini": "GPT-4.1-mini", "gpt41_nano": "GPT-4.1-nano",
         "gpt35_turbo": "GPT-3.5-turbo", "o3_mini": "o3-mini", "gemini_20_flash": "Gemini-2.0-Flash",
         "gemini_25_flash": "Gemini-2.5-Flash", "deepseek_v3": "DeepSeek-V3", "qwen25_3b": "Qwen2.5-3B",
         "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "hermes3_llama8b": "Hermes-3-Llama-3.1-8B"}
rows = []
for sid, x in P.items():
    t = T[sid]; mtc = int(t["made_tool_call"])
    j1 = 1 if t["llm_label"] == "FABRICATED" else 0
    j2 = 1 if (not mtc and J2.get(sid, {}).get("label") == "FABRICATED") else 0
    rows.append({"sid": sid, "model": x["model"], "domain": x["domain"], "mtc": mtc, "rule": int(t["rule_flag"]),
                 "j1": j1 if not mtc else 0, "j2": j2, "cat1": t["llm_category"], "cat2": J2.get(sid, {}).get("category", "")})
for r in rows:
    r["cons"] = int(r["j1"] and r["j2"]); r["union"] = int(r["j1"] or r["j2"])
    # guidance-conformant adjustment: pure bracketed placeholders are NOT fabricated per GUIDANCE.md
    r["placeholder_only"] = int(bool(PLACEHOLDER_ONLY.match(P[r["sid"]]["final_answer"].strip())))
    r["j1adj"] = int(r["j1"] and not r["placeholder_only"])
models = sorted(set(r["model"] for r in rows), key=lambda m: (m in OPEN, NAMES[m]))
out = {"notes": {}, "per_model": {}, "classifier_validity": {}, "prop_vs_open": {}, "explanations": {}}

def rate(xs, k):
    n = len(xs); s = sum(r[k] for r in xs); lo, hi = wilson(s, n)
    return {"k": s, "n": n, "rate": s / n if n else None, "ci95": [lo, hi]}

for m in models:
    xs = [r for r in rows if r["model"] == m]
    nt = [r for r in xs if not r["mtc"]]
    out["per_model"][m] = {"name": NAMES[m], "kind": "open-weight" if m in OPEN else "proprietary", "family": FAMILY[m],
        "n": len(xs), "tool_call": {"k": len(xs) - len(nt), "rate": (len(xs) - len(nt)) / len(xs), "ci95": list(wilson(len(xs) - len(nt), len(xs)))},
        "published_pct": pub.get(m), "rule": rate(xs, "rule"), "judge1_gpt41mini": rate(xs, "j1"), "judge2_gpt4o": rate(xs, "j2"),
        "consensus_both_judges": rate(xs, "cons"), "judge1_adjusted_no_placeholder": rate(xs, "j1adj"), "union_either_judge": rate(xs, "union"),
        "conditional_on_no_tool_judge1": rate(nt, "j1") if nt else None,
        "judge1_categories_no_tool": dict(collections.Counter(r["cat1"] for r in nt))}

# ---- classifier validity (rule vs LLM references) on the full population ----
def pr(pred, ref, xs):
    tp = sum(1 for r in xs if r[pred] and r[ref]); fp = sum(1 for r in xs if r[pred] and not r[ref])
    fn = sum(1 for r in xs if not r[pred] and r[ref])
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None, "precision_ci95": list(wilson(tp, tp + fp)),
            "recall": tp / (tp + fn) if tp + fn else None, "recall_ci95": list(wilson(tp, tp + fn))}
nt_all = [r for r in rows if not r["mtc"]]
agree = [r for r in nt_all if r["j1"] == r["j2"]]
out["classifier_validity"]["population"] = {
    "n_traces": len(rows), "n_no_tool": len(nt_all),
    "judge1_vs_judge2_kappa_no_tool": cohen_kappa([r["j1"] for r in nt_all], [r["j2"] for r in nt_all]),
    "judge1_vs_judge2_agreement": sum(1 for r in nt_all if r["j1"] == r["j2"]) / len(nt_all),
    "rule_vs_judge1": pr("rule", "j1", rows), "rule_vs_judge2": pr("rule", "j2", rows),
    "rule_vs_judge1_adjusted": pr("rule", "j1adj", rows), "judge1_fab_on_pure_placeholders": sum(r["j1"] and r["placeholder_only"] for r in rows),
    "rule_vs_consensus_on_judge_agreed_traces": pr("rule", "j1", [r for r in rows if r["mtc"] or r["j1"] == r["j2"]]),
    "n_judge_disagreements": len(nt_all) - len(agree),
    "rule_kappa_vs_judge1_no_tool": cohen_kappa([r["rule"] for r in nt_all], [r["j1"] for r in nt_all])}
# per-kind recall (open vs proprietary)
for kind, sel in (("proprietary", lambda r: r["model"] not in OPEN), ("open-weight", lambda r: r["model"] in OPEN)):
    out["classifier_validity"]["population"][f"rule_vs_judge1_{kind}"] = pr("rule", "j1", [r for r in rows if sel(r)])
# ---- on the 150-item stratified labeling sample (weighted), as the human-label analysis will be run ----
K = list(csv.DictReader(open(os.path.join(D, "labeling", "fabrication_labeling_KEY.csv"))))
def wpr(K, ref):
    tp = fp = fn = 0.0; tpu = fpu = fnu = 0
    for k in K:
        w = float(k["weight"]); pred = int(k["rule_flag"]); y = ref(k)
        tp += w * (pred and y); fp += w * (pred and not y); fn += w * ((not pred) and y)
        tpu += (pred and y); fpu += (pred and not y); fnu += ((not pred) and y)
    return {"weighted_precision": tp / (tp + fp) if tp + fp else None, "weighted_recall": tp / (tp + fn) if tp + fn else None,
            "unweighted_counts": {"tp": tpu, "fp": fpu, "fn": fnu}}
idx = {r["sid"]: r for r in rows}
out["classifier_validity"]["sample150"] = {
    "n": len(K), "strata": dict(collections.Counter(k["stratum"] for k in K)),
    "models": dict(collections.Counter(k["model"] for k in K)),
    "vs_judge1": wpr(K, lambda k: idx[k["session_id"]]["j1"]), "vs_judge2": wpr(K, lambda k: idx[k["session_id"]]["j2"]),
    "precision_flagged_stratum_vs_judge1": {"k": sum(1 for k in K if k["stratum"] == "A_flagged" and idx[k["session_id"]]["j1"]),
                                             "n": sum(1 for k in K if k["stratum"] == "A_flagged")}}
f = out["classifier_validity"]["sample150"]["precision_flagged_stratum_vs_judge1"]; f["ci95"] = list(wilson(f["k"], f["n"]))

# ---- proprietary vs open-weight ----
def perm_test(vals_a, vals_b):
    allv = vals_a + vals_b; na = len(vals_a); obs = sum(vals_b) / len(vals_b) - sum(vals_a) / na
    cnt = tot = 0
    for comb in itertools.combinations(range(len(allv)), len(vals_b)):
        b = [allv[i] for i in comb]; a = [allv[i] for i in range(len(allv)) if i not in comb]
        d = sum(b) / len(b) - sum(a) / len(a); tot += 1; cnt += abs(d) >= abs(obs) - 1e-12
    return obs, cnt / tot
for key in ("rule", "j1", "j1adj", "j2", "cons"):
    pv = [r for r in rows if r["model"] not in OPEN]; ov = [r for r in rows if r["model"] in OPEN]
    a, b = sum(r[key] for r in pv), sum(r[key] for r in ov)
    ma = [sum(r[key] for r in rows if r["model"] == m) / sum(1 for r in rows if r["model"] == m) for m in models if m not in OPEN]
    mb = [sum(r[key] for r in rows if r["model"] == m) / sum(1 for r in rows if r["model"] == m) for m in models if m in OPEN]
    d, p = perm_test(ma, mb)
    out["prop_vs_open"][key] = {"model_rates_prop": ma, "model_rates_open": mb,"trace_pooled_prop": [a, len(pv), a / len(pv), list(wilson(a, len(pv)))],
        "trace_pooled_open": [b, len(ov), b / len(ov), list(wilson(b, len(ov)))],
        "trace_level_fisher_p_ignores_clustering": fisher_exact_2x2(b, len(ov) - b, a, len(pv) - a),
        "model_level_mean_prop": sum(ma) / len(ma), "model_level_mean_open": sum(mb) / len(mb),
        "model_level_diff_open_minus_prop": d, "model_level_exact_permutation_p": p, "n_models": [len(ma), len(mb)]}
# published numbers (model-level only)
ma = [pub[m] / 100 for m in models if m not in OPEN]; mb = [pub[m] / 100 for m in models if m in OPEN]
d, p = perm_test(ma, mb)
out["prop_vs_open"]["published"] = {"model_level_mean_prop": sum(ma) / len(ma), "model_level_mean_open": sum(mb) / len(mb),
                                    "model_level_diff_open_minus_prop": d, "model_level_exact_permutation_p": p}

# ---- explanations ----
tc = [1 - out["per_model"][m]["tool_call"]["rate"] for m in models]
for key in ("rule", "judge1_gpt41mini", "judge1_adjusted_no_placeholder"):
    fr = [out["per_model"][m][key]["rate"] for m in models]
    out["explanations"][f"spearman_no_tool_rate_vs_{key}"] = spearman(tc, fr)
out["explanations"]["no_tool_traces_judge1_fabricated_share"] = {"k": sum(r["j1"] for r in nt_all), "n": len(nt_all)}
dom = {}
for dmn in sorted(set(r["domain"] for r in rows)):
    xs = [r for r in rows if r["domain"] == dmn]
    dom[dmn] = {"no_tool": rate([dict(r, nt=1 - r["mtc"]) for r in xs], "nt"), "judge1": rate(xs, "j1"),
                "judge1_given_no_tool": rate([r for r in xs if not r["mtc"]], "j1")}
out["explanations"]["by_domain"] = dom
fam = {}
for fm in sorted(set(FAMILY.values())):
    xs = [r for r in rows if FAMILY[r["model"]] == fm]
    fam[fm] = {"models": sorted(set(r["model"] for r in xs)), "judge1": rate(xs, "j1"), "rule": rate(xs, "rule"),
               "no_tool": rate([dict(r, nt=1 - r["mtc"]) for r in xs], "nt")}
out["explanations"]["by_family"] = fam
dbn = [DB[r["sid"]] for r in nt_all if r["sid"] in DB]
out["explanations"]["db_mechanism_proprietary"] = {
    "n_no_tool_traces_in_db": len(dbn), "csv_db_agreement_on_tool_executed": "see db_structural_features.csv",
    "final_completion_contains_Action_line": sum(int(x["final_turn_has_action_text"]) for x in dbn),
    "of_which_names_registered_tool": sum(int(x["inline_action_known_tool"]) for x in dbn),
    "with_multiple_Action_lines": sum(int(x["inline_n_actions"]) > 1 for x in dbn),
    "with_fake_Observation_text": sum(int(x["final_turn_has_observation_text"]) for x in dbn),
    "zero_observations_returned": sum(1 for x in dbn if x["db_n_observations"] == "0")}
allDB = list(csv.DictReader(open(os.path.join(D, "db_structural_features.csv"))))
out["explanations"]["db_mechanism_proprietary"]["csv_vs_db_tool_executed_agree"] = sum(
    1 for x in allDB if bool(x["csv_tool_selected"]) == (int(x["db_n_actions"]) > 0))
out["explanations"]["db_mechanism_proprietary"]["n_db_matched"] = len(allDB)
# fabricated among inline-action traces (judge1)
inl = [r for r in nt_all if r["sid"] in DB and DB[r["sid"]]["final_turn_has_action_text"] == "1"]
out["explanations"]["db_mechanism_proprietary"]["judge1_fabricated_among_inline_action"] = rate(inl, "j1")
nat = {}
for m in ("qwen25_7b", "qwen25_14b"):
    p = os.path.join(R, f"native_toolcall_{m}.json")
    if os.path.exists(p): nat[m] = {"native": json.load(open(p))["native_tool_call_rate_pct"], "prompted": 100 * out["per_model"][m]["tool_call"]["rate"]}
out["explanations"]["native_function_calling_probe"] = nat
json.dump(out, open(os.path.join(D, "fabrication_results.json"), "w"), indent=2)

# ---- print summary ----
print(f"{'model':24s} TC%   pub   rule[CI]            J1[CI]              J2   cons  J1adj  J1|noTool")
for m in models:
    v = out["per_model"][m]
    f = lambda k: f"{100*v[k]['rate']:5.1f} [{100*v[k]['ci95'][0]:4.1f},{100*v[k]['ci95'][1]:4.1f}]"
    c = v["conditional_on_no_tool_judge1"]
    print(f"{v['name']:24s} {100*v['tool_call']['rate']:5.1f} {v['published_pct']:5.1f} {f('rule')} {f('judge1_gpt41mini')} {100*v['judge2_gpt4o']['rate']:5.1f} {100*v['consensus_both_judges']['rate']:5.1f} {100*v['judge1_adjusted_no_placeholder']['rate']:5.1f} "
          + (f"{100*c['rate']:.1f} ({c['k']}/{c['n']})" if c else "-"))
cv = out["classifier_validity"]["population"]
for k in ("rule_vs_judge1", "rule_vs_judge2", "rule_vs_judge1_adjusted", "rule_vs_judge1_proprietary", "rule_vs_judge1_open-weight"):
    v = cv[k]; print(k, {kk: (round(vv, 3) if isinstance(vv, float) else ([round(z, 3) for z in vv] if isinstance(vv, list) else vv)) for kk, vv in v.items()})
for k, v in out["prop_vs_open"].items():
    print("PvO", k, {kk: vv for kk, vv in v.items() if not kk.startswith("model_rates")})
e = out["explanations"]
print({k: v for k, v in e.items() if k.startswith("spearman")}, e["db_mechanism_proprietary"], e["native_function_calling_probe"])
