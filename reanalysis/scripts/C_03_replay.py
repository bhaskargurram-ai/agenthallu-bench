"""C_03: Offline replay of the Interceptor's three layers on full GPT-4o-mini traces (C_02 output).

Layer definitions replayed EXACTLY as in experiments/run_interceptor.py (the code that produced
interceptor_heldout_600.csv), plus the variants described in the paper's Algorithm 1:
  L1 hit  : a tool step whose validation_errors is non-empty OR whose result dict contains key "error".
  L2 hits : number of distinct keywords (substring match, lower-cased) in one LLM message;
            CODE set  = {error, invalid, unknown, failed, not found, missing, incorrect, wrong}
            PAPER set = {error, invalid, unknown, missing, incorrect, wrong, cannot}
  L3 flag : answer contains a number (len>1 or >9) absent from all observations; never fires when there are
            no observations (no tool executed).
Policies:
  ORIG (as actually run): abstain if #L1 >= 2, else if L3; L2 only logged (never abstains).
  PAPER-COVERAGE: abstain if #L1 >= tau1 or L2(any message, >=tau2 hits) or L3.
  PAPER-PRECISION: abstain if #L1 >= tau1.
Labels: heuristic (repo's substring rule) and GPT-4o-mini judge (C_04 cache).
"""
import json, os, re, sys, glob, random, math, time, statistics, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import OUT, mcnemar_exact, hypergeom_sf, wilson, boot_ci
from C_04_judge import key as jkey, load_cache

CODE_KW = ["error", "invalid", "unknown", "failed", "not found", "missing", "incorrect", "wrong"]
PAPER_KW = ["error", "invalid", "unknown", "missing", "incorrect", "wrong", "cannot"]
JC = load_cache()


def l3_flag(final, obs):
    if not final or not obs:
        return False
    a = set(re.findall(r"-?\d+\.?\d*", final))
    o = set()
    for x in obs:
        o.update(re.findall(r"-?\d+\.?\d*", str(x)))
    return bool({n for n in a - o if len(n) > 1 or int(float(n)) > 9})


def l3_flag_fixed(final, obs):
    """Variant: strip list enumerators / trailing dots and ignore dates' pieces present in query? (diagnostic)"""
    if not final or not obs:
        return False
    a = set(re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"(?m)^\s*\d+\.\s", "", final)))
    o = set()
    for x in obs:
        o.update(re.findall(r"-?\d+(?:\.\d+)?", str(x)))
    return bool({n for n in a - o if len(n) > 1 or int(float(n)) > 9})


def features(d):
    steps = d["steps"]
    tool_steps = [s for s in steps if "tool_name" in s]
    l1 = [bool(s["validation_errors"]) or (isinstance(s["tool_result"], dict) and "error" in s["tool_result"]) for s in tool_steps]
    l1_schema = [bool(s["validation_errors"]) for s in tool_steps]
    obs = [s["obs"] for s in tool_steps]
    texts = [s["llm_text"] for s in steps]
    l2c = max([sum(1 for k in CODE_KW if k in t.lower()) for t in texts] or [0])
    l2p = max([sum(1 for k in PAPER_KW if k in t.lower()) for t in texts] or [0])
    final_msg = steps[-1]["llm_text"] if steps and steps[-1].get("final") else ""
    inline_fab = bool(final_msg) and bool(re.search(r"Action:\s*\w+", final_msg.split("Final Answer:")[0]))
    n_tool = len(tool_steps)
    needs_tool = d["domain"] != "knowledge"
    inj = d.get("injection")
    inj_schema_valid = None
    if inj:
        first = tool_steps[0] if tool_steps else None
        inj_schema_valid = bool(first) and not first["validation_errors"]
    # argument-grounding check (diagnostic L4): every string arg must appear in query or earlier observations
    ctx = d["query"].lower()
    ungrounded = False
    for s in tool_steps:
        for v in (s.get("exec_params") or {}).values():
            if isinstance(v, str) and len(v) >= 3 and v.lower() not in ctx:
                ungrounded = True
        ctx += " " + s["obs"].lower()
    hc = d["heuristic_correct"]
    jc = JC.get(jkey(d["query"], d["gt"], d["final_answer"]))
    return {"task_id": d["task_id"], "domain": d["domain"], "source": d["task_id"].rsplit("_", 1)[0],
            "n_tool": n_tool, "l1": sum(l1), "l1_schema": sum(l1_schema), "l2_code": l2c, "l2_paper": l2p,
            "l3": l3_flag(d["final_answer"], obs), "l3_fixed": l3_flag_fixed(d["final_answer"], obs),
            "no_tool": n_tool == 0, "inline_fab": inline_fab, "needs_tool": needs_tool,
            "maxsteps": d["final_answer"] == "Max steps reached.",
            "injected": bool(inj), "inj_schema_valid": inj_schema_valid, "ungrounded_arg": ungrounded,
            "h_correct": hc, "j_correct": jc, "final": d["final_answer"], "cost": d["cost"],
            "llm_ms": d["llm_ms"], "n_llm": len(d["llm_ms"])}


POLICIES = {
    "ORIG (L1>=2 | L3; as run)": lambda f: f["l1"] >= 2 or f["l3"],
    "L1>=1": lambda f: f["l1"] >= 1,
    "L1>=2": lambda f: f["l1"] >= 2,
    "L2code>=2": lambda f: f["l2_code"] >= 2,
    "L2paper>=1": lambda f: f["l2_paper"] >= 1,
    "L2paper>=2": lambda f: f["l2_paper"] >= 2,
    "L3": lambda f: f["l3"],
    "L3 (enumerator-fixed)": lambda f: f["l3_fixed"],
    "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)": lambda f: f["l1"] >= 1 or f["l2_paper"] >= 1 or f["l3"],
    "TUNED (L1>=1 | L2code>=2 | L3)": lambda f: f["l1"] >= 1 or f["l2_code"] >= 2 or f["l3"],
    "PAPER-PRECISION (L1>=1)": lambda f: f["l1"] >= 1,
    "No tool call": lambda f: f["no_tool"],
    "No tool call | inline-fabricated": lambda f: f["no_tool"] or f["inline_fab"],
    "ORIG | no tool call": lambda f: f["l1"] >= 2 or f["l3"] or f["no_tool"],
    "Arg-grounding (diag. L4)": lambda f: f["ungrounded_arg"],
}


def evaluate(F, label, pol, rng_seed=42):
    n = len(F)
    wrong = [not f[label] for f in F]
    ab = [pol(f) for f in F]
    k = sum(ab); H = sum(wrong)
    sv = sum(1 for i in range(n) if ab[i] and wrong[i]); bk = k - sv
    ans = [i for i in range(n) if not ab[i]]
    res = {"n": n, "abstain": k, "abstain_rate": k / n, "coverage": 1 - k / n,
           "halluc_before": H / n, "halluc_after": (H - sv) / n, "delta_pp": -100 * sv / n,
           "saved": sv, "broken": bk,
           "precision": sv / k if k else None, "precision_wilson": wilson(sv, k) if k else None,
           "recall": sv / H if H else None,
           "selective_risk": sum(wrong[i] for i in ans) / len(ans) if ans else None,
           "selective_accuracy": 1 - sum(wrong[i] for i in ans) / len(ans) if ans else None,
           "random_matched_expected_halluc": (H - k * H / n) / n,
           "random_matched_expected_saved": k * H / n,
           "p_better_than_random(hypergeom one-sided)": hypergeom_sf(sv, n, H, k) if k else None}
    # random-abstention simulation CI for halluc_after at matched k
    rng = random.Random(rng_seed); sims = []
    for _ in range(4000):
        S = rng.sample(range(n), k)
        sims.append(sum(wrong[i] for i in S))
    sims.sort()
    res["random_matched_saved_95"] = [sims[100], sims[3899]]
    # bootstrap CI on the policy's improvement over random-at-matched-rate in selective risk
    return res


def rc_curve(F, label, score):
    """Risk-coverage: abstain on highest-score first (ties broken randomly, averaged)."""
    n = len(F); wrong = [not f[label] for f in F]
    rng = random.Random(0)
    curves = []
    for rep in range(50):
        order = sorted(range(n), key=lambda i: (-score(F[i]), rng.random()))
        cov, risk = [], []
        for k in range(0, n):
            kept = order[k:]
            cov.append(len(kept) / n); risk.append(sum(wrong[i] for i in kept) / len(kept))
        curves.append(risk)
    risk = [statistics.mean(c[j] for c in curves) for j in range(n)]
    return [1 - k / n for k in range(n)], risk


def main():
    files = sorted(glob.glob(f"{OUT}/traces/*.jsonl"))
    data = {}
    for p in files:
        name = os.path.basename(p)[:-6]
        data[name] = [features(json.loads(l)) for l in open(p)]
    # 'scoreable' subset: drop halueval (empty reference answer -> heuristic always 'wrong') and
    # toolbench_syn (reference = "Completed: <query>", i.e. no ground-truth fact)
    for name in list(data):
        data[name + "__scoreable"] = [f for f in data[name] if f["source"] not in ("halueval", "toolbench_syn")]
    out = {"runs": {}, "notes": {"scoreable": "excludes halueval (empty GT) and toolbench_syn (GT echoes query)"}}
    for name, F in data.items():
        R = {"n": len(F)}
        R["judge_coverage"] = sum(f["j_correct"] is not None for f in F)
        for label in ["h_correct", "j_correct"]:
            if any(f[label] is None for f in F):
                continue
            R[label] = {p: evaluate(F, label, fn) for p, fn in POLICIES.items()}
        # failure taxonomy (judge label if available else heuristic)
        lab = "j_correct" if all(f["j_correct"] is not None for f in F) else "h_correct"
        tax = collections.Counter(); catch = collections.defaultdict(lambda: collections.Counter())
        for f in F:
            wrong = not f[lab]
            if f["l1"] >= 1: t = "schema/tool-error"
            elif f["injected"] and f["inj_schema_valid"]: t = "propagation (schema-valid wrong arg)"
            elif f["no_tool"] and f["needs_tool"]: t = "fabrication: no tool executed"
            elif f["inline_fab"]: t = "fabrication: inline (tool executed, answer has unexecuted Action)"
            elif f["no_tool"]: t = "knowledge QA (no tool needed)"
            else: t = "tools ok"
            key = (t, "wrong" if wrong else "correct")
            tax[key] += 1
            for p in ["ORIG (L1>=2 | L3; as run)", "L1>=1", "L2paper>=1", "L3", "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)", "Arg-grounding (diag. L4)"]:
                if POLICIES[p](f): catch[p][key] += 1
        R["taxonomy_label"] = lab
        R["taxonomy"] = {f"{a} | {b}": v for (a, b), v in sorted(tax.items())}
        R["caught_by_policy"] = {p: {f"{a} | {b}": v for (a, b), v in sorted(c.items())} for p, c in catch.items()}
        R["label_agreement_heur_vs_judge"] = None
        if all(f["j_correct"] is not None for f in F):
            a = sum(1 for f in F if f["h_correct"] == f["j_correct"])
            R["label_agreement_heur_vs_judge"] = a / len(F)
        R["no_tool_rate"] = sum(f["no_tool"] for f in F) / len(F)
        R["inline_fab_rate"] = sum(f["inline_fab"] for f in F) / len(F)
        R["l1_any_rate"] = sum(f["l1"] >= 1 for f in F) / len(F)
        R["l1_schema_any_rate"] = sum(f["l1_schema"] >= 1 for f in F) / len(F)
        if any(f["injected"] for f in F):
            inj = [f for f in F if f["injected"]]
            R["injection"] = {"applied": len(inj), "schema_valid": sum(1 for f in inj if f["inj_schema_valid"]),
                              "not_applied(no tool call)": len(F) - len(inj)}
        lat = [ms for f in F for ms in f["llm_ms"]]
        R["agent_latency"] = {"llm_calls": len(lat), "median_call_ms": statistics.median(lat) if lat else None,
                              "median_trace_ms": statistics.median([sum(f["llm_ms"]) for f in F]),
                              "mean_trace_ms": statistics.mean([sum(f["llm_ms"]) for f in F]),
                              "median_llm_calls_per_trace": statistics.median([f["n_llm"] for f in F]),
                              "mean_cost_usd": statistics.mean([f["cost"] for f in F])}
        out["runs"][name] = R
    # run-to-run reproducibility (clean r1 vs r2)
    if "clean_r1" in data and "clean_r2" in data:
        A = {f["task_id"]: f for f in data["clean_r1"]}; B = {f["task_id"]: f for f in data["clean_r2"]}
        ids = sorted(set(A) & set(B))
        rr = {}
        for lab in ["h_correct", "j_correct"]:
            if all(A[i][lab] is not None and B[i][lab] is not None for i in ids):
                rr[lab] = {"n": len(ids), "agree": sum(A[i][lab] == B[i][lab] for i in ids) / len(ids),
                           "flip_wrong_to_right": sum((not A[i][lab]) and B[i][lab] for i in ids),
                           "flip_right_to_wrong": sum(A[i][lab] and (not B[i][lab]) for i in ids)}
        rr["identical_final_answer"] = sum(A[i]["final"] == B[i]["final"] for i in ids) / len(ids)
        rr["orig_policy_decision_agree"] = sum(POLICIES["ORIG (L1>=2 | L3; as run)"](A[i]) == POLICIES["ORIG (L1>=2 | L3; as run)"](B[i]) for i in ids) / len(ids)
        out["run_to_run"] = rr
    # L2 keyword-level precision/recall on clean_r1 (judge label)
    if "clean_r1" in data:
        F = data["clean_r1"]; raw = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{OUT}/traces/clean_r1.jsonl")}
        lab = "j_correct" if all(f["j_correct"] is not None for f in F) else "h_correct"
        H = sum(not f[lab] for f in F)
        kwres = {}
        for kw in sorted(set(CODE_KW + PAPER_KW)):
            hit = [f for f in F if any(kw in s["llm_text"].lower() for s in raw[f["task_id"]]["steps"])]
            tp = sum(not f[lab] for f in hit)
            kwres[kw] = {"fires": len(hit), "precision": tp / len(hit) if hit else None, "recall": tp / H if H else None}
        out["L2_keyword_pr_clean_r1"] = {"label": lab, "base_rate": H / len(F), "per_keyword": kwres}
        # examples of L2/L3 false positives for qualitative analysis
        ex = []
        for f in F:
            if f[lab] and (f["l3"] or f["l2_paper"] >= 1) and len(ex) < 12:
                ex.append({"task_id": f["task_id"], "l2_paper": f["l2_paper"], "l3": f["l3"], "final": f["final"][:220]})
        out["false_positive_examples_clean_r1"] = ex
    json.dump(out, open(f"{OUT}/replay_results.json", "w"), indent=1, default=str)
    # risk-coverage data for figure
    rc = {}
    for name in ["clean_r1__scoreable", "clean_schema__scoreable", "p2sem_r1__scoreable", "p2sem_schema__scoreable", "retail_r1", "retail_schema"]:
        if name not in data: continue
        F = data[name]
        lab = "j_correct" if all(f["j_correct"] is not None for f in F) else "h_correct"
        score = lambda f: 4 * min(f["l1"], 3) + 2 * f["l3"] + 0.5 * min(f["l2_paper"], 3)
        cov, risk = rc_curve(F, lab, score)
        nt_cov, nt_risk = rc_curve(F, lab, lambda f: 1.0 * f["no_tool"] + 0.01 * f["inline_fab"])
        pts = {p: (evaluate(F, lab, fn)["coverage"], evaluate(F, lab, fn)["selective_risk"]) for p, fn in POLICIES.items()}
        wrong = [not f[lab] for f in F]
        rc[name] = {"label": lab, "cov": cov, "risk_interceptor_score": risk, "risk_notool_score": nt_risk,
                    "base_risk": sum(wrong) / len(F), "points": pts,
                    "oracle_risk": [max(0.0, (sum(wrong) - (len(F) - int(round(c * len(F))))) / max(1, int(round(c * len(F))))) for c in cov]}
    json.dump(rc, open(f"{OUT}/risk_coverage_data.json", "w"))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("h_correct", "j_correct", "caught_by_policy")} for k, v in out["runs"].items()}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
