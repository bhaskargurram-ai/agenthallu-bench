"""V_01: re-score EVERY available AgentProp trace with the structured validator (default config V1.0).

Outputs (V_validator/):
  labels_all.csv     one row per trace
  results.json       determinism check, gold audit, per-model accuracy, P2/interceptor breakdowns, calibration
"""
import collections
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_core as V  # noqa: E402
import V_traces as L  # noqa: E402

OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/V_validator"
EXCLUDE_FROM_ACC = {"not_verifiable", "not_verifiable_answer_only", "infra_error"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def score_all(recs, tasks, cfg=None, with_answer_only=True):
    specs = {}
    rows = []
    for r in recs:
        t = tasks[r["task_id"]]
        if r["task_id"] not in specs:
            specs[r["task_id"]] = V.build_spec(t, dict(V.DEFAULT_CFG, **(cfg or {})))
        sp = specs[r["task_id"]]
        ao = r["mode"] == "answer_only"
        s = V.score_trace(t, r["calls"], r["final_answer"], cfg=cfg, spec=sp, answer_only=ao, n_exec_hint=r["n_exec_hint"])
        if with_answer_only and not ao:
            s2 = V.score_trace(t, r["calls"], r["final_answer"], cfg=cfg, spec=sp, answer_only=True,
                               n_exec_hint=sum(c["executed"] for c in r["calls"]))
            s["A_answer_only"] = s2.get("A")
            s["outcome_answer_only"] = s2.get("outcome")
        rows.append((r, s))
    return rows, specs


def acc(rows, flt):
    sel = [s for r, s in rows if flt(r, s) and s["outcome"] not in EXCLUDE_FROM_ACC]
    k = sum(1 for s in sel if s.get("CORRECT"))
    return {"n": len(sel), "correct": k, "acc": round(k / len(sel), 4) if sel else None, "ci95": wilson(k, len(sel))}


def main():
    recs, stats = L.load_all()
    tasks = L.load_tasks()
    rows, specs = score_all(recs, tasks)
    res = {"validator_version": V.VALIDATOR_VERSION, "config": V.DEFAULT_CFG}

    # ---------------- determinism / replay
    ex = stats.pop("mismatch_examples", [])
    res["replay_determinism"] = {"per_tool": {k: v for k, v in stats.items() if k.startswith("replay_")},
                                 "n_match": sum(v for k, v in stats.items() if k.endswith("_match")),
                                 "n_mismatch": sum(v for k, v in stats.items() if k.endswith("_mismatch")),
                                 "reconciled_unlogged_injection": stats.get("reconciled_unlogged_injection", 0),
                                 "injection_original_mismatch": stats.get("injection_original_mismatch", 0),
                                 "mismatch_examples": ex[:5]}

    # ---------------- gold audit (task-file gold vs query)
    all_tasks_used = sorted({r["task_id"] for r in recs})
    audit = collections.Counter()
    n_ver, n_bad = 0, 0
    tmpl_bad = collections.Counter()
    examples = []
    for tid in all_tasks_used:
        sp = specs.get(tid) or V.build_spec(tasks[tid])
        if not sp.verifiable:
            continue
        n_ver += 1
        mm = V.audit_taskfile_vs_query(sp)
        if mm:
            n_bad += 1
            tmpl_bad[sp.template] += 1
            for m in mm:
                audit[m[0]] += 1
            if len(examples) < 12:
                examples.append({"task_id": tid, "query": tasks[tid]["query"], "mismatches": [list(map(str, m)) for m in mm]})
    tmpl_n = collections.Counter(specs[t].template for t in all_tasks_used if t in specs)
    res["gold_audit"] = {"n_tasks_in_traces": len(all_tasks_used), "n_verifiable_tasks": n_ver,
                         "n_tasks_taskfile_contradicts_query": n_bad,
                         "by_slot": dict(audit.most_common()), "by_template": {k: [v, tmpl_n[k]] for k, v in tmpl_bad.items()},
                         "examples": examples}
    # verifiability at task and trace level
    nv = collections.Counter()
    for tid in all_tasks_used:
        sp = specs[tid]
        if not sp.verifiable:
            nv[sp.template] += 1
    res["not_verifiable_tasks"] = {"by_template": dict(nv), "reasons": {specs[t].template: specs[t].reason
                                                                      for t in all_tasks_used if not specs[t].verifiable}}
    res["template_counts_tasks"] = dict(tmpl_n)

    # ---------------- outcome distribution per source
    res["outcomes_by_source"] = {}
    for (src, oc), n in sorted(collections.Counter((r["source"], s["outcome"]) for r, s in rows).items()):
        res["outcomes_by_source"].setdefault(src, {})[oc] = n

    # ---------------- per-model baseline accuracy (full traces)
    old = {}
    with open(f"{L.RESULTS}/baseline_all_rejudged_v2.csv") as fh:
        for x in csv.DictReader(fh):
            old[x["session_id"]] = x
    base = {}
    for model in sorted({r["model"] for r, s in rows if r["source"] == "baseline"}):
        f = lambda r, s, m=model: r["source"] == "baseline" and r["model"] == m
        sel = [(r, s) for r, s in rows if f(r, s)]
        d = acc(rows, f)
        d["n_sessions"] = len(sel)
        d["n_not_verifiable"] = sum(1 for r, s in sel if s["outcome"] == "not_verifiable")
        d["n_infra_error"] = sum(1 for r, s in sel if s["outcome"] == "infra_error")
        d["outcomes"] = dict(collections.Counter(s["outcome"] for r, s in sel))
        ver = [(r, s) for r, s in sel if s["outcome"] not in EXCLUDE_FROM_ACC]
        for comp in ("T", "P", "A"):
            d[f"{comp}_rate"] = round(sum(1 for r, s in ver if s.get(comp)) / len(ver), 4) if ver else None
        wr = [(r, s) for r, s in ver if s.get("S") is not None]
        d["S_rate_write_tasks"] = round(sum(1 for r, s in wr if s["S"]) / len(wr), 4) if wr else None
        # old metrics on the same sessions
        o_all = [old[r["session_id"]] for r, s in sel if r["session_id"] in old]
        o_ver = [old[r["session_id"]] for r, s in ver if r["session_id"] in old]
        tr = lambda v: str(v).strip().lower() in ("true", "1", "yes")
        if o_all:
            d["old_heuristic_all_sessions"] = round(sum(tr(x["heuristic_correct"]) for x in o_all) / len(o_all), 4)
            d["old_ensemble_all_sessions"] = round(sum(tr(x["ensemble_correct"]) for x in o_all) / len(o_all), 4)
            d["old_n_all_sessions"] = len(o_all)
        if o_ver:
            d["old_heuristic_same_verifiable"] = round(sum(tr(x["heuristic_correct"]) for x in o_ver) / len(o_ver), 4)
            d["old_ensemble_same_verifiable"] = round(sum(tr(x["ensemble_correct"]) for x in o_ver) / len(o_ver), 4)
            d["old_n_same_verifiable"] = len(o_ver)
        base[model] = d
    res["baseline_accuracy_by_model"] = base
    res["baseline_retail_accuracy_by_model"] = {
        m: acc(rows, lambda r, s, m=m: r["source"] == "baseline_retail" and r["model"] == m)
        for m in sorted({r["model"] for r, s in rows if r["source"] == "baseline_retail"})}
    # per-slot P on baseline (matched calls)
    slot = collections.defaultdict(lambda: [0, 0, 0])
    for r, s in rows:
        if r["source"] != "baseline" or s["outcome"] in EXCLUDE_FROM_ACC:
            continue
        for k, v in (s.get("per_slot") or {}).items():
            slot[k][2] += 1
            if v is True:
                slot[k][0] += 1
            elif v is False:
                slot[k][1] += 1
    res["baseline_per_slot"] = {k: {"correct": v[0], "wrong": v[1], "gold_slot_instances": v[2],
                                    "acc_given_called": round(v[0] / (v[0] + v[1]), 4) if v[0] + v[1] else None}
                                for k, v in sorted(slot.items())}

    # ---------------- answer-only calibration on full traces & open-weight answer-only accuracy
    cal = collections.Counter()
    for r, s in rows:
        if r["mode"] == "full" and r["source"] == "baseline" and s["outcome"] not in EXCLUDE_FROM_ACC \
                and s.get("A_answer_only") is not None:
            cal[(bool(s["A_answer_only"]), bool(s["CORRECT"]))] += 1
    n = sum(cal.values())
    res["answer_only_calibration_on_proprietary_baseline"] = {
        "n": n, "AO_yes_CORRECT_yes": cal[(True, True)], "AO_yes_CORRECT_no": cal[(True, False)],
        "AO_no_CORRECT_yes": cal[(False, True)], "AO_no_CORRECT_no": cal[(False, False)],
        "AO_rate": round((cal[(True, True)] + cal[(True, False)]) / n, 4) if n else None,
        "CORRECT_rate": round((cal[(True, True)] + cal[(False, True)]) / n, 4) if n else None,
        "note": "answer-only mode = A computed against deterministic gold observations without the trace; "
                "only tasks without write/store-dependent tools are scored in this mode"}
    ao = {}
    for (src, model) in sorted({(r["source"], r["model"]) for r, s in rows if r["mode"] == "answer_only"}):
        sel = [s for r, s in rows if r["source"] == src and r["model"] == model]
        sc = [s for s in sel if s["outcome"] not in EXCLUDE_FROM_ACC]
        k = sum(1 for s in sc if s.get("A"))
        ao[f"{src}|{model}"] = {"n_sessions": len(sel), "n_scored": len(sc), "A_answer_only": k,
                                "acc": round(k / len(sc), 4) if sc else None, "ci95": wilson(k, len(sc)),
                                "outcomes": dict(collections.Counter(s["outcome"] for s in sel))}
    # proprietary models in answer-only mode on the SAME task subset (for comparability with open-weight)
    for model in sorted({r["model"] for r, s in rows if r["source"] == "baseline"}):
        sc = [s for r, s in rows if r["source"] == "baseline" and r["model"] == model
              and s.get("outcome_answer_only") not in (None, "not_verifiable", "not_verifiable_answer_only",
                                                      "infra_error")]
        k = sum(1 for s in sc if s.get("A_answer_only"))
        ao[f"baseline_proprietary_answer_only_mode|{model}"] = {"n_scored": len(sc), "A_answer_only": k,
                                                               "acc": round(k / len(sc), 4) if sc else None,
                                                               "ci95": wilson(k, len(sc))}
    res["answer_only_accuracy"] = ao

    # ---------------- P2 / interceptor / other conditions
    def by(src):
        out = {}
        keys = sorted({(r["condition"], r["model"]) for r, s in rows if r["source"] == src})
        for c, m in keys:
            out[f"{c}|{m}"] = acc(rows, lambda r, s, c=c, m=m: r["source"] == src and r["condition"] == c and r["model"] == m)
            out[f"{c}|{m}"]["outcomes"] = dict(collections.Counter(
                s["outcome"] for r, s in rows if r["source"] == src and r["condition"] == c and r["model"] == m))
        return out
    for src in ("p2", "p2_retail", "interceptor", "interceptor_control", "p4", "ablation", "multiseed", "rerun_C"):
        res[f"accuracy_{src}"] = by(src)
    # P2 per injection type pooled over models (only sessions whose model has baseline traces too)
    res["p2_pooled_by_type"] = {c: acc(rows, lambda r, s, c=c: r["source"] == "p2" and r["condition"] == c)
                                for c in sorted({r["condition"] for r, s in rows if r["source"] == "p2"})}

    # ---------------- write labels
    cols = ["session_id", "task_id", "model", "source", "condition", "db", "mode", "template", "verifiable", "outcome",
            "T", "P", "S", "A", "CORRECT", "A_recall", "A_precision", "P_emitted", "A_answer_only", "n_calls", "n_exec",
            "n_success", "n_schema_err", "n_unsupported", "T_missing", "A_key_fail", "ungrounded", "claim_detail",
            "conditions", "per_slot", "validator_version"]
    with open(f"{OUT}/labels_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r, s in rows:
            row = {**{k: r.get(k) for k in ("session_id", "task_id", "model", "source", "condition", "db", "mode")}, **s}
            row["per_slot"] = json.dumps(s.get("per_slot")) if s.get("per_slot") else ""
            w.writerow(["" if row.get(c) is None else row.get(c) for c in cols])
    res["n_traces_total"] = len(rows)
    res["n_by_mode"] = dict(collections.Counter(r["mode"] for r, s in rows))
    res["excluded_dbs"] = L.EXCLUDED_DBS
    with open(f"{OUT}/results_main.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps({k: res[k] for k in ("n_traces_total", "n_by_mode", "replay_determinism")}, indent=1, default=str)[:3000])
    print(json.dumps(res["baseline_accuracy_by_model"], indent=1)[:6000])


if __name__ == "__main__":
    main()
