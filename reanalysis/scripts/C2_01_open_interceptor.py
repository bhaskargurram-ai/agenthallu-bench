"""C2_01: Re-analysis of the open-weight interceptor CSVs (original project, data only).
Arms: no_interceptor (control) vs early_interceptor, same 300 task_ids, run concurrently in one job.
Variants: 'full' (L1+L3 as run, interceptor_<m>.csv) and 'schemaonly' (INT_NO_L3=1, L1 only; separate re-run of BOTH arms).
Also re-runs identical metrics on GPT-4o-mini / Gemini (interceptor_{control,heldout}_600.csv) for a 5-model table.
Labels: harness substring heuristic (final_correct). Writes C2_openweight/open_results.json.
"""
import json, re, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import *  # noqa


def analyse(C, I, label_fn=None, exclude=None):
    ids = sorted(set(C) & set(I))
    if exclude:
        ids = [t for t in ids if not exclude(t)]
    n = len(ids)
    if label_fn is None:
        cwrong = [not tf(C[t]["final_correct"]) for t in ids]
        iwrong = [not tf(I[t]["final_correct"]) for t in ids]
    else:
        cwrong = [label_fn(C[t]) for t in ids]
        iwrong = [label_fn(I[t]) for t in ids]
    ab = [tf(I[t]["abstained"]) for t in ids]
    ch = cwrong
    ih = [iwrong[i] and not ab[i] for i in range(n)]
    H_c, H_i, k = sum(ch), sum(ih), sum(ab)
    sv = sum(1 for i in range(n) if ab[i] and ch[i]); bk = k - sv
    b = sum(1 for i in range(n) if ch[i] and not ih[i]); c = sum(1 for i in range(n) if (not ch[i]) and ih[i])
    lo, hi = paired_boot([int(x) for x in ch], [int(x) for x in ih])
    ulo, uhi = unpaired_normal_ci(H_c / n, H_i / n, n)
    ans = [i for i in range(n) if not ab[i]]
    exp_h, sims = random_abstention(ch, k)
    p_sim = sum(1 for s in sims if s >= sv) / len(sims)
    sims_sorted = sorted(sims)
    # hallucination rate a random abstainer would reach given sims (5/95 pct)
    rand_h_lo = (H_c - sims_sorted[int(0.975 * len(sims)) - 1]) / n
    rand_h_hi = (H_c - sims_sorted[int(0.025 * len(sims))]) / n
    # answered-subset: control errors on the tasks interceptor answered vs int errors there (run-to-run noise)
    c_wrong_ans = sum(ch[i] for i in ans); i_wrong_ans = sum(iwrong[i] for i in ans)
    lay = [layer(I[t]["abstain_reason"]) if ab[i] else None for i, t in enumerate(ids)]
    bylayer = {}
    for L in ["L1", "L3"]:
        ix = [i for i in range(n) if lay[i] == L]
        s = sum(ch[i] for i in ix)
        bylayer[L] = {"abstentions": len(ix), "saved": s, "broken": len(ix) - s,
                      "precision": s / len(ix) if ix else None,
                      "p_vs_random_hypergeom": hypergeom_sf(s, n, H_c, len(ix)) if ix else None}
    return {
        "n": n, "ctrl_halluc_pct": 100 * H_c / n, "int_halluc_pct": 100 * H_i / n,
        "delta_pp(int-ctrl)": 100 * (H_i - H_c) / n,
        "delta_paired_boot95_pp": [100 * lo, 100 * hi],
        "delta_unpaired_normal95_pp": [100 * ulo, 100 * uhi],
        "mcnemar": {"b_ctrlH_intNotH": b, "c_ctrlNotH_intH": c, "p_exact": mcnemar_exact(b, c)},
        "abstentions": k, "abstain_pct": 100 * k / n, "coverage_pct": 100 * (1 - k / n),
        "acc_answered_int_pct": 100 * (1 - i_wrong_ans / len(ans)) if ans else None,
        "ctrl_acc_pct": 100 * (1 - H_c / n),
        "ctrl_acc_on_int_answered_tasks_pct": 100 * (1 - c_wrong_ans / len(ans)) if ans else None,
        "saved": sv, "broken": bk,
        "precision": sv / k if k else None, "precision_wilson95": wilson(sv, k) if k else None,
        "base_rate": H_c / n, "recall": sv / H_c if H_c else None,
        "random_abstention": {"expected_halluc_pct": 100 * exp_h, "expected_saved": k * H_c / n,
                              "sim95_halluc_pct": [100 * rand_h_lo, 100 * rand_h_hi],
                              "int_minus_random_pp": 100 * (H_i / n - exp_h),
                              "transfer_halluc_pct(int abstentions applied to ctrl)": 100 * (H_c - sv) / n,
                              "transfer_minus_random_pp": 100 * ((H_c - sv) / n - exp_h),
                              "p_sim_one_sided(saved>=obs)": p_sim,
                              "p_hypergeom_one_sided": hypergeom_sf(sv, n, H_c, k),
                              "p_hypergeom_worse_than_random": 1 - hypergeom_sf(sv + 1, n, H_c, k)},
        "answered_tasks_run_to_run": {"n_answered": len(ans), "ctrl_wrong": c_wrong_ans, "int_wrong": i_wrong_ans},
        "by_layer": bylayer,
    }


def l3_kind(reason):
    nums = re.findall(r"'(-?[\d.]+)'", reason)
    if not nums:
        return "unparsed"
    if all(re.fullmatch(r"\d\.", x) for x in nums):
        return "list_numbering_only"
    if any(re.fullmatch(r"(19|20)\d\d\.?", x) for x in nums):
        return "contains_year"
    return "other_numbers"


def ctrl_failure_type(r):
    fa = r["final_answer"]; st = int(r["steps"])
    if fa.startswith("ERROR"):
        return "llm_error"
    if fa.startswith("Max steps"):
        return "max_steps_no_answer"
    if st <= 2:
        return "answered_without_tool"
    return "used_tool(s)"


def details(C, I):
    ids = sorted(set(C) & set(I))
    ch = {t: not tf(C[t]["final_correct"]) for t in ids}
    ab = {t: tf(I[t]["abstained"]) for t in ids}
    out = {}
    # failure type of control arm x abstention (cross-arm: types come from the control run)
    ft = collections.defaultdict(lambda: collections.Counter())
    for t in ids:
        k = ctrl_failure_type(C[t])
        ft[k]["n"] += 1; ft[k]["ctrl_wrong"] += ch[t]
        ft[k]["int_abst"] += ab[t]; ft[k]["int_abst_on_ctrl_wrong"] += ab[t] and ch[t]
    out["ctrl_failure_type"] = {k: dict(v) for k, v in ft.items()}
    # interceptor-arm own trajectory type x abstention layer
    it = collections.defaultdict(lambda: collections.Counter())
    for t in ids:
        st = int(I[t]["steps"]); L = layer(I[t]["abstain_reason"]) if ab[t] else "answered"
        kind = "no_tool(steps<=2)" if st <= 2 else "tool_used"
        it[kind][L] += 1
    out["int_arm_tooluse_x_outcome"] = {k: dict(v) for k, v in it.items()}
    # L1 subtype
    l1 = collections.Counter()
    for t in ids:
        if ab[t] and layer(I[t]["abstain_reason"]) == "L1":
            r = I[t]["abstain_reason"]
            l1[("param_errors" if "param_errors" in r else "tool_error") + ("|ctrl_wrong" if ch[t] else "|ctrl_right")] += 1
    out["L1_subtype"] = dict(l1)
    l3 = collections.Counter()
    for t in ids:
        if ab[t] and layer(I[t]["abstain_reason"]) == "L3":
            l3[l3_kind(I[t]["abstain_reason"]) + ("|ctrl_wrong" if ch[t] else "|ctrl_right")] += 1
    out["L3_trigger_kind"] = dict(l3)
    # param error messages (top)
    pe = collections.Counter()
    for t in ids:
        if ab[t] and layer(I[t]["abstain_reason"]) == "L1":
            m = re.search(r"param_errors=\[(.*)", I[t]["abstain_reason"])
            pe[(m.group(1) if m else I[t]["abstain_reason"])[:70]] += 1
    out["L1_top_messages"] = pe.most_common(8)
    # by source
    bys = {}
    for s in sorted({src(t) for t in ids}):
        ix = [t for t in ids if src(t) == s]
        bys[s] = {"n": len(ix), "ctrl_wrong": sum(ch[t] for t in ix),
                  "int_halluc": sum((not tf(I[t]["final_correct"])) and not ab[t] for t in ix),
                  "abst": sum(ab[t] for t in ix), "saved": sum(ab[t] and ch[t] for t in ix),
                  "broken": sum(ab[t] and not ch[t] for t in ix)}
    out["by_source"] = bys
    out["ctrl_steps_hist"] = dict(collections.Counter(int(C[t]["steps"]) for t in ids))
    out["int_steps_hist"] = dict(collections.Counter(int(I[t]["steps"]) for t in ids))
    return out


def arms(rows, m):
    C = {r["task_id"]: r for r in rows if r["model"] == m and r["condition"] == "no_interceptor"}
    I = {r["task_id"]: r for r in rows if r["model"] == m and r["condition"] == "early_interceptor"}
    return C, I


if __name__ == "__main__":
    res = {"open": {}, "proprietary": {}}
    halu = lambda t: src(t) == "halueval"
    for m in OPEN:
        res["open"][m] = {}
        for var, fn in [("full", f"interceptor_{m}.csv"), ("schemaonly", f"interceptor_{m}_schemaonly.csv")]:
            C, I = arms(read_orig(fn), m)
            assert len(C) == 300 and len(I) == 300 and set(C) == set(I)
            res["open"][m][var] = {"all": analyse(C, I), "excl_halueval": analyse(C, I, exclude=halu),
                                   "details": details(C, I)}
        # replicate controls (main vs schemaonly)
        C1, _ = arms(read_orig(f"interceptor_{m}.csv"), m)
        C2, _ = arms(read_orig(f"interceptor_{m}_schemaonly.csv"), m)
        a = [not tf(C1[t]["final_correct"]) for t in sorted(C1)]
        b = [not tf(C2[t]["final_correct"]) for t in sorted(C1)]
        res["open"][m]["control_replicate"] = {
            "ctrl1_halluc_pct": 100 * sum(a) / 300, "ctrl2_halluc_pct": 100 * sum(b) / 300,
            "flips_wrong_to_right": sum(x and not y for x, y in zip(a, b)),
            "flips_right_to_wrong": sum(y and not x for x, y in zip(a, b)),
            "identical_answer_text": sum(C1[t]["final_answer"] == C2[t]["final_answer"] for t in C1)}
        # cross-run: apply schema-only abstention set to the FULL run's control (more stable estimate)
        _, I2 = arms(read_orig(f"interceptor_{m}_schemaonly.csv"), m)
        res["open"][m]["schemaonly_vs_full_control"] = analyse(C1, I2)
    ctrl = read_orig("interceptor_control_600.csv"); intr = read_orig("interceptor_heldout_600.csv")
    for m in ["gpt4o_mini", "gemini_20_flash"]:
        C = {r["task_id"]: r for r in ctrl if r["model"] == m}
        I = {r["task_id"]: r for r in intr if r["model"] == m}
        res["proprietary"][m] = {"all": analyse(C, I), "excl_halueval": analyse(C, I, exclude=halu), "details": details(C, I)}
    json.dump(res, open(f"{OUT}/open_results.json", "w"), indent=1, default=str)
    for grp in res.values():
        for m, d in grp.items():
            for var in ("full", "schemaonly", "all"):
                if var not in d:
                    continue
                x = d[var]["all"] if var != "all" else d["all"]
                r = x["random_abstention"]
                print(f"{m:16s} {var:10s} n={x['n']} ctrl={x['ctrl_halluc_pct']:.1f} int={x['int_halluc_pct']:.1f} "
                      f"d={x['delta_pp(int-ctrl)']:.1f} CI={[round(v,1) for v in x['delta_paired_boot95_pp']]} "
                      f"unp={[round(v,1) for v in x['delta_unpaired_normal95_pp']]} McN={x['mcnemar']} "
                      f"abst={x['abstentions']} cov={x['coverage_pct']:.1f} accAns={x['acc_answered_int_pct']:.1f} "
                      f"ctrlAcc={x['ctrl_acc_pct']:.1f} ctrlAccOnAns={x['ctrl_acc_on_int_answered_tasks_pct']:.1f} "
                      f"Sv/Bk={x['saved']}/{x['broken']} P={x['precision']:.3f} base={x['base_rate']:.3f} R={x['recall']:.3f} "
                      f"rand={r['expected_halluc_pct']:.1f} int-rand={r['int_minus_random_pp']:.1f} transfer={r['transfer_halluc_pct(int abstentions applied to ctrl)']:.1f} "
                      f"psim={r['p_sim_one_sided(saved>=obs)']:.4f} pworse={r['p_hypergeom_worse_than_random']:.4g} "
                      f"ansRun={x['answered_tasks_run_to_run']} layers={x['by_layer']}")
                print()
