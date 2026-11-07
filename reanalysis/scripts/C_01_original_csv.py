"""C_01: Analysis of the ORIGINAL interceptor CSVs (GPT-4o-mini, Gemini-2.0-Flash; n=600 per arm).

Paired by task_id. Writes C_interceptor/original_csv_results.json.
Limitations: the two arms are separate stochastic agent runs (control was run after the interceptor arm);
the CSVs do not contain chain-of-thought / observations, so layers cannot be replayed from them.
"""
import json, random, re, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import read, tf, mcnemar_exact, hypergeom_sf, boot_ci, wilson, OUT

ctrl = read("interceptor_control_600.csv")
intr = read("interceptor_heldout_600.csv")
out = {}


def src(tid):
    return tid.rsplit("_", 1)[0]


def layer(reason):
    if not reason:
        return None
    if reason.startswith("Early interceptor L3"):
        return "L3"
    if reason.startswith("Early interceptor: L1"):
        return "L1"
    return "other"


for m in ["gpt4o_mini", "gemini_20_flash"]:
    C = {r["task_id"]: r for r in ctrl if r["model"] == m}
    I = {r["task_id"]: r for r in intr if r["model"] == m}
    assert len(C) == 600 and len(I) == 600 and set(C) == set(I)
    ids = sorted(C)
    n = len(ids)
    ch = [not tf(C[t]["final_correct"]) for t in ids]           # control hallucination (= heuristic wrong)
    # sanity: control hallucination column equals not final_correct
    assert all(ch[i] == tf(C[t]["hallucination"]) for i, t in enumerate(ids))
    ab = [tf(I[t]["abstained"]) for t in ids]
    ic = [tf(I[t]["final_correct"]) for t in ids]
    ih = [tf(I[t]["hallucination"]) for t in ids]
    assert all(ih[i] == ((not ic[i]) and (not ab[i])) for i in range(n))
    # any abstained answers judged correct?
    ab_correct = sum(1 for i in range(n) if ab[i] and ic[i])
    lay = [layer(I[t]["abstain_reason"]) if ab[i] else None for i, t in enumerate(ids)]
    k_ab = sum(ab)
    svA = sum(1 for i in range(n) if ab[i] and ch[i])
    bkA = sum(1 for i in range(n) if ab[i] and not ch[i])
    svB = sum(1 for i in range(n) if ch[i] and not ih[i])
    bkB = sum(1 for i in range(n) if (not ch[i]) and ih[i])
    # arm-level
    H_c, H_i = sum(ch), sum(ih)
    delta = (H_i - H_c) / n

    def dfn(idx):
        return (sum(ih[i] for i in idx) - sum(ch[i] for i in idx)) / len(idx)
    lo, hi = boot_ci(dfn, n)
    answered = [i for i in range(n) if not ab[i]]
    sel_acc = sum(ic[i] for i in answered) / len(answered)
    ctrl_acc = 1 - H_c / n
    # transfer of interceptor's abstention set onto control (removes run-to-run noise in the answered part)
    transfer_h = sum(1 for i in range(n) if ch[i] and not ab[i])
    # random abstention at matched k on the control arm: hypergeometric
    exp_sv_rand = k_ab * H_c / n
    p_better_than_random = hypergeom_sf(svA, n, H_c, k_ab)
    rng = random.Random(42)
    sims = []
    for _ in range(10000):
        S = rng.sample(range(n), k_ab)
        sims.append(sum(ch[i] for i in range(n) if i not in set(S)) / n)
    sims.sort()
    rand_h = sum(sims) / len(sims)
    rand_ci = (sims[249], sims[9749])
    # no-tool-call baseline on control (steps==2: one LLM turn ending in Final Answer, no tool executed)
    steps_c = [int(C[t]["steps"]) for t in ids]
    notool = [s <= 2 for s in steps_c]
    k_nt = sum(notool)
    sv_nt = sum(1 for i in range(n) if notool[i] and ch[i])
    # by layer
    bylayer = {}
    for L in ["L1", "L3"]:
        idx = [i for i in range(n) if lay[i] == L]
        sv = sum(ch[i] for i in idx)
        bylayer[L] = {"abstentions": len(idx), "ctrl_hallucinated": sv, "ctrl_correct": len(idx) - sv,
                      "precision_proxy": sv / len(idx) if idx else None,
                      "p_vs_random(hypergeom, one-sided)": hypergeom_sf(sv, n, H_c, len(idx)) if idx else None}
    # L3 triggers driven by list numbering tokens like '1.', '2.'
    l3_reasons = [I[t]["abstain_reason"] for i, t in enumerate(ids) if lay[i] == "L3"]
    l3_enum = sum(1 for r in l3_reasons if re.search(r"'\d\.'", r) and not re.search(r"'\d{2,}", r))
    # by source
    bysrc = {}
    for s in sorted({src(t) for t in ids}):
        idx = [i for i, t in enumerate(ids) if src(t) == s]
        bysrc[s] = {"n": len(idx), "ctrl_h": sum(ch[i] for i in idx) / len(idx), "int_h": sum(ih[i] for i in idx) / len(idx),
                    "abst": sum(ab[i] for i in idx) / len(idx),
                    "svA": sum(1 for i in idx if ab[i] and ch[i]), "bkA": sum(1 for i in idx if ab[i] and not ch[i]),
                    "ctrl_notool": sum(notool[i] for i in idx) / len(idx)}
    # agreement of answered pairs: run-to-run noise among tasks where interceptor did NOT abstain
    disc = sum(1 for i in answered if ch[i] != (not ic[i]))
    out[m] = {
        "n": n, "ctrl_halluc": H_c / n, "int_halluc": H_i / n, "delta_pp": 100 * delta,
        "delta_boot95": [100 * lo, 100 * hi],
        "mcnemar": {"saved_B(ctrl_h & !int_h)": svB, "broken_B(!ctrl_h & int_h)": bkB, "p_exact": mcnemar_exact(svB, bkB)},
        "abstain_rate": k_ab / n, "abstentions": k_ab, "abstained_but_judged_correct": ab_correct,
        "saved_A(abst & ctrl_h)": svA, "broken_A(abst & !ctrl_h)": bkA,
        "coverage": 1 - k_ab / n,
        "selective_accuracy_int_answered": sel_acc, "selective_halluc_int_answered": 1 - sel_acc,
        "ctrl_accuracy": ctrl_acc,
        "abstention_precision_proxy": svA / k_ab if k_ab else None,
        "abstention_precision_wilson95": wilson(svA, k_ab) if k_ab else None,
        "abstention_recall_proxy": svA / H_c,
        "base_rate_ctrl_halluc": H_c / n,
        "transfer_halluc_if_int_abstentions_applied_to_ctrl": transfer_h / n,
        "random_abstention_matched": {"k": k_ab, "expected_saved": exp_sv_rand, "expected_halluc": rand_h,
                                      "sim95": list(rand_ci), "p_one_sided_int_better(hypergeom)": p_better_than_random},
        "no_tool_call_baseline_on_ctrl": {"k": k_nt, "rate": k_nt / n, "saved": sv_nt, "broken": k_nt - sv_nt,
                                          "precision": sv_nt / k_nt if k_nt else None,
                                          "halluc_after": (H_c - sv_nt) / n},
        "by_layer": bylayer,
        "L3_abstentions_triggered_only_by_list_numbering": l3_enum,
        "by_source": bysrc,
        "answered_pairs_label_disagreement": {"n_answered": len(answered), "disagree": disc},
        "ctrl_steps_hist": dict(collections.Counter(steps_c)),
        "flag_count_hist_int": dict(collections.Counter(I[t]["flag_count"] for t in ids)),
    }

json.dump(out, open(f"{OUT}/original_csv_results.json", "w"), indent=1, default=str)
for m, d in out.items():
    print("=====", m)
    for k, v in d.items():
        if k not in ("by_source",):
            print(k, v)
    for s, v in d["by_source"].items():
        print("  ", s, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
