"""Reviewer 1, point 5: propagate human-calibration uncertainty.

Calibration model (as in the paper): a = P(human correct | ensemble correct), b = P(human correct | ensemble wrong),
estimated on the 100 human-labelled P2 traces. For any trace set whose ensemble-wrong rate is w,
the calibrated (human-scale) wrong rate is  w_h = 1 - [a(1-w) + b w] = (1-a) + (a-b) w.
For a paired risk difference this gives RD_h = (a-b) * RD_ens (assumes the ensemble's error rates are the same
in the injected and baseline arms, i.e. non-differential misclassification -- an assumption, not verified).

Uncertainty: joint bootstrap -- resample the 100 labels (-> a*, b*) AND the target traces (-> w*) -- 10,000 reps.
Primary labels: Bhaskar (as in paper). Sensitivity: Yoshitha; consensus (both YES); either YES.
"""
import sys, json
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
import numpy as np
from A_common import *
from A_stats import *
from A_10_paired import build_pairs, MODEL_ORDER, TOOLS

B = 10000

def labels():
    bh = {r["session_id"]: r["YOUR_LABEL"].strip().upper() for r in read("human_labeling_sheet_Bhaskar_labeled.csv")}
    yo = {r["session_id"]: r["YOUR_LABEL"].strip().upper() for r in read("human_labeling_sheet_Yoshitha_labeled.csv")}
    p2 = {r["session_id"]: r for r in read("p2_all_rejudged.csv")}
    out = []
    for sid in bh:
        r = p2.get(sid)
        if r is None: continue
        out.append(dict(sid=sid, e=int(tf(r["ensemble_correct"])), g4m=int(tf(r["gpt4o_mini_verdict"])), eps=int(r["eps"]),
                        bh=int(bh[sid] == "YES"), yo=int(yo.get(sid, "") == "YES") if sid in yo else None,
                        err_type=r["error_type"]))
    return out

def ab(L, key, judge="e"):
    ec = [l[key] for l in L if l[judge] == 1]; ew = [l[key] for l in L if l[judge] == 0]
    return (np.mean(ec) if ec else np.nan, np.mean(ew) if ew else np.nan, len(ec), len(ew))

def calib(w, a, b): return (1 - a) + (a - b) * w

def boot_label_ab(L, key, rng, judge="e"):
    idx = rng.integers(0, len(L), size=(B, len(L)))
    E = np.array([l[judge] for l in L]); H = np.array([l[key] for l in L], float)
    Eb, Hb = E[idx], H[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (Hb * Eb).sum(1) / Eb.sum(1); b = (Hb * (1 - Eb)).sum(1) / (1 - Eb).sum(1)
    return a, b

def boot_mean(v, rng):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), size=(B, len(v))); return v[idx].mean(1)

def main():
    L = labels()
    rng = np.random.default_rng(2024)
    res = {"n_labels_joined": len(L), "label_error_types": {}, "label_eps_strata": {}}
    for l in L:
        res["label_error_types"][l["err_type"]] = res["label_error_types"].get(l["err_type"], 0) + 1
        res["label_eps_strata"][str(l["eps"])] = res["label_eps_strata"].get(str(l["eps"]), 0) + 1
    Lyo = [l for l in L if l["yo"] is not None]
    for l in Lyo: l["cons"] = int(l["bh"] and l["yo"]); l["either"] = int(l["bh"] or l["yo"])
    label_sets = {"bhaskar": (L, "bh"), "yoshitha": (Lyo, "yo"), "consensus_both_yes": (Lyo, "cons"), "either_yes": (Lyo, "either")}
    res["ab"] = {}
    for name, (LL, key) in label_sets.items():
        a, b, nc, nw = ab(LL, key)
        ab_ = boot_label_ab(LL, key, np.random.default_rng(11))
        res["ab"][name] = {"a_P_h_correct_given_ens_correct": a, "b_P_h_correct_given_ens_wrong": b, "n_ens_correct": nc,
                           "n_ens_wrong": nw, "a_ci": [float(np.nanpercentile(ab_[0], 2.5)), float(np.nanpercentile(ab_[0], 97.5))],
                           "b_ci": [float(np.nanpercentile(ab_[1], 2.5)), float(np.nanpercentile(ab_[1], 97.5))],
                           "a_minus_b": a - b, "a_minus_b_ci": [float(np.nanpercentile(ab_[0]-ab_[1], 2.5)), float(np.nanpercentile(ab_[0]-ab_[1], 97.5))]}
    # human-direct r23 among labelled S2 traces (tiny n)
    s2l = [l for l in L if l["eps"] >= 2]
    res["human_direct_S2"] = {"n": len(s2l), "human_wrong": int(sum(1 - l["bh"] for l in s2l)),
                              "wilson": list(wilson(sum(1 - l["bh"] for l in s2l), len(s2l)))}
    # target quantities
    p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
    W = lambda rows: [int(not tf(r["ensemble_correct"])) for r in rows]
    targets = {"pooled_r23": W([r for r in p2 if int(r["eps"]) >= 2]),
               "pooled_P_S3_given_S1": W([r for r in p2 if int(r["eps"]) >= 1]),
               "pooled_P_S3_all": W(p2)}
    for m in MODEL_ORDER:
        targets[f"r23|{m}"] = W([r for r in p2 if r["model"] == m and int(r["eps"]) >= 2])
        targets[f"P_S3_given_S1|{m}"] = W([r for r in p2 if r["model"] == m and int(r["eps"]) >= 1])
    res["calibrated"] = {}
    for name, (LL, key) in label_sets.items():
        a, b, _, _ = ab(LL, key)
        A, Bb = boot_label_ab(LL, key, np.random.default_rng(99))
        out = {}
        for t, v in targets.items():
            if not v: continue
            w = np.mean(v); wb = boot_mean(v, rng)
            cb = calib(wb, A, Bb)
            out[t] = {"n": len(v), "ens": float(w), "calibrated": float(calib(w, a, b)),
                      "ci_labels_and_traces": [float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))],
                      "ci_labels_only": [float(np.nanpercentile(calib(w, A, Bb), 2.5)), float(np.nanpercentile(calib(w, A, Bb), 97.5))]}
        res["calibrated"][name] = out
    # calibrated paired RD (ensemble RD * (a-b)), joint bootstrap over labels and paired traces
    pairs, _ = build_pairs()
    strata = {"all": lambda d: True, "applied_T0": lambda d: d["applied"] and d["T"] == 0, "S2": lambda d: d["eps"] >= 2,
              "applied": lambda d: d["applied"]}
    res["calibrated_RD"] = {}
    for name, (LL, key) in label_sets.items():
        a, b, _, _ = ab(LL, key); A, Bb = boot_label_ab(LL, key, np.random.default_rng(5))
        o = {}
        for s, f in strata.items():
            diff = [d["inj_wrong_ensemble"] - d["bl_wrong_ensemble"] for d in pairs if f(d)]
            rd = np.mean(diff); rdb = boot_mean(diff, rng); cb = (A - Bb) * rdb
            o[s] = {"n": len(diff), "RD_ens": float(rd), "RD_calibrated": float((a - b) * rd),
                    "ci": [float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))]}
        res["calibrated_RD"][name] = o
    json.dump(res, open(os.path.join(OUT, "calibration.json"), "w"), indent=2, default=float)
    print(json.dumps(res["ab"], indent=1, default=float))
    print("human direct S2:", res["human_direct_S2"])
    for name in label_sets:
        c = res["calibrated"][name]
        print(name, "pooled r23 ens %.3f -> cal %.3f CI %s" % (c["pooled_r23"]["ens"], c["pooled_r23"]["calibrated"], np.round(c["pooled_r23"]["ci_labels_and_traces"], 3)),
              "| P(S3|S1) %.3f -> %.3f %s" % (c["pooled_P_S3_given_S1"]["ens"], c["pooled_P_S3_given_S1"]["calibrated"], np.round(c["pooled_P_S3_given_S1"]["ci_labels_and_traces"], 3)))
        print("   RD:", {s: (round(v["RD_calibrated"], 4), np.round(v["ci"], 4).tolist()) for s, v in res["calibrated_RD"][name].items()})
    c = res["calibrated"]["bhaskar"]
    for m in MODEL_ORDER:
        v = c.get(f"r23|{m}")
        if v: print(f"  {m:16s} nS2={v['n']:3d} r23 ens={v['ens']:.3f} cal={v['calibrated']:.3f} CI[{v['ci_labels_and_traces'][0]:.3f},{v['ci_labels_and_traces'][1]:.3f}]")

if __name__ == "__main__":
    main()
