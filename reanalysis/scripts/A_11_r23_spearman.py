"""Reviewer 1, points 3-4: per-model r23 with Wilson and bootstrap CIs; rejection/admission vs recovery Spearman
with bootstrap CI and power.

Definitions (paper): S1 = eps>=1, S2 = eps>=2, S3 = judge says wrong; r23 = P(S3 | S2);
'rejection' = 1 - P(S1); 'recovery' = 1 - r23.
Renamed axis: 'tool-call admission' = P(first Action names a valid tool) = P(injector actually invoked).
"""
import sys, json
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
import numpy as np
from scipy import stats
from A_common import *
from A_stats import *
from A_10_paired import TOOLS, JUDGES, MODEL_ORDER

B = 10000

def load():
    p2 = [r for r in read("p2_all_rejudged.csv") if r["error_type"] == "semantic_wrong"]
    rows = []
    for r in p2:
        eps = int(r["eps"]); H = tf(r["heuristic_correct"]); applied = r["tool_selected"] in TOOLS
        d = dict(model=r["model"], task=r["task_id"], S1=int(eps >= 1), S2=int(eps >= 2), applied=int(applied),
                 T=(eps - (0 if H else 1)) if applied else 0)
        for j, col in JUDGES.items(): d["W_" + j] = int(not tf(r[col]))
        rows.append(d)
    return rows

def per_model_r23(rows, judge):
    out = {}
    for m in MODEL_ORDER:
        sub = [d for d in rows if d["model"] == m]
        s2 = [d["W_" + judge] for d in sub if d["S2"]]
        s1 = [d["W_" + judge] for d in sub if d["S1"]]
        k, n = sum(s2), len(s2)
        o = {"n": len(sub), "p_S1": np.mean([d["S1"] for d in sub]), "p_S2": np.mean([d["S2"] for d in sub]),
             "p_S3": np.mean([d["W_" + judge] for d in sub]), "admission": np.mean([d["applied"] for d in sub]),
             "n_S1": len(s1), "n_S2": n, "k_S3_given_S2": k,
             "r23": k / n if n else None, "r23_wilson": list(wilson(k, n)) if n else None,
             "r23_boot": list(boot_ci(s2, B=B)) if n else None,
             "P_S3_given_S1": (sum(s1) / len(s1)) if s1 else None, "P_S3_given_S1_wilson": list(wilson(sum(s1), len(s1))) if s1 else None}
        o["r12"] = o["p_S2"] / o["p_S1"] if o["p_S1"] else None
        out[m] = {k2: (float(v) if isinstance(v, (np.floating,)) else v) for k2, v in o.items()}
    s2 = [d["W_" + judge] for d in rows if d["S2"]]
    out["_pooled"] = {"n_S2": len(s2), "k": sum(s2), "r23": sum(s2) / len(s2), "wilson": list(wilson(sum(s2), len(s2))),
                      "task_cluster_boot": list(cluster_boot_rd([(d["task"], d["W_" + judge], 0) for d in rows if d["S2"]], "task", B=B))}
    return out

def spearman_boot(rows, judge, xaxis, min_ns2=1, seed=7):
    """Bootstrap: resample traces within each model (stratified), recompute per-model x and recovery, then rho."""
    rng = np.random.default_rng(seed)
    by = {m: [d for d in rows if d["model"] == m] for m in MODEL_ORDER}
    def point(samples):
        xs, ys = [], []
        for m, sub in samples.items():
            s2 = [d for d in sub if d["S2"]]
            if len(s2) < min_ns2: continue
            x = 1 - np.mean([d["S1"] for d in sub]) if xaxis == "rejection" else np.mean([d["applied"] for d in sub])
            y = 1 - np.mean([d["W_" + judge] for d in s2])
            xs.append(x); ys.append(y)
        return xs, ys
    xs, ys = point(by)
    rho, p = stats.spearmanr(xs, ys)
    bs = []
    for _ in range(2000):
        smp = {m: [sub[i] for i in rng.integers(0, len(sub), len(sub))] for m, sub in by.items()}
        bx, byy = point(smp)
        if len(bx) >= 4 and len(set(bx)) > 1 and len(set(byy)) > 1:
            bs.append(stats.spearmanr(bx, byy)[0])
    bs = np.array(bs)
    # exact permutation p-value (n small): enumerate via Monte Carlo 200k permutations
    prng = np.random.default_rng(seed + 1); yv = np.array(ys); cnt = 0; NP = 200000
    rx = stats.rankdata(xs)
    for _ in range(NP):
        if abs(stats.spearmanr(rx, prng.permutation(yv))[0]) >= abs(rho) - 1e-12: cnt += 1
    return {"n_models": len(xs), "rho": float(rho), "p_asymptotic": float(p), "p_permutation": (cnt + 1) / (NP + 1),
            "boot_ci_95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "boot_reps_valid": int(len(bs)),
            "points": [{"x": float(a), "recovery": float(b)} for a, b in zip(xs, ys)]}

def power_spearman(n, rhos=(0.3, 0.5, 0.7, 0.8), alpha=0.05, sims=20000, seed=3):
    """Monte-Carlo power of a two-sided Spearman test with n points drawn from a bivariate normal with
    Pearson correlation r chosen so that the population Spearman equals rho_s (r = 2 sin(pi*rho_s/6))."""
    rng = np.random.default_rng(seed); out = {}
    crit = None
    # critical |rho| from the null distribution
    null = np.array([stats.spearmanr(rng.standard_normal(n), rng.standard_normal(n))[0] for _ in range(sims)])
    crit = float(np.quantile(np.abs(null), 1 - alpha))
    for rs in rhos:
        r = 2 * np.sin(np.pi * rs / 6)
        cov = [[1, r], [r, 1]]
        hits = 0
        for _ in range(sims // 4):
            z = rng.multivariate_normal([0, 0], cov, size=n)
            if abs(stats.spearmanr(z[:, 0], z[:, 1])[0]) >= crit: hits += 1
        out[str(rs)] = hits / (sims // 4)
    # minimal detectable |rho| at 80% power (grid)
    mdr = None
    for rs in np.arange(0.40, 0.96, 0.025):
        r = 2 * np.sin(np.pi * rs / 6); cov = [[1, r], [r, 1]]
        hits = sum(abs(stats.spearmanr(*rng.multivariate_normal([0, 0], cov, size=n).T)[0]) >= crit for _ in range(3000))
        if hits / 3000 >= 0.80: mdr = float(round(rs, 3)); break
    return {"n": n, "critical_abs_rho_alpha_0.05": crit, "power_at_true_rho": out, "min_detectable_rho_80pct_power": mdr}

def main():
    rows = load()
    res = {"r23": {j: per_model_r23(rows, j) for j in ["ensemble", "gpt4o_mini", "heuristic", "gpt4o", "gemini_25_flash"]}}
    res["spearman"] = {}
    for j in ["ensemble", "gpt4o_mini"]:
        for xa in ["rejection", "admission"]:
            for mn in [1, 10]:
                key = f"{j}|{xa}|min_nS2={mn}"
                res["spearman"][key] = spearman_boot(rows, j, xa, min_ns2=mn)
                v = res["spearman"][key]
                print(key, "n=", v["n_models"], "rho=%.3f" % v["rho"], "p_asym=%.3f p_perm=%.3f" % (v["p_asymptotic"], v["p_permutation"]),
                      "boot CI [%.3f, %.3f]" % tuple(v["boot_ci_95"]))
    res["power"] = {str(n): power_spearman(n) for n in (13, 9)}
    print(json.dumps(res["power"], indent=1))
    # admission vs rejection relationship
    adm = [res["r23"]["ensemble"][m]["admission"] for m in MODEL_ORDER]
    rej = [1 - res["r23"]["ensemble"][m]["p_S1"] for m in MODEL_ORDER]
    res["admission_vs_rejection_spearman"] = list(map(float, stats.spearmanr(adm, rej)))
    json.dump(res, open(os.path.join(OUT, "r23_spearman.json"), "w"), indent=2, default=float)
    for j in ["ensemble", "gpt4o_mini", "heuristic"]:
        print("== r23 judge", j, "pooled", res["r23"][j]["_pooled"])
        for m in MODEL_ORDER:
            v = res["r23"][j][m]
            print(f"  {m:16s} adm={v['admission']:.3f} S1={v['p_S1']:.3f} S2={v['p_S2']:.3f} nS2={v['n_S2']:3d} r23={v['r23']:.3f} "
                  f"W[{v['r23_wilson'][0]:.3f},{v['r23_wilson'][1]:.3f}] B[{v['r23_boot'][0]:.3f},{v['r23_boot'][1]:.3f}] P(S3|S1)={v['P_S3_given_S1']:.3f}")

if __name__ == "__main__":
    main()
