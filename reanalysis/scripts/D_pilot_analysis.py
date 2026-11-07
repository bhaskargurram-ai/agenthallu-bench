"""Merge paired sim-vs-real pilot JSONL (all models) and compute paired statistics.
Usage: python3 -I D_pilot_analysis.py <revised_results_dir> <batches_dir> <outdir>"""
import json, os, sys, glob, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D_stats import wilson, mcnemar_exact, newcombe_paired
res_dir, bdir, outdir = sys.argv[1:4]
MARGIN = 0.10  # pre-stated equivalence margin (+/-10pp), TOST at alpha=0.05 <=> 90% CI inside margin
recs = collections.OrderedDict()
DROPPED = collections.Counter()
def add(model, path, origin):
    bd = os.path.dirname(path); ans = {}
    for arm in ("sim", "real"):
        sp = os.path.join(bd, f"{arm}_scores.jsonl")
        if os.path.exists(sp):
            for l in open(sp):
                if l.strip():
                    x = json.loads(l); ans[(arm, x["id"])] = x.get("answer", "")
    for l in open(path):
        if not l.strip(): continue
        d = json.loads(l)
        if d.get("status") != "ok": DROPPED[(model, "status")] += 1; continue
        # validity guard: revision-run records must have non-empty answers in both arms
        if origin == "revision_run" and (not ans.get(("sim", d["id"])) or not ans.get(("real", d["id"]))):
            DROPPED[(model, "empty_answer")] += 1; continue
        recs[(model, d["id"])] = {"model": model, "origin": origin, **d}
# original 37 gpt-4o-mini tasks (run before this revision)
add("gpt-4o-mini", os.path.join(res_dir, "paired_robust", "incremental.jsonl"), "original_run")
add("gpt-4o-mini", os.path.join(res_dir, "paired_robust_tail", "incremental.jsonl"), "original_run")
for p in sorted(glob.glob(os.path.join(bdir, "*", "b*", "incremental.jsonl"))):
    model = p.split(os.sep)[-3]
    k0 = len(recs); add(model, p, "revision_run")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "pilot_merged.jsonl"), "w") as f:
    for r in recs.values(): f.write(json.dumps(r) + "\n")
by = collections.defaultdict(list)
for r in recs.values(): by[r["model"]].append(r)

def paired(rows, key):
    a = sum(1 for r in rows if r["real"][key] and r["sim"][key])
    b = sum(1 for r in rows if r["real"][key] and not r["sim"][key])
    c = sum(1 for r in rows if not r["real"][key] and r["sim"][key])
    d = len(rows) - a - b - c
    D, L95, U95 = newcombe_paired(a, b, c, d, 0.95)
    _, L90, U90 = newcombe_paired(a, b, c, d, 0.90)
    n = len(rows)
    return {"n": n, "sim_rate": (a + c) / n, "sim_ci95": wilson(a + c, n), "real_rate": (a + b) / n,
            "real_ci95": wilson(a + b, n), "both": a, "real_only": b, "sim_only": c, "neither": d,
            "diff_real_minus_sim": D, "diff_ci95": [L95, U95], "diff_ci90": [L90, U90],
            "mcnemar_exact_p": mcnemar_exact(b, c),
            "tost_equivalent_at_10pp": bool(L90 > -MARGIN and U90 < MARGIN)}

def cluster_boot(rows, key, B=5000, seed=1):
    """Pooled paired diff with task-clustered bootstrap (tasks shared across models)."""
    tasks = sorted(set(r["id"] for r in rows)); byt = collections.defaultdict(list)
    for r in rows: byt[r["id"]].append(int(r["real"][key]) - int(r["sim"][key]))
    rng = random.Random(seed); est = []
    for _ in range(B):
        s = [x for t in (rng.choice(tasks) for _ in tasks) for x in byt[t]]
        est.append(sum(s) / len(s))
    est.sort()
    return [est[int(0.025 * B)], est[int(0.975 * B) - 1]], [est[int(0.05 * B)], est[int(0.95 * B) - 1]]

out = {"dropped": {f"{k[0]}:{k[1]}": v for k, v in DROPPED.items()}, "equivalence_margin_pp": MARGIN * 100, "per_model": {}, "pooled": {}}
for m, rows in by.items():
    out["per_model"][m] = {"fabrication": paired(rows, "fabricated"), "tool_call": paired(rows, "made_call"),
                           "n_original_run": sum(1 for r in rows if r["origin"] == "original_run"),
                           "domains": dict(collections.Counter(r["domain"] for r in rows))}
allrows = list(recs.values())
for key in ("fabricated", "made_call"):
    p = paired(allrows, key)
    ci95, ci90 = cluster_boot(allrows, key)
    p["diff_ci95_task_cluster_bootstrap"] = ci95; p["diff_ci90_task_cluster_bootstrap"] = ci90
    p["tost_equivalent_at_10pp_cluster"] = bool(ci90[0] > -MARGIN and ci90[1] < MARGIN)
    out["pooled"]["fabrication" if key == "fabricated" else "tool_call"] = p
# mechanism of discordance: was the arm difference decided BEFORE any tool output existed?
mech = collections.Counter()
for r in allrows:
    sm, rl = r["sim"], r["real"]
    if sm["fabricated"] == rl["fabricated"]: continue
    if sm["made_call"] != rl["made_call"]: mech["tool_call_decision_differs_pre_execution"] += 1
    elif sm["n_success"] == 0 and rl["n_success"] == 0: mech["no_successful_call_in_either_arm_answer_hedging_differs"] += 1
    else: mech["same_decision_execution_success_differs"] += 1
out["discordance_mechanism"] = dict(mech)
out["discordance_mechanism"]["n_discordant"] = sum(mech.values())
# reproduce the originally reported 37-task result
orig = [r for r in allrows if r["origin"] == "original_run"]
out["original_37_reproduced"] = paired(orig, "fabricated")
json.dump(out, open(os.path.join(outdir, "pilot_results.json"), "w"), indent=2)
print("dropped", dict(DROPPED))
for m, v in out["per_model"].items():
    f = v["fabrication"]
    print(f"{m:13s} n={f['n']} sim={f['sim_rate']:.3f} real={f['real_rate']:.3f} diff={100*f['diff_real_minus_sim']:+.1f} "
          f"CI95=[{100*f['diff_ci95'][0]:+.1f},{100*f['diff_ci95'][1]:+.1f}] CI90=[{100*f['diff_ci90'][0]:+.1f},{100*f['diff_ci90'][1]:+.1f}] "
          f"b={f['real_only']} c={f['sim_only']} p={f['mcnemar_exact_p']:.3f} TOST={f['tost_equivalent_at_10pp']} | TC sim={v['tool_call']['sim_rate']:.3f} real={v['tool_call']['real_rate']:.3f}")
for k, f in out["pooled"].items():
    print("POOLED", k, f"n={f['n']} sim={f['sim_rate']:.3f} real={f['real_rate']:.3f} diff={100*f['diff_real_minus_sim']:+.1f} "
          f"CI95N=[{100*f['diff_ci95'][0]:+.1f},{100*f['diff_ci95'][1]:+.1f}] CI90N=[{100*f['diff_ci90'][0]:+.1f},{100*f['diff_ci90'][1]:+.1f}] "
          f"boot95={[round(100*x,1) for x in f['diff_ci95_task_cluster_bootstrap']]} boot90={[round(100*x,1) for x in f['diff_ci90_task_cluster_bootstrap']]} "
          f"b={f['real_only']} c={f['sim_only']} p={f['mcnemar_exact_p']:.4f} TOST={f['tost_equivalent_at_10pp']}/{f['tost_equivalent_at_10pp_cluster']}")
o = out["original_37_reproduced"]
print("orig37", o["n"], o["sim_rate"], o["real_rate"], [round(100*x,1) for x in o["diff_ci95"]], o["real_only"], o["sim_only"])
