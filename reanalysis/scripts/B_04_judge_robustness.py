import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""Critique 3 analysis: LLM-judge robustness to prompt wording and judge choice."""
import statistics
import sys
from collections import defaultdict
sys.path.insert(0, _REPO + "/reanalysis/scripts")
from B_common import *  # noqa

ann = load_annotators()
ids = sample_ids()
p2 = load_p2()
maj = {s: int(sum(ann[a][s] for a in ann) >= 2) for s in ids}

V = defaultdict(dict)
errors = defaultdict(int)
unparsed = defaultdict(int)
cost = 0.0
with open(os.path.join(OUT, "B03_rejudge_raw.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        cost += d.get("cost", 0.0)
        k = (d["judge"], d["prompt"])
        if d.get("error"):
            errors[k] += 1
            continue
        if d["verdict"] is None:
            unparsed[k] += 1
            V[k][d["session_id"]] = 0  # unparseable -> NO (same convention as original llm_judge)
        else:
            V[k][d["session_id"]] = d["verdict"]

res = {"total_cost_usd": cost, "configs": {}, "errors": {f"{a}|{b}": v for (a, b), v in errors.items()},
       "unparsed_counted_as_NO": {f"{a}|{b}": v for (a, b), v in unparsed.items()}}
refs = {"bhaskar": ann["bhaskar"], "yoshitha": ann["yoshitha"], "third": ann["third"], "majority3": maj}
for (j, pid), m in sorted(V.items()):
    if len(m) < 100:
        print("incomplete", j, pid, len(m))
    sids = [s for s in ids if s in m]
    pred = [m[s] for s in sids]
    e = {"n": len(sids), "yes_rate": sum(pred) / len(pred)}
    for rn, rv in refs.items():
        k, ci = kappa_ci(pred, [rv[s] for s in sids])
        e[rn] = {"kappa": k, "ci95": ci, "agreement": raw_agree(pred, [rv[s] for s in sids])}
    # direction of disagreement vs majority
    e["judge_yes_human_no"] = sum(m[s] and not maj[s] for s in sids)
    e["judge_no_human_yes"] = sum((not m[s]) and maj[s] for s in sids)
    res["configs"][f"{j}|{pid}"] = e

# Reproducibility of the stored original verdicts (same prompt, same model, temperature 0)
repro = {}
for j, col in [("gpt-4o-mini", "gpt4o_mini_verdict"), ("gpt-4o", "gpt4o_verdict")]:
    m = V.get((j, "P0_original"))
    if m:
        stored = [to01(p2[s][col]) for s in ids]
        new = [m[s] for s in ids]
        repro[j] = {"agreement_with_stored": raw_agree(stored, new), "kappa_with_stored": cohen_kappa(stored, new),
                    "stored_yes": sum(stored), "new_yes": sum(new)}
res["reproducibility_P0_vs_stored"] = repro

# Variance across prompts per judge (vs majority3)
by_judge = defaultdict(list)
for key, e in res["configs"].items():
    j, pid = key.split("|")
    by_judge[j].append((pid, e["majority3"]["kappa"], e["yes_rate"]))
res["prompt_sensitivity_vs_majority3"] = {
    j: {"kappas": {p: k for p, k, _ in v}, "yes_rates": {p: y for p, _, y in v},
        "min": min(k for _, k, _ in v), "max": max(k for _, k, _ in v),
        "range": max(k for _, k, _ in v) - min(k for _, k, _ in v),
        "sd": statistics.pstdev([k for _, k, _ in v]) if len(v) > 1 else 0.0,
        "yes_rate_range": [min(y for *_, y in v), max(y for *_, y in v)]}
    for j, v in by_judge.items()}
allk = [e["majority3"]["kappa"] for e in res["configs"].values()]
res["all_configs_kappa_vs_majority3"] = {"n_configs": len(allk), "min": min(allk), "max": max(allk),
                                         "mean": statistics.mean(allk), "sd": statistics.pstdev(allk)}

# Variance across judges per prompt
by_prompt = defaultdict(list)
for key, e in res["configs"].items():
    j, pid = key.split("|")
    by_prompt[pid].append(e["majority3"]["kappa"])
res["judge_sensitivity_per_prompt"] = {p: {"min": min(v), "max": max(v), "sd": statistics.pstdev(v)}
                                       for p, v in by_prompt.items()}

# Majority-of-new-judges ensembles per prompt (judges outside the original 3)
for pid in sorted(by_prompt):
    js = [j for j in ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini"] if (j, pid) in V]
    if len(js) == 3:
        pred = [int(sum(V[(j, pid)][s] for j in js) >= 2) for s in ids]
        k, ci = kappa_ci(pred, [maj[s] for s in ids])
        res["configs"][f"ens(4.1-mini,4.1,5-mini)|{pid}"] = {"n": 100, "yes_rate": sum(pred) / 100,
                                                            "majority3": {"kappa": k, "ci95": ci}}
save_json(res, "B04_judge_robustness.json")

print(f"total cost ${cost:.3f}; errors {dict(errors)}; unparsed {dict(unparsed)}")
print("repro", json.dumps(repro))
for key, e in res["configs"].items():
    mj = e["majority3"]
    print(f"{key:42s} yes={e['yes_rate']:.2f} k_maj={mj['kappa']:.3f} [{mj['ci95'][0]:.2f},{mj['ci95'][1]:.2f}] "
          + (f"B={e['bhaskar']['kappa']:.3f} Y={e['yoshitha']['kappa']:.3f} T={e['third']['kappa']:.3f} "
             f"FP={e['judge_yes_human_no']} FN={e['judge_no_human_yes']}" if "bhaskar" in e else ""))
print(json.dumps(res["prompt_sensitivity_vs_majority3"], indent=0))
print(res["all_configs_kappa_vs_majority3"])
