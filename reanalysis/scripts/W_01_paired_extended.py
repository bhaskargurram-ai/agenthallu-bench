"""W_01: extensions of X_paired_validator.py for the JIIS manuscript.

Reproduces the X_paired_validator.py pairing exactly (same filters, same first-row-per-(task, model) rule,
same task-cluster bootstrap with B=2000 and seed 1, exact McNemar), then adds:
  (a) per-model results in the 'injected call executed clean' stratum;
  (b) the same 1,085 pairs scored by the three-LLM ensemble and by the substring heuristic
      (labels joined by session_id from the original rejudged CSVs) -> attenuation on identical pairs;
  (c) validator outcome codes for the P2 semantic-wrong arm of every proprietary model
      (first verifiable non-infra trace per (task, model), as in X), incl. 'fabricated' counts;
  (d) baseline vs injected outcome-code transitions on the paired set.
Run: python3 -I W_01_paired_extended.py   (reads only; writes ../W_writeup/W01_results.json and .txt)
"""
import csv, sys, json, random, collections
from scipy.stats import binomtest
csv.field_size_limit(sys.maxsize)

ROOT = "/Users/challagullayoshitha/Downloads/agentprop_revision"
REPO = "/Users/challagullayoshitha/Downloads/agentprop_repo/src/data/results/"
OUT = ROOT + "/W_writeup"

R = list(csv.DictReader(open(ROOT + "/V_validator/labels_all.csv")))
ok = lambda r: r["verifiable"].lower() in ("true", "1") and r["outcome"] != "infra_error"
B = {(r["task_id"], r["model"]): r for r in R if r["source"] == "baseline" and r["condition"] == "baseline" and ok(r)}
P = {}
for r in R:
    if r["condition"] in ("p2_semantic_wrong", "semantic_wrong") and ok(r):
        P.setdefault((r["task_id"], r["model"]), r)
keys = sorted(set(B) & set(P))

# old labels by session id
old = {}
for fn in ("p2_all_rejudged.csv", "baseline_all_rejudged.csv"):
    for r in csv.DictReader(open(REPO + fn)):
        old[r["session_id"]] = r
tf = lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")

def wrong_fn(judge):
    if judge == "validator":
        return lambda r: str(r["CORRECT"]).lower() not in ("true", "1")
    col = {"ensemble": "ensemble_correct", "heuristic": "heuristic_correct", "gpt4o_mini": "gpt4o_mini_verdict"}[judge]
    return lambda r: not tf(old[r["session_id"]][col])

def stat(ks, judge="validator"):
    w = wrong_fn(judge)
    b = sum(1 for k in ks if w(P[k]) and not w(B[k])); c = sum(1 for k in ks if not w(P[k]) and w(B[k]))
    n = len(ks)
    if n == 0:
        return None
    d = (b - c) / n
    tasks = sorted(set(k[0] for k in ks)); by = collections.defaultdict(list)
    for k in ks: by[k[0]].append(k)
    random.seed(1); bs = []
    for _ in range(2000):
        s = [k for t in random.choices(tasks, k=len(tasks)) for k in by[t]]
        bb = sum(1 for k in s if w(P[k]) and not w(B[k])); cc = sum(1 for k in s if not w(P[k]) and w(B[k]))
        bs.append((bb - cc) / len(s))
    bs.sort(); p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(n=n, err_base=100 * sum(w(B[k]) for k in ks) / n, err_inj=100 * sum(w(P[k]) for k in ks) / n,
                rd=100 * d, ci=[100 * bs[49], 100 * bs[1949]], b=b, c=c, p=p)

ex = [k for k in keys if int(P[k]["n_exec"] or 0) > 0 and int(P[k]["n_schema_err"] or 0) == 0]
nt = [k for k in keys if int(P[k]["n_exec"] or 0) == 0 and int(B[k]["n_exec"] or 0) == 0]
models = sorted(set(k[1] for k in keys))
res = {"n_pairs": len(keys), "n_old_label_join": sum(1 for k in keys if P[k]["session_id"] in old and B[k]["session_id"] in old)}
for judge in ("validator", "ensemble", "heuristic", "gpt4o_mini"):
    res[judge] = {"all": stat(keys, judge), "executed_clean": stat(ex, judge), "placebo": stat(nt, judge),
                  "per_model": {m: stat([k for k in keys if k[1] == m], judge) for m in models},
                  "per_model_executed_clean": {m: stat([k for k in ex if k[1] == m], judge) for m in models}}

# stratum sizes per model
res["stratum_sizes"] = {m: {"all": sum(1 for k in keys if k[1] == m), "executed_clean": sum(1 for k in ex if k[1] == m),
                            "placebo": sum(1 for k in nt if k[1] == m),
                            "p2_no_exec": sum(1 for k in keys if k[1] == m and int(P[k]["n_exec"] or 0) == 0),
                            "p2_schema_err_any": sum(1 for k in keys if k[1] == m and int(P[k]["n_schema_err"] or 0) > 0)}
                        for m in models}

# (c) outcome codes of the P2 semantic-wrong arm, all proprietary models (first verifiable non-infra trace per pair key)
oc = collections.defaultdict(collections.Counter)
for (t, m), r in P.items():
    if r["source"] == "p2":
        oc[m][r["outcome"]] += 1
res["p2_outcomes_by_model"] = {m: dict(v) | {"n": sum(v.values())} for m, v in sorted(oc.items())}
ob = collections.defaultdict(collections.Counter)
for (t, m), r in B.items():
    ob[m][r["outcome"]] += 1
res["baseline_outcomes_by_model"] = {m: dict(v) | {"n": sum(v.values())} for m, v in sorted(ob.items())}

# (d) transitions on paired set
tr = collections.Counter((B[k]["outcome"], P[k]["outcome"]) for k in keys)
res["transitions_paired"] = {f"{a}->{b}": n for (a, b), n in tr.most_common()}
# outcome among pairs that flip correct -> wrong
flip = collections.Counter(P[k]["outcome"] for k in keys if B[k]["outcome"] == "correct" and P[k]["outcome"] != "correct")
res["flip_correct_to_wrong_outcomes"] = dict(flip)

json.dump(res, open(OUT + "/W01_results.json", "w"), indent=1, default=float)
with open(OUT + "/W01_results.txt", "w") as f:
    def line(lab, s):
        if s is None: return f"{lab:34s} n=0\n"
        return (f"{lab:34s} n={s['n']:5d} err_base={s['err_base']:5.1f} err_inj={s['err_inj']:5.1f} "
                f"RD={s['rd']:+5.1f}pp CI[{s['ci'][0]:+.1f},{s['ci'][1]:+.1f}] disc {s['b']}/{s['c']} p={s['p']:.2g}\n")
    f.write(f"pairs {res['n_pairs']}; old-label join {res['n_old_label_join']}\n")
    for judge in ("validator", "ensemble", "heuristic", "gpt4o_mini"):
        f.write(f"\n== {judge} ==\n")
        f.write(line("ALL", res[judge]["all"])); f.write(line("executed clean", res[judge]["executed_clean"]))
        f.write(line("placebo", res[judge]["placebo"]))
        for m in models:
            f.write(line("  " + m, res[judge]["per_model"][m]))
        for m in models:
            f.write(line("  clean " + m, res[judge]["per_model_executed_clean"][m]))
    f.write("\nstratum sizes " + json.dumps(res["stratum_sizes"]) + "\n")
    f.write("\nP2 outcomes " + json.dumps(res["p2_outcomes_by_model"], indent=0) + "\n")
    f.write("\nbaseline outcomes " + json.dumps(res["baseline_outcomes_by_model"], indent=0) + "\n")
    f.write("\nflip outcomes " + json.dumps(res["flip_correct_to_wrong_outcomes"]) + "\n")
    f.write("\ntransitions " + json.dumps(res["transitions_paired"], indent=0) + "\n")
print(open(OUT + "/W01_results.txt").read())
