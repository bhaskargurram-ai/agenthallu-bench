"""Critique 5 (provisional, no new human labels): does the ensemble's error profile differ on
open-weight vs proprietary traces?

Reference judges (neither is a member of the ensemble):
  * strong LLM judge  = gpt-4.1 with the P3 reason-then-verdict prompt (best/equal-best non-ensemble
    config vs the 3-annotator majority on the 100 calibration traces; NOT validated out of sample)
  * slot_strict       = query-grounded rule-based check (B_slot.py), computed on every trace
Sample for the LLM judge: all S2=1 (EPS>=2) traces + 150 random EPS<2 traces per family,
excluding the 100 calibration traces. slot_strict is computed on all 2150 rows.
Reported per family x S2: ensemble YES-rate, reference YES-rate, kappa(ensemble, reference) with CI,
and the disagreement direction (ensemble NO & ref YES = conservative; ensemble YES & ref NO = lenient).
"""
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa
from B_slot import slot_verdicts
import B_03_rejudge as R

OPEN = {"qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b"}
RAW = os.path.join(OUT, "B08_strongjudge_raw.jsonl")
JUDGE, PID = "gpt-4.1", "P3_reason_then_verdict"

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="store_true")
ap.add_argument("--budget", type=float, default=1.5)
args = ap.parse_args()

p2 = load_p2()
tasks = load_tasks()
calib = set(sample_ids())
rng = random.Random(SEED)
rows = [r for r in p2.values() if r["session_id"] not in calib]
fam = lambda r: "open_weight" if r["model"] in OPEN else "proprietary"
s2 = lambda r: int(int(r["eps"] or 0) >= 2)
sample = [r for r in rows if s2(r)]
for f in ["open_weight", "proprietary"]:
    pool = [r for r in rows if fam(r) == f and not s2(r)]
    rng.shuffle(pool)
    sample += pool[:150]

done = {}
if os.path.exists(RAW):
    with open(RAW) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("verdict") is not None:
                done[d["session_id"]] = d["verdict"]

if args.run:
    from openai import OpenAI
    client = OpenAI(api_key=R.get_key(), timeout=60, max_retries=0)
    lock = threading.Lock()
    spent = [0.0]
    todo = [r for r in sample if r["session_id"] not in done]
    print("calls to make:", len(todo))

    def work(r):
        with lock:
            if spent[0] > args.budget:
                return
        prompt = R.PROMPTS[PID].format(query=str(r["query"])[:400], reference=str(r["ground_truth"])[:300],
                                       candidate=str(r["final_answer"])[:300])
        try:
            text, cost, ti, to = R.call(client, JUDGE, prompt, PID)
            rec = {"session_id": r["session_id"], "judge": JUDGE, "prompt": PID, "raw": text,
                   "verdict": R.parse(text, PID), "cost": cost}
        except Exception as e:
            rec = {"session_id": r["session_id"], "judge": JUDGE, "prompt": PID, "verdict": None, "cost": 0.0,
                   "error": str(e)[:200]}
        with lock:
            spent[0] += rec["cost"]
            with open(RAW, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(as_completed([pool.submit(work, r) for r in todo]))
    print(f"spent ${spent[0]:.3f}")
    with open(RAW) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("verdict") is not None:
                done[d["session_id"]] = d["verdict"]

cost_total = 0.0
if os.path.exists(RAW):
    with open(RAW) as fh:
        cost_total = sum(json.loads(l).get("cost", 0.0) for l in fh)


def block(items, refkey):
    e = [i["ens"] for i in items]
    f = [i[refkey] for i in items]
    k, ci = kappa_ci(e, f)
    return {"n": len(items), "ens_yes": sum(e) / len(e), "ref_yes": sum(f) / len(f), "kappa": k, "ci95": ci,
            "ens_NO_ref_YES(conservative)": sum((not a) and b for a, b in zip(e, f)) / len(e),
            "ens_YES_ref_NO(lenient)": sum(a and not b for a, b in zip(e, f)) / len(e),
            "P_refYES_given_ensYES": (sum(b for a, b in zip(e, f) if a) / max(1, sum(e))),
            "P_refYES_given_ensNO": (sum(b for a, b in zip(e, f) if not a) / max(1, len(e) - sum(e)))}


items_all = []
for r in p2.values():
    q = r["query"] or tasks.get(r["task_id"], {}).get("query", "")
    items_all.append({"sid": r["session_id"], "model": r["model"], "fam": fam(r), "S2": s2(r),
                      "ens": to01(r["ensemble_correct"]), "slot": slot_verdicts(q, r["final_answer"] or "")["slot_strict"],
                      "strong": done.get(r["session_id"])})

res = {"strong_judge": f"{JUDGE}|{PID}", "strong_judge_cost_usd": cost_total, "vs_slot_strict_all2150": {},
       "vs_strong_judge_sample": {}}
for f in ["open_weight", "proprietary"]:
    for sv, nm in [(None, "all"), (1, "S2"), (0, "nonS2")]:
        sub = [i for i in items_all if i["fam"] == f and (sv is None or i["S2"] == sv)]
        res["vs_slot_strict_all2150"][f"{f}|{nm}"] = block(sub, "slot")
        sub2 = [i for i in sub if i["strong"] is not None and i["sid"] not in calib]
        if len(sub2) >= 10:
            res["vs_strong_judge_sample"][f"{f}|{nm}"] = block(sub2, "strong")
for m in sorted({i["model"] for i in items_all}):
    sub = [i for i in items_all if i["model"] == m and i["strong"] is not None and i["sid"] not in calib]
    if len(sub) >= 10:
        res["vs_strong_judge_sample"][f"model:{m}"] = block(sub, "strong")


# difference in conservative-disagreement rate, open - proprietary, on S2 (bootstrap)
def diff_ci(key):
    A = [i for i in items_all if i["fam"] == "open_weight" and i["S2"] and i[key] is not None and i["sid"] not in calib]
    B = [i for i in items_all if i["fam"] == "proprietary" and i["S2"] and i[key] is not None and i["sid"] not in calib]
    fA = lambda xs: sum(i[key] for i in xs if not i["ens"]) / max(1, sum(1 for i in xs if not i["ens"]))
    point = fA(A) - fA(B)
    vals = []
    for _ in range(4000):
        a = [A[rng.randrange(len(A))] for _ in A]
        b = [B[rng.randrange(len(B))] for _ in B]
        vals.append(fA(a) - fA(b))
    vals.sort()
    return {"P_refYES_given_ensNO_open_minus_prop_S2": point, "ci95": (vals[100], vals[3899]),
            "n_open": len(A), "n_prop": len(B)}


res["transfer_S2_vs_slot"] = diff_ci("slot")
if done:
    res["transfer_S2_vs_strong"] = diff_ci("strong")
save_json(res, "B08_transfer_provisional.json")
for sec in ["vs_slot_strict_all2150", "vs_strong_judge_sample"]:
    print("==", sec)
    for k, v in res[sec].items():
        print(f"{k:28s} n={v['n']:4d} ens_yes={v['ens_yes']:.2f} ref_yes={v['ref_yes']:.2f} "
              f"k={v['kappa']:.3f}[{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] cons={v['ens_NO_ref_YES(conservative)']:.2f} "
              f"len={v['ens_YES_ref_NO(lenient)']:.2f} P(refY|ensN)={v['P_refYES_given_ensNO']:.2f} P(refY|ensY)={v['P_refYES_given_ensYES']:.2f}")
print(res.get("transfer_S2_vs_slot"), res.get("transfer_S2_vs_strong"), "cost", cost_total)
