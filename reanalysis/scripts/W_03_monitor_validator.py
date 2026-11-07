import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""W_03: the runtime-monitor experiment re-scored with the structured validator (proprietary models with traces).
Pairs the interceptor arm (early_interceptor) with the task-matched control arm (no_interceptor) on verifiable tasks
(first trace per task, infra errors excluded). Abstention = interceptor-arm outcome 'refused'.
Reports: accuracy both arms, abstentions, abstention precision (share of abstained tasks the control got wrong),
base error rate, one-sided hypergeometric p that random abstention of the same size would catch >= as many errors.
Run: python3 -I W_03_monitor_validator.py  -> ../W_writeup/W03_monitor_validator.json
"""
import csv, sys, json
from scipy.stats import hypergeom
csv.field_size_limit(sys.maxsize)
ROOT = _REPO + "/reanalysis"
R = list(csv.DictReader(open(ROOT + "/V_validator/labels_all.csv")))
ok = lambda r: r["verifiable"].lower() in ("true", "1") and r["outcome"] != "infra_error"
out = {}
for m in ("gpt4o_mini", "gemini_20_flash"):
    C, I = {}, {}
    for r in R:
        if r["model"] != m or not ok(r): continue
        if r["source"] == "interceptor_control": C.setdefault(r["task_id"], r)
        if r["source"] == "interceptor" and r["condition"] == "interceptor_early_interceptor": I.setdefault(r["task_id"], r)
    ks = sorted(set(C) & set(I)); n = len(ks)
    wrong_c = [k for k in ks if C[k]["outcome"] != "correct"]
    ab = [k for k in ks if I[k]["outcome"] == "refused"]
    caught = sum(1 for k in ab if C[k]["outcome"] != "correct")
    ans = [k for k in ks if I[k]["outcome"] != "refused"]
    p = float(hypergeom.sf(caught - 1, n, len(wrong_c), len(ab)))
    out[m] = dict(n=n, control_correct=sum(C[k]["outcome"] == "correct" for k in ks),
                  interceptor_correct=sum(I[k]["outcome"] == "correct" for k in ks),
                  control_refused=sum(C[k]["outcome"] == "refused" for k in ks), abstained=len(ab),
                  abstained_control_wrong=caught, abstained_control_correct=len(ab) - caught,
                  precision=caught / len(ab) if ab else None, base_error=len(wrong_c) / n,
                  expected_caught_random=len(ab) * len(wrong_c) / n, p_better_than_random=p,
                  answered=len(ans), answered_int_correct=sum(I[k]["outcome"] == "correct" for k in ans),
                  answered_ctrl_correct=sum(C[k]["outcome"] == "correct" for k in ans))
json.dump(out, open(ROOT + "/W_writeup/W03_monitor_validator.json", "w"), indent=1)
print(json.dumps(out, indent=1))
