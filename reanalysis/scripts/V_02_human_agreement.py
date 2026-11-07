import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""V_02: agreement of the structured validator with the 100-trace human labels (3 annotators, majority vote).

Humans saw only query + descriptive reference + final answer.  The validator sees gold observations.
Outputs (V_validator/): human_agreement.json, human_disagreements.csv (every disagreement with gold evidence).
Usage: python3 -I V_02_human_agreement.py [cfg_json]
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_core as V  # noqa: E402
import V_traces as L  # noqa: E402

OUT = _REPO + "/reanalysis/V_validator"
RES = L.RESULTS
SHEETS = {
    "bhaskar": f"{RES}/human_labeling_sheet_Bhaskar_labeled.csv",
    "yoshitha": f"{RES}/human_labeling_sheet_Yoshitha_labeled.csv",
    "third": _REPO + "/data/results/human_labels_third_annotator.csv",
}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot_ci(a, b, f, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        aa, bb = [a[i] for i in idx], [b[i] for i in idx]
        try:
            vals.append(f(aa, bb))
        except ZeroDivisionError:
            continue
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def load_labels():
    lab = {}
    for name, p in SHEETS.items():
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                lab.setdefault(r["session_id"], {})[name] = r["YOUR_LABEL"].strip().upper() == "YES"
    return lab


def score_human_set(cfg=None):
    lab = load_labels()
    tasks = L.load_tasks()
    stats = {}
    recs = [r for r in L.load_db("p2_day6", "p2", stats) if r["session_id"] in lab]
    assert len(recs) == len(lab) == 100, (len(recs), len(lab))
    rows = []
    for r in recs:
        t = tasks[r["task_id"]]
        s = V.score_trace(t, r["calls"], r["final_answer"], cfg=cfg)
        h = lab[r["session_id"]]
        maj = sum(h.values()) >= 2
        rows.append({"r": r, "s": s, "h": h, "maj": maj, "task": t})
    return rows


def main():
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = score_human_set(cfg)
    ver = [x for x in rows if x["s"]["verifiable"] and x["s"]["outcome"] != "infra_error"]
    out = {"n_total": len(rows), "n_verifiable": len(ver),
           "n_not_verifiable": sum(1 for x in rows if not x["s"]["verifiable"]),
           "not_verifiable_templates": sorted({x["s"]["template"] for x in rows if not x["s"]["verifiable"]}),
           "n_infra_error": sum(1 for x in rows if x["s"]["outcome"] == "infra_error")}
    v = [int(bool(x["s"]["CORRECT"])) for x in ver]
    m = [int(x["maj"]) for x in ver]
    out["validator_yes_rate"] = sum(v) / len(v)
    out["human_majority_yes_rate"] = sum(m) / len(m)
    out["kappa_vs_majority"] = kappa(v, m)
    out["kappa_vs_majority_ci95"] = boot_ci(v, m, kappa)
    out["raw_agreement_vs_majority"] = sum(a == b for a, b in zip(v, m)) / len(v)
    out["raw_agreement_ci95"] = boot_ci(v, m, lambda a, b: sum(x == y for x, y in zip(a, b)) / len(a))
    out["confusion_vs_majority"] = {"both_yes": sum(a and b for a, b in zip(v, m)),
                                    "validator_yes_human_no": sum(a and not b for a, b in zip(v, m)),
                                    "validator_no_human_yes": sum((not a) and b for a, b in zip(v, m)),
                                    "both_no": sum((not a) and (not b) for a, b in zip(v, m))}
    for name in SHEETS:
        hh = [int(x["h"][name]) for x in ver]
        out[f"kappa_vs_{name}"] = kappa(v, hh)
        out[f"kappa_vs_{name}_ci95"] = boot_ci(v, hh, kappa)
    # human-human reference on the same verifiable subset
    names = list(SHEETS)
    for i in range(3):
        for j in range(i + 1, 3):
            a = [int(x["h"][names[i]]) for x in ver]
            b = [int(x["h"][names[j]]) for x in ver]
            out[f"kappa_human_{names[i]}_{names[j]}"] = kappa(a, b)
    # also on all 100 (not-verifiable counted as validator NO) for transparency
    v_all = [int(bool(x["s"]["CORRECT"])) for x in rows]
    m_all = [int(x["maj"]) for x in rows]
    out["kappa_vs_majority_all100_notverifiable_as_NO"] = kappa(v_all, m_all)
    out["outcomes_on_verifiable"] = {}
    for x in ver:
        out["outcomes_on_verifiable"][x["s"]["outcome"]] = out["outcomes_on_verifiable"].get(x["s"]["outcome"], 0) + 1
    tag = "" if cfg is None else "_cfg"
    if cfg is None:
        with open(f"{OUT}/human_disagreements.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["session_id", "template", "query", "final_answer", "human_bhaskar", "human_yoshitha",
                        "human_third", "human_majority", "validator_CORRECT", "outcome", "T", "P", "S", "A",
                        "A_key_fail", "ungrounded", "claim_detail", "executed_calls", "gold_observation_summary"])
            for x in rows:
                s = x["s"]
                if not s["verifiable"] or bool(s["CORRECT"]) == x["maj"]:
                    continue
                calls = " || ".join(f"{c['tool']}({json.dumps(c['params'])})->{json.dumps(c['result'])[:200]}"
                                    for c in x["r"]["calls"])
                spec = V.build_spec(x["task"])
                gsum = []
                for g in spec.calls:
                    if g.tool in V.STATELESS:
                        gsum.append(f"{g.tool}({json.dumps(g.params)})->{json.dumps(V.sim(g.tool, g.params))[:250]}")
                    else:
                        gsum.append(f"{g.tool}({json.dumps(g.params)})->[store-dependent: received obs used]")
                w.writerow([x["r"]["session_id"], s["template"], x["task"]["query"], x["r"]["final_answer"],
                            x["h"]["bhaskar"], x["h"]["yoshitha"], x["h"]["third"], x["maj"], s["CORRECT"],
                            s["outcome"], s["T"], s["P"], s["S"], s["A"], s.get("A_key_fail"), s.get("ungrounded"),
                            s.get("claim_detail"), calls, " || ".join(gsum)])
        with open(f"{OUT}/human_agreement.json", "w") as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
