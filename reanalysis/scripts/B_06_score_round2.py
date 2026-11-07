import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""Score the round-2 human labels and compute calibration factors.

Usage:
  python3 -I B_06_score_round2.py --sheets sheetA.csv [sheetB.csv ...] [--extra-verdicts file.jsonl]

Inputs
  --sheets           one or more filled copies of labeling_sheet_round2_BLANK.csv (YOUR_LABEL = YES/NO)
  --key              PRIVATE_key_round2.csv (default: next to this round's sheet)
  --extra-verdicts   optional JSONL {"session_id","judge","prompt","verdict"} (e.g. B03-style runs on the
                     round-2 traces) to evaluate additional judges.
Outputs (B_judge/labeling_round2/):
  round2_scores.json, round2_scores.md

Computed:
  * per-annotator completion / YES-rate; pairwise Cohen kappa + bootstrap CI; Fleiss kappa (>=3 sheets)
  * human reference label = majority over annotators (2 annotators: items where they disagree are
    reported, and analyses are run on (a) agreed items and (b) annotator-1 labels)
  * ANCHOR drift: agreement of the new labels with round-1 3-annotator majority
  * per stratum (OW_S2, OW_LO, PR_S2, OW=OW_S2+OW_LO, NEW=all non-anchor) and per open-weight model:
      kappa (+CI) of gpt4o, gemini-2.5-flash, gpt4o-mini, ensemble, substring heuristic, slot_strict
      vs human; judge-minus-human YES-rate (bias direction)
  * calibration factors c1 = P(human YES | ensemble YES), c0 = P(human YES | ensemble NO)
      - unweighted with Wilson 95% CI, and sampling-weighted (population-level) point estimates
      - by family (open-weight vs proprietary) restricted to S2=1, which is what r23 calibration uses
      - bootstrap CI of the open-minus-proprietary difference (transfer test)
      - comparison with round-1 factors (computed here from the round-1 files, not hard-coded)
  * calibrated r23 per open-weight model using family-specific S2 factors, vs. round-1 factors
"""
import argparse
import math
import sys
from collections import defaultdict
from itertools import combinations
sys.path.insert(0, _REPO + "/reanalysis/scripts")
from B_common import *  # noqa
from B_slot import slot_verdicts

L2 = os.path.join(OUT, "labeling_round2")
OPEN = ["qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def calib_factors(items, lab="human", w=None):
    """c1 = P(human YES | ens YES), c0 = P(human YES | ens NO)."""
    out = {}
    for name, ev in [("c1_P_humanYES_given_ensYES", 1), ("c0_P_humanYES_given_ensNO", 0)]:
        sub = [it for it in items if it["ensemble"] == ev]
        k = sum(it[lab] for it in sub)
        n = len(sub)
        e = {"k": k, "n": n, "p": (k / n if n else float("nan")), "wilson95": wilson(k, n)}
        if w:
            W = sum(it[w] for it in sub)
            e["p_weighted"] = (sum(it[w] * it[lab] for it in sub) / W) if W else float("nan")
        out[name] = e
    return out


def round1_factors():
    ann = load_annotators()
    p2 = load_p2()
    ids = sample_ids()
    res = {}
    for ref in ["bhaskar", "majority3"]:
        items = []
        for s in ids:
            h = ann["bhaskar"][s] if ref == "bhaskar" else int(sum(ann[a][s] for a in ann) >= 2)
            items.append({"human": h, "ensemble": to01(p2[s]["ensemble_correct"]), "S2": int(int(p2[s]["eps"]) >= 2)})
        res[ref] = {"all": calib_factors(items), "S2_only": calib_factors([i for i in items if i["S2"]])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--key", default=os.path.join(L2, "PRIVATE_key_round2.csv"))
    ap.add_argument("--extra-verdicts", default=None)
    args = ap.parse_args()

    key = {r["item_id"]: r for r in read_csv(args.key)}
    p2 = load_p2()
    tasks = load_tasks()
    sheets = {}
    for path in args.sheets:
        name = os.path.splitext(os.path.basename(path))[0].replace("labeling_sheet_round2_", "")
        sheets[name] = {r["item_id"]: norm_label(r.get("YOUR_LABEL")) for r in read_csv(path)}
    names = list(sheets)
    report = {"annotators": names, "per_annotator": {}, "pairs": {}}
    for n_ in names:
        lab = sheets[n_]
        filled = [i for i in key if lab.get(i) is not None]
        report["per_annotator"][n_] = {"filled": len(filled), "of": len(key),
                                       "yes_rate": (sum(lab[i] for i in filled) / len(filled)) if filled else None}

    # inter-annotator
    for a, b in combinations(names, 2):
        common = [i for i in key if sheets[a].get(i) is not None and sheets[b].get(i) is not None]
        if len(common) >= 5:
            k, ci = kappa_ci([sheets[a][i] for i in common], [sheets[b][i] for i in common])
            report["pairs"][f"{a}-{b}"] = {"n": len(common), "kappa": k, "ci95": ci,
                                           "agreement": raw_agree([sheets[a][i] for i in common],
                                                                  [sheets[b][i] for i in common])}
    if len(names) >= 3:
        common = [i for i in key if all(sheets[n_].get(i) is not None for n_ in names)]
        ratings = [[sheets[n_][i] for n_ in names] for i in common]
        report["fleiss"] = {"n": len(common), "kappa": fleiss_kappa(ratings),
                            "ci95": bootstrap(lambda idx: fleiss_kappa([ratings[j] for j in idx]), len(common))}

    # consensus label
    items = []
    n_tied = 0
    for i, k in key.items():
        labs = [sheets[n_][i] for n_ in names if sheets[n_].get(i) is not None]
        if not labs:
            continue
        y = sum(labs)
        if len(labs) % 2 == 0 and y * 2 == len(labs):
            n_tied += 1
            human = None
        else:
            human = int(y * 2 > len(labs))
        s = k["session_id"]
        r = p2[s]
        q = r["query"] or tasks.get(r["task_id"], {}).get("query", "")
        sv = slot_verdicts(q, r["final_answer"] or "")
        items.append({"item_id": i, "session_id": s, "model": k["model"], "family": k["family"],
                      "stratum": k["stratum"], "S2": int(k["S2"]),
                      "w": float(k["sampling_weight"]) if k["sampling_weight"] else 1.0,
                      "human": human, "human_first": labs[0], "orig_labels": k["orig_labels"],
                      "gpt4o": to01(r["gpt4o_verdict"]), "gemini25flash": to01(r["gemini_25_flash_verdict"]),
                      "gpt4o_mini": to01(r["gpt4o_mini_verdict"]), "ensemble": to01(r["ensemble_correct"]),
                      "heuristic": to01(r["heuristic_correct"]), "slot_strict": sv["slot_strict"]})
    report["n_labeled_items"] = len(items)
    report["n_tied_between_annotators"] = n_tied
    for it in items:
        if it["human"] is None:
            it["human"] = it["human_first"]  # fallback for ties (2 annotators); flagged above
    judges = ["gpt4o", "gemini25flash", "gpt4o_mini", "ensemble", "heuristic", "slot_strict"]

    if args.extra_verdicts:
        ev = defaultdict(dict)
        with open(args.extra_verdicts) as f:
            for line in f:
                d = json.loads(line)
                if d.get("verdict") is not None:
                    ev[f"{d['judge']}|{d['prompt']}"][d["session_id"]] = d["verdict"]
        for j, m in ev.items():
            if all(it["session_id"] in m for it in items):
                for it in items:
                    it[j] = m[it["session_id"]]
                judges.append(j)

    # anchors (drift)
    anc = [it for it in items if it["stratum"] == "ANCHOR"]
    if anc:
        orig = [int(sum(json.loads(it["orig_labels"]).values()) >= 2) for it in anc]
        report["anchor_drift"] = {"n": len(anc), "agreement_with_round1_majority":
                                  raw_agree([it["human"] for it in anc], orig)}

    def judge_block(sub):
        out = {"n": len(sub), "human_yes_rate": sum(i["human"] for i in sub) / len(sub) if sub else None}
        for j in judges:
            if len(sub) < 5:
                continue
            k, ci = kappa_ci([i[j] for i in sub], [i["human"] for i in sub])
            out[j] = {"kappa": k, "ci95": ci, "agreement": raw_agree([i[j] for i in sub], [i["human"] for i in sub]),
                      "judge_minus_human_yes_rate": (sum(i[j] for i in sub) - sum(i["human"] for i in sub)) / len(sub)}
        return out

    new = [it for it in items if it["stratum"] != "ANCHOR"]
    strata = {"NEW_all": new,
              "OW_all": [i for i in new if i["family"] == "open_weight"],
              "OW_S2": [i for i in new if i["stratum"] == "OW_S2"],
              "OW_LO": [i for i in new if i["stratum"] == "OW_LO"],
              "PR_S2": [i for i in new if i["stratum"] == "PR_S2"],
              "S2_all": [i for i in new if i["S2"]]}
    for m in OPEN:
        strata["model_" + m] = [i for i in new if i["model"] == m]
    report["judge_agreement_by_stratum"] = {k: judge_block(v) for k, v in strata.items()}

    # calibration factors
    report["calibration_round1"] = round1_factors()
    report["calibration_round2"] = {k: calib_factors(v, w="w") for k, v in strata.items()
                                    if not k.startswith("model_")}

    def diff_fn(A, B, which):
        def f(idxA, idxB):
            a = [A[i] for i in idxA if A[i]["ensemble"] == which]
            b = [B[i] for i in idxB if B[i]["ensemble"] == which]
            if not a or not b:
                return float("nan")
            return sum(x["human"] for x in a) / len(a) - sum(x["human"] for x in b) / len(b)
        return f

    rng = random.Random(SEED)
    A, B = strata["OW_S2"], strata["PR_S2"]
    transfer = {}
    for which, nm in [(1, "c1_diff_open_minus_prop_S2"), (0, "c0_diff_open_minus_prop_S2")]:
        fn = diff_fn(A, B, which)
        point = fn(range(len(A)), range(len(B)))
        vals = []
        for _ in range(N_BOOT):
            v = fn([rng.randrange(len(A)) for _ in A], [rng.randrange(len(B)) for _ in B]) if A and B else float("nan")
            if v == v:
                vals.append(v)
        vals.sort()
        ci = (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]) if vals else (None, None)
        transfer[nm] = {"diff": point, "ci95": ci}
    report["calibration_transfer_test"] = transfer

    # calibrated r23 per open-weight model (all S2=1 traces of that model in p2_all_rejudged.csv)
    f_r1 = report["calibration_round1"]["majority3"]["all"]
    f_ow = report["calibration_round2"]["OW_S2"]
    cal = {}
    for m in OPEN:
        s2 = [r for r in p2.values() if r["model"] == m and int(r["eps"] or 0) >= 2]
        n = len(s2)
        ey = sum(to01(r["ensemble_correct"]) for r in s2)
        en = n - ey
        def r23(c1, c0):
            if n == 0 or c1 != c1 or c0 != c0:
                return None
            return 1 - (ey * c1 + en * c0) / n
        cal[m] = {"n_S2": n, "r23_raw_ensemble": (en / n if n else None),
                  "r23_cal_round1_factors": r23(f_r1["c1_P_humanYES_given_ensYES"]["p"], f_r1["c0_P_humanYES_given_ensNO"]["p"]),
                  "r23_cal_round2_OW_S2_factors": r23(f_ow["c1_P_humanYES_given_ensYES"]["p"], f_ow["c0_P_humanYES_given_ensNO"]["p"])}
    report["r23_open_weight"] = cal

    save = os.path.join(L2, "round2_scores.json")
    with open(save, "w") as f:
        json.dump(report, f, indent=2, default=str)
    lines = ["# Round-2 scoring summary", "", f"annotators: {names}; labeled items: {len(items)}; ties: {n_tied}", ""]
    for k, v in report["pairs"].items():
        lines.append(f"- inter-annotator {k}: kappa={v['kappa']:.3f} CI=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] (n={v['n']})")
    lines += ["", "| stratum | n | human YES | " + " | ".join(judges) + " |", "|---|---|---|" + "---|" * len(judges)]
    for s, b in report["judge_agreement_by_stratum"].items():
        if not b.get("n"):
            continue
        cells = [f"{b[j]['kappa']:.2f} [{b[j]['ci95'][0]:.2f},{b[j]['ci95'][1]:.2f}]" if j in b else "–" for j in judges]
        lines.append(f"| {s} | {b['n']} | {b['human_yes_rate']:.2f} | " + " | ".join(cells) + " |")
    lines += ["", "## Calibration factors (P(human YES | ensemble verdict))", ""]
    for s, b in report["calibration_round2"].items():
        c1, c0 = b["c1_P_humanYES_given_ensYES"], b["c0_P_humanYES_given_ensNO"]
        lines.append(f"- {s}: c1={c1['p']:.2f} ({c1['k']}/{c1['n']}), c0={c0['p']:.2f} ({c0['k']}/{c0['n']})")
    r1 = report["calibration_round1"]["majority3"]["all"]
    lines.append(f"- round-1 (majority3, all 100): c1={r1['c1_P_humanYES_given_ensYES']['p']:.2f}, "
                 f"c0={r1['c0_P_humanYES_given_ensNO']['p']:.2f}")
    lines += ["", "transfer test (open-weight S2 minus proprietary S2): " + json.dumps(transfer, default=str), "",
              "r23 per open-weight model: " + json.dumps(cal, indent=1, default=str)]
    with open(os.path.join(L2, "round2_scores.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
