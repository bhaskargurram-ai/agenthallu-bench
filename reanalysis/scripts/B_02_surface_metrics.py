"""Critique 2: is the substring heuristic a strawman?

Implements, on the same 100 calibration traces:
  exact match, normalized match (equality + containment), token-F1>=0.5,
  numeric-tolerance match, reference-entity match, query-grounded slot checks
  (presence / +no-conflict / +non-answer filter), the existing heuristic (as
  coded in run_p2_experiment.py) and the heuristic as *described* in the paper.
All rules were fixed before computing agreement (no tuning on human labels).
Also quantifies how many references are task descriptions rather than gold values.
"""
import re
import string
import sys
import unicodedata
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa

p2 = load_p2()
TASKS = load_tasks()
ann = load_annotators()
ids = sample_ids()

from B_slot import *  # noqa  (gazetteers + metric functions)

rows = []
for s in ids:
    r = p2[s]
    ans_full = r["final_answer"] or ""
    gt = r["ground_truth"] or ""
    # 11/100 rows have an empty query in the CSV (judges and annotators saw it blank);
    # recover it from the task file (verified identical where both exist).
    query = r["query"] or TASKS.get(r["task_id"], {}).get("query", "")
    ents = extract_entities(query)
    al = clean(ans_full).lower()
    miss = [(t, e) for t, es in ents.items() for e in es if not present(al, t, e)]
    conf = conflicts(ans_full, ents)
    non_ans = bool(NON_ANSWER.search(clean(ans_full)))
    ref_ents = extract_entities(gt)
    ref_miss = [(t, e) for t, es in ref_ents.items() for e in es if not present(al, t, e)]
    nm, has_num = numeric_match(ans_full, gt)
    row = {
        "session_id": s, "model": r["model"], "domain": r["domain"], "source": r["source"],
        "query_used": query, "query_was_empty": not bool(r["query"]),
        "reference_kind": reference_kind(gt, query),
        "entities": ents, "n_entities": sum(len(v) for v in ents.values()),
        "missing": miss, "conflicts": conf, "non_answer": non_ans,
        "exact": int(clean(ans_full).strip().lower() == clean(gt).strip().lower()),
        "normalized_eq": int(normalize(ans_full) == normalize(gt)),
        "normalized_contain": int(bool(normalize(gt)) and normalize(gt) in normalize(ans_full)),
        "token_f1": token_f1(ans_full, gt),
        "token_f1_ge_0.5": int(token_f1(ans_full, gt) >= 0.5),
        "numeric_tol": nm, "ref_has_number": has_num,
        "ref_entity_match": int(bool(ref_ents) and not ref_miss),
        "slot_present": int(not miss),
        "slot_noconflict": int(not miss and not conf),
        "slot_strict": int(not miss and not conf and not non_ans),
        "non_answer_filter_only": int(not non_ans),
        "heuristic_csv": to01(r["heuristic_correct"]),
        "heuristic_recomputed": heuristic_code(ans_full, gt),
        "heuristic_paper_described": heuristic_paper_text(ans_full, gt),
        "gpt4o": to01(r["gpt4o_verdict"]), "gemini25flash": to01(r["gemini_25_flash_verdict"]),
        "gpt4o_mini": to01(r["gpt4o_mini_verdict"]), "ensemble": to01(r["ensemble_correct"]),
    }
    for a in ann:
        row["h_" + a] = ann[a][s]
    row["h_majority3"] = int(sum(ann[a][s] for a in ann) >= 2)
    row["h_unanimous"] = int(len({ann[a][s] for a in ann}) == 1)
    rows.append(row)

METRICS = ["exact", "normalized_eq", "normalized_contain", "token_f1_ge_0.5", "numeric_tol",
           "ref_entity_match", "heuristic_csv", "heuristic_paper_described",
           "slot_present", "slot_noconflict", "non_answer_filter_only", "slot_strict",
           "gpt4o", "gemini25flash", "gpt4o_mini", "ensemble"]
REFS = ["h_bhaskar", "h_yoshitha", "h_third", "h_majority3"]

res = {"n": len(rows), "metrics": {}}
for m in METRICS:
    pred = [r[m] for r in rows]
    entry = {"yes_rate": sum(pred) / len(pred)}
    for h in REFS:
        hum = [r[h] for r in rows]
        k, ci = kappa_ci(pred, hum)
        entry[h] = {"kappa": k, "ci95": ci, "agreement": raw_agree(pred, hum),
                    "fp_metric_yes_human_no": sum(p and not q for p, q in zip(pred, hum)),
                    "fn_metric_no_human_yes": sum((not p) and q for p, q in zip(pred, hum))}
    # unanimous subset vs consensus label
    sub = [r for r in rows if r["h_unanimous"]]
    k, ci = kappa_ci([r[m] for r in sub], [r["h_majority3"] for r in sub])
    entry["unanimous_subset"] = {"n": len(sub), "kappa": k, "ci95": ci}
    res["metrics"][m] = entry

# Paired bootstrap: slot_strict vs heuristic vs ensemble vs gpt4o_mini (vs majority3)
def paired_diff(m1, m2, h="h_majority3"):
    a = [r[m1] for r in rows]; b = [r[m2] for r in rows]; y = [r[h] for r in rows]
    d = cohen_kappa(a, y) - cohen_kappa(b, y)
    ci = bootstrap(lambda idx: cohen_kappa([a[i] for i in idx], [y[i] for i in idx]) -
                   cohen_kappa([b[i] for i in idx], [y[i] for i in idx]), len(rows))
    return {"diff": d, "ci95": ci}

res["paired_kappa_diffs_vs_majority3"] = {
    "slot_strict_minus_heuristic": paired_diff("slot_strict", "heuristic_csv"),
    "slot_strict_minus_ensemble": paired_diff("slot_strict", "ensemble"),
    "slot_strict_minus_gpt4o_mini": paired_diff("slot_strict", "gpt4o_mini"),
    "ensemble_minus_heuristic": paired_diff("ensemble", "heuristic_csv"),
    "gpt4o_mini_minus_ensemble": paired_diff("gpt4o_mini", "ensemble"),
}

# Reference-quality audit
from collections import Counter
res["reference_audit"] = {
    "kinds": dict(Counter(r["reference_kind"] for r in rows)),
    "ref_has_any_number": sum(r["ref_has_number"] for r in rows),
    "query_empty_in_csv": sum(r["query_was_empty"] for r in rows),
    "rows_with_no_query_entities": sum(r["n_entities"] == 0 for r in rows),
    "heuristic_csv_equals_recomputed": sum(r["heuristic_csv"] == r["heuristic_recomputed"] for r in rows),
    "possible_gold_value_examples": [ (p2[r["session_id"]]["ground_truth"]) for r in rows
                                      if r["reference_kind"] == "possible_gold_value"][:10],
}
# Reference kind over the whole P2 file (all 2150 rows), for context
allk = Counter()
for r in p2.values():
    gt = r["ground_truth"] or ""
    q = r["query"] or TASKS.get(r["task_id"], {}).get("query", "")
    allk[reference_kind(gt, q)] += 1
res["reference_audit"]["kinds_all_p2_rows"] = dict(allk)
res["reference_audit"]["distinct_references_in_sample"] = len({p2[s]["ground_truth"] for s in ids})

# Sensitivity: rule design involved inspecting the first 12 traces of the sheet
# (ids[:12]); report agreement on the 88 traces never inspected during design.
held = rows[12:]
res["heldout_88_uninspected"] = {}
for m in ["slot_strict", "slot_noconflict", "non_answer_filter_only", "heuristic_csv", "ensemble", "gpt4o_mini"]:
    k, ci = kappa_ci([r[m] for r in held], [r["h_majority3"] for r in held])
    res["heldout_88_uninspected"][m] = {"n": len(held), "kappa": k, "ci95": ci}
res["reference_audit"]["gold_value_examples_all_p2"] = sorted({
    r["ground_truth"] for r in p2.values()
    if reference_kind(r["ground_truth"] or "", r["query"] or TASKS.get(r["task_id"], {}).get("query", "")) == "possible_gold_value"})[:15]
save_json(res, "B02_surface_metrics.json")
with open(os.path.join(OUT, "B02_per_trace.csv"), "w", newline="") as f:
    cols = [k for k in rows[0].keys()]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})

print("reference audit:", json.dumps(res["reference_audit"], indent=1)[:1500])
for m in METRICS:
    e = res["metrics"][m]
    print(f"{m:28s} yes={e['yes_rate']:.2f} " + " ".join(
        f"{h[2:]}={e[h]['kappa']:.3f}[{e[h]['ci95'][0]:.2f},{e[h]['ci95'][1]:.2f}]" for h in REFS))
print(json.dumps(res["paired_kappa_diffs_vs_majority3"], indent=0))
