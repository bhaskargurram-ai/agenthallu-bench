"""Critique 1: precise inter-annotator agreement on the 100 calibration traces."""
import sys
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from itertools import combinations
from B_common import *  # noqa

ann = load_annotators()
ids = sample_ids()
assert all(set(ann[a]) == set(ids) for a in ann)
assert all(ann[a][s] is not None for a in ann for s in ids), "blank labels"

# Yoshitha file copies identical?
yc = {r["session_id"]: norm_label(r["YOUR_LABEL"]) for r in read_csv(YOSHITHA_COMPLETED)}
yoshitha_copies_identical = all(yc[s] == ann["yoshitha"][s] for s in ids)

L = {a: [ann[a][s] for s in ids] for a in ann}
res = {"n": len(ids), "yoshitha_completed_equals_labeled": yoshitha_copies_identical,
       "yes_rate": {a: sum(L[a]) / len(ids) for a in L}, "pairs": {}}

for a, b in combinations(["bhaskar", "yoshitha", "third"], 2):
    k, ci = kappa_ci(L[a], L[b])
    po = raw_agree(L[a], L[b])
    po_ci = bootstrap(lambda idx: raw_agree([L[a][i] for i in idx], [L[b][i] for i in idx]), len(ids))
    cm = {"both_yes": sum(x and y for x, y in zip(L[a], L[b])),
          f"{a}_yes_{b}_no": sum(x and not y for x, y in zip(L[a], L[b])),
          f"{a}_no_{b}_yes": sum((not x) and y for x, y in zip(L[a], L[b])),
          "both_no": sum((not x) and (not y) for x, y in zip(L[a], L[b]))}
    res["pairs"][f"{a}-{b}"] = {"kappa": k, "kappa_ci95": ci, "raw_agreement": po,
                                "raw_agreement_ci95": po_ci, "gwet_ac1": gwet_ac1(L[a], L[b]),
                                "confusion": cm,
                                "disagree_ids": [s for s, x, y in zip(ids, L[a], L[b]) if x != y]}

ratings = [[L[a][i] for a in ["bhaskar", "yoshitha", "third"]] for i in range(len(ids))]
fk = fleiss_kappa(ratings)
fk_ci = bootstrap(lambda idx: fleiss_kappa([ratings[i] for i in idx]), len(ids))
res["fleiss_kappa_3"] = {"kappa": fk, "ci95": fk_ci}
res["unanimous_items"] = sum(len(set(r)) == 1 for r in ratings)
res["majority_yes_rate"] = sum(sum(r) >= 2 for r in ratings) / len(ids)
# majority-of-3 vs each annotator (note: not independent - annotator is part of majority)
maj = [1 if sum(r) >= 2 else 0 for r in ratings]
res["each_vs_majority3_kappa_nonindependent"] = {a: cohen_kappa(L[a], maj) for a in L}
# leave-one-out: each annotator vs the other two's agreement set
loo = {}
for a in L:
    others = [o for o in L if o != a]
    keep = [i for i in range(len(ids)) if L[others[0]][i] == L[others[1]][i]]
    loo[a] = {"n": len(keep), "kappa": cohen_kappa([L[a][i] for i in keep], [L[others[0]][i] for i in keep]),
              "agreement": raw_agree([L[a][i] for i in keep], [L[others[0]][i] for i in keep])}
res["leave_one_out_vs_other_two_consensus"] = loo
# disagreement items by domain/model
p2 = load_p2()
dis = [ids[i] for i in range(len(ids)) if len(set(ratings[i])) > 1]
res["items_with_any_disagreement"] = len(dis)
res["disagreement_detail"] = [{"session_id": s, "model": p2[s]["model"], "domain": p2[s]["domain"],
                               "labels": {a: ann[a][s] for a in ann}} for s in dis]
save_json(res, "B01_interannotator.json")

for k, v in res["pairs"].items():
    print(f"{k:20s} kappa={v['kappa']:.3f} CI=[{v['kappa_ci95'][0]:.3f},{v['kappa_ci95'][1]:.3f}] "
          f"agree={v['raw_agreement']:.2f} AC1={v['gwet_ac1']:.3f} {v['confusion']}")
print("Fleiss3", round(fk, 3), [round(x, 3) for x in fk_ci], "unanimous", res["unanimous_items"],
      "yes rates", res["yes_rate"], "maj yes", res["majority_yes_rate"])
print("LOO", loo)
