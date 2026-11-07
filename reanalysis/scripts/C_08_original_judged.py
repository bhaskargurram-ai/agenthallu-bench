"""C_08: Original interceptor experiment (CSV) re-scored with the GPT-4o-mini judge (C_04 cache),
full task set and 'scoreable' subset (no halueval / toolbench_syn). Also heuristic for the subset."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import read, tf, REPO, OUT, mcnemar_exact, hypergeom_sf, boot_ci, wilson
from C_04_judge import key, load_cache

JC = load_cache()
tasks = {t["task_id"]: t for t in json.load(open(f"{REPO}/data/tasks/full_2000_tasks.json"))}
ctrl, intr = read("interceptor_control_600.csv"), read("interceptor_heldout_600.csv")
out = {}
for m in ["gpt4o_mini", "gemini_20_flash"]:
    C = {r["task_id"]: r for r in ctrl if r["model"] == m}
    I = {r["task_id"]: r for r in intr if r["model"] == m}
    for subset in ["all", "scoreable"]:
        ids = sorted(t for t in C if subset == "all" or t.rsplit("_", 1)[0] not in ("halueval", "toolbench_syn"))
        for lab in ["judge", "heuristic"]:
            def wrong(r):
                if lab == "heuristic":
                    return not tf(r["final_correct"])
                t = tasks[r["task_id"]]
                return not JC[key(t["query"], t["correct_final_answer"], r["final_answer"])]
            n = len(ids)
            ab = [tf(I[t]["abstained"]) for t in ids]
            ch = [wrong(C[t]) for t in ids]
            ih = [(not ab[i]) and wrong(I[t]) for i, t in enumerate(ids)]
            k, H = sum(ab), sum(ch)
            sv = sum(1 for i in range(n) if ab[i] and ch[i])
            svB = sum(1 for i in range(n) if ch[i] and not ih[i]); bkB = sum(1 for i in range(n) if (not ch[i]) and ih[i])
            lo, hi = boot_ci(lambda idx: (sum(ih[i] for i in idx) - sum(ch[i] for i in idx)) / len(idx), n)
            ans = [i for i in range(n) if not ab[i]]
            out[f"{m}|{subset}|{lab}"] = {
                "n": n, "ctrl_halluc": H / n, "int_halluc": sum(ih) / n, "delta_pp": 100 * (sum(ih) - H) / n,
                "delta_paired_boot95_pp": [100 * lo, 100 * hi], "mcnemar_p": mcnemar_exact(svB, bkB),
                "mcnemar_b_c": [svB, bkB], "abstain_rate": k / n, "Sv": sv, "Bk": k - sv,
                "precision": sv / k if k else None, "precision_wilson95": wilson(sv, k) if k else None,
                "recall": sv / H if H else None,
                "selective_acc_int": 1 - sum(ih[i] for i in ans) / len(ans), "ctrl_acc": 1 - H / n,
                "random_matched_halluc": (H - k * H / n) / n,
                "p_better_than_random": hypergeom_sf(sv, n, H, k) if k else None}
json.dump(out, open(f"{OUT}/original_csv_judged.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ("precision_wilson95",)})
