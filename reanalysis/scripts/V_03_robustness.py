"""V_03: sensitivity of validator scores to each normalisation / rubric choice (one switch at a time).

For each variant: per-model baseline accuracy, pooled baseline accuracy, pooled P2 accuracy, fraction of CORRECT
labels that flip vs the default (baseline + P2 + retail full traces), and Cohen's kappa vs the human majority.
Output: V_validator/robustness.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_core as V  # noqa: E402
import V_traces as L  # noqa: E402
import V_02_human_agreement as H  # noqa: E402

OUT = "/Users/challagullayoshitha/Downloads/agentprop_revision/V_validator"
VARIANTS = {
    "default": {},
    "no_casefold": {"casefold": False},
    "no_accent_fold": {"accent_fold": False},
    "no_date_norm": {"date_norm": False},
    "no_time_norm": {"time_norm": False},
    "temp_tol_0": {"temp_tol": 0.0},
    "temp_tol_0.5": {"temp_tol": 0.5},
    "temp_tol_2": {"temp_tol": 2.0},
    "num_tol_1": {"num_tol": 1.0},
    "no_unit_conversion": {"unit_conv": False},
    "no_derived_aggregates": {"derived": False},
    "precision_ignored": {"precision_required": None},
    "precision_ge_0.8": {"precision_required": 0.8},
    "recall_ge_0.5": {"recall_required": 0.5},
    "state_lenient": {"state": "lenient"},
    "gold_from_taskfile": {"gold_source": "taskfile"},
    "no_equivalent_obs": {"equiv_obs": False},
    "no_condition_synonyms": {"cond_synonyms": False},
    "hist_lenient": {"hist_lenient": True},
    "no_aux_obs": {"aux_obs": False},
    "title_lenient": {"title_lenient": True},
}
EXCL = {"not_verifiable", "not_verifiable_answer_only", "infra_error"}


def main():
    recs, _ = L.load_all()
    recs = [r for r in recs if r["source"] in ("baseline", "baseline_retail", "p2", "p2_retail") and r["mode"] == "full"]
    tasks = L.load_tasks()
    human_rows = None
    out = {}
    default_labels = None
    for name, cfg in VARIANTS.items():
        labels = {}
        specs = {}
        for r in recs:
            t = tasks[r["task_id"]]
            if r["task_id"] not in specs:
                specs[r["task_id"]] = V.build_spec(t, dict(V.DEFAULT_CFG, **cfg))
            s = V.score_trace(t, r["calls"], r["final_answer"], cfg=cfg, spec=specs[r["task_id"]])
            labels[r["session_id"]] = (r, s["outcome"], bool(s.get("CORRECT")))
        if default_labels is None:
            default_labels = labels
        res = {"cfg": cfg}
        for src in ("baseline", "p2", "baseline_retail"):
            per = {}
            for m in sorted({r["model"] for r, o, c in labels.values() if r["source"] == src}):
                sel = [c for r, o, c in labels.values() if r["source"] == src and r["model"] == m and o not in EXCL]
                if sel:
                    per[m] = round(sum(sel) / len(sel), 4)
            pooled = [c for r, o, c in labels.values() if r["source"] == src and o not in EXCL]
            res[f"{src}_by_model"] = per
            res[f"{src}_pooled"] = round(sum(pooled) / len(pooled), 4) if pooled else None
        common = [k for k, (r, o, c) in labels.items() if o not in EXCL and default_labels[k][1] not in EXCL]
        res["n_scored"] = len(common)
        res["flip_rate_vs_default"] = round(sum(labels[k][2] != default_labels[k][2] for k in common) / len(common), 4)
        # human agreement
        hr = H.score_human_set(cfg)
        ver = [x for x in hr if x["s"]["verifiable"] and x["s"]["outcome"] != "infra_error"]
        v = [int(bool(x["s"]["CORRECT"])) for x in ver]
        mj = [int(x["maj"]) for x in ver]
        res["kappa_vs_human_majority"] = round(H.kappa(v, mj), 4)
        res["raw_agreement_vs_human_majority"] = round(sum(a == b for a, b in zip(v, mj)) / len(v), 4)
        out[name] = res
        print(name, res["baseline_pooled"], res["p2_pooled"], res["flip_rate_vs_default"], res["kappa_vs_human_majority"],
              flush=True)
    # spread of per-model accuracy across variants (excluding gold_from_taskfile, reported separately)
    spread = {}
    for m in out["default"]["baseline_by_model"]:
        vals = [out[k]["baseline_by_model"].get(m) for k in out if k != "gold_from_taskfile"]
        vals = [x for x in vals if x is not None]
        spread[m] = {"default": out["default"]["baseline_by_model"][m], "min": min(vals), "max": max(vals)}
    # model ranking stability (Kendall tau vs default)
    def ranks(d):
        return [m for m, _ in sorted(d.items(), key=lambda kv: -kv[1])]
    base_rank = ranks(out["default"]["baseline_by_model"])
    for k in out:
        r2 = ranks(out[k]["baseline_by_model"])
        pos = {m: i for i, m in enumerate(r2)}
        conc = disc = 0
        for i in range(len(base_rank)):
            for j in range(i + 1, len(base_rank)):
                a, b = base_rank[i], base_rank[j]
                if pos[a] < pos[b]:
                    conc += 1
                else:
                    disc += 1
        out[k]["kendall_tau_model_ranking_vs_default"] = round((conc - disc) / (conc + disc), 4) if conc + disc else None
    with open(f"{OUT}/robustness.json", "w") as fh:
        json.dump({"variants": out, "per_model_spread_excl_taskfile": spread}, fh, indent=2)


if __name__ == "__main__":
    main()
