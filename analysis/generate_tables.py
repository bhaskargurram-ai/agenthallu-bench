"""Day 8 STEP 8: Generate all paper tables with bootstrap CIs.

Table 1: Baseline vs P2 per model — final_correct, halluc rate, EPS with 95% CIs.
Table 2: Interceptor — hallucination, abstain, recall, FP% with CIs.
Table 3: Multi-seed stability — std across seeds.
Table 4: Ablation — L1_only vs L1_L2 vs L1_L2_L3 deltas.
Table A4 (power): Achieved power per subgroup claim.

Outputs both Markdown (for review) and LaTeX (for paper).
Written to: analysis/tables_day8.md + analysis/tables_day8.tex
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from analysis.bootstrap import bootstrap_ci

console = Console()


def ci_cell(point: float, lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"{100*point:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ── TABLE 1: Baseline + P2 per model ─────────────────────────────────────────

def table_1_main(baseline_csv: str, p2_csv: str) -> pd.DataFrame:
    b = pd.read_csv(baseline_csv)
    p = pd.read_csv(p2_csv)

    # Paired subset: only tasks that appear in both
    common_tasks = set(b["task_id"]) & set(p["task_id"])
    b = b[b["task_id"].isin(common_tasks)]
    p = p[p["task_id"].isin(common_tasks)]

    rows = []
    for model in sorted(set(b["model"].unique()) | set(p["model"].unique())):
        bm = b[b["model"] == model]
        pm = p[p["model"] == model]

        # Baseline correctness
        if len(bm) > 0:
            bc = bm["final_correct"].astype(int).values
            bc_pt, bc_lo, bc_hi = bootstrap_ci(bc)
        else:
            bc_pt = bc_lo = bc_hi = 0.0

        # P2 correctness
        if len(pm) > 0:
            pc = pm["final_correct"].astype(int).values
            pc_pt, pc_lo, pc_hi = bootstrap_ci(pc)
        else:
            pc_pt = pc_lo = pc_hi = 0.0

        # P2 EPS
        eps_col = "eps" if "eps" in pm.columns else None
        if eps_col and len(pm) > 0:
            eps = pm[eps_col].astype(float).values
            eps_pt, eps_lo, eps_hi = bootstrap_ci(eps)
        else:
            eps_pt = eps_lo = eps_hi = 0.0

        rows.append({
            "model": model,
            "n_baseline": len(bm),
            "n_p2": len(pm),
            "baseline_correct_ci": ci_cell(bc_pt, bc_lo, bc_hi),
            "p2_correct_ci": ci_cell(pc_pt, pc_lo, pc_hi),
            "p2_eps_ci": ci_cell(eps_pt, eps_lo, eps_hi, pct=False),
        })
    return pd.DataFrame(rows)


# ── TABLE 2: Interceptor ─────────────────────────────────────────────────────

def table_2_interceptor(intercept_csv: str) -> pd.DataFrame | None:
    if not os.path.exists(intercept_csv):
        return None
    df = pd.read_csv(intercept_csv)
    rows = []
    for model in sorted(df["model"].unique()):
        for cond in sorted(df["condition"].unique()):
            sub = df[(df["model"] == model) & (df["condition"] == cond)]
            if len(sub) == 0:
                continue
            h = sub["hallucination"].astype(int).values
            a = sub["abstained"].astype(int).values
            fc = sub["final_correct"].astype(int).values
            h_pt, h_lo, h_hi = bootstrap_ci(h)
            a_pt, a_lo, a_hi = bootstrap_ci(a)
            c_pt, c_lo, c_hi = bootstrap_ci(fc)

            # FP rate: abstained when final_correct would have been true — proxy: abstained AND gt matches
            # Since we don't have pre-abstain answers we report abstain rate on correct-looking answers
            fp_den = sub[sub["final_correct"] == True]
            fp_num = fp_den["abstained"].astype(int).sum()
            fp_rate = fp_num / len(fp_den) if len(fp_den) else 0.0

            rows.append({
                "model": model,
                "condition": cond,
                "n": len(sub),
                "halluc_ci": ci_cell(h_pt, h_lo, h_hi),
                "abstain_ci": ci_cell(a_pt, a_lo, a_hi),
                "correct_ci": ci_cell(c_pt, c_lo, c_hi),
                "fp_rate": f"{100*fp_rate:.1f}%",
            })
    return pd.DataFrame(rows)


# ── TABLE 3: Multi-seed stability ────────────────────────────────────────────

def table_3_seeds(seed_csv: str) -> pd.DataFrame | None:
    if not os.path.exists(seed_csv):
        return None
    df = pd.read_csv(seed_csv)
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        per_seed = g.groupby("seed").agg(
            halluc=("hallucination", "mean"),
            eps=("eps", "mean"),
        )
        rows.append({
            "model": model,
            "condition": cond,
            "seeds": len(per_seed),
            "halluc_mean": f"{100*per_seed['halluc'].mean():.1f}",
            "halluc_std": f"{100*per_seed['halluc'].std():.2f}",
            "eps_mean": f"{per_seed['eps'].mean():.3f}",
            "eps_std": f"{per_seed['eps'].std():.3f}",
        })
    return pd.DataFrame(rows)


# ── TABLE 4: Ablation ────────────────────────────────────────────────────────

def table_4_ablation(ablation_csv: str) -> pd.DataFrame | None:
    if not os.path.exists(ablation_csv):
        return None
    df = pd.read_csv(ablation_csv)
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        h = g["hallucination"].astype(int).values
        a = g["abstained"].astype(int).values
        c = g["final_correct"].astype(int).values
        rows.append({
            "model": model,
            "condition": cond,
            "n": len(g),
            "halluc": ci_cell(*bootstrap_ci(h)),
            "abstain": ci_cell(*bootstrap_ci(a)),
            "correct": ci_cell(*bootstrap_ci(c)),
        })
    return pd.DataFrame(rows)


# ── TABLE 5: Self-Consistency ────────────────────────────────────────────────

def table_5_self_consistency(sc_csv: str) -> pd.DataFrame | None:
    if not os.path.exists(sc_csv):
        return None
    df = pd.read_csv(sc_csv)
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        h = g["hallucination"].astype(int).values
        a = g["abstained"].astype(int).values
        c = g["final_correct"].astype(int).values
        rows.append({
            "model": model,
            "condition": cond,
            "n": len(g),
            "halluc": ci_cell(*bootstrap_ci(h)),
            "abstain": ci_cell(*bootstrap_ci(a)),
            "correct": ci_cell(*bootstrap_ci(c)),
            "avg_unique_ans": f"{g['unique_answers'].mean():.2f}",
        })
    return pd.DataFrame(rows)


# ── TABLE 6: Ensemble-Judge Agreement ────────────────────────────────────────

def table_6_ensemble_judge(metrics_json: str) -> pd.DataFrame | None:
    if not os.path.exists(metrics_json):
        return None
    with open(metrics_json) as f:
        m = json.load(f)
    rows = [{"rater": "heuristic", "kappa": f"{m.get('heuristic_kappa', 0):.3f}"}]
    for k, v in m.items():
        if k.endswith("_kappa") and k != "heuristic_kappa":
            rows.append({"rater": k.replace("_kappa", ""), "kappa": f"{v:.3f}"})
    rows.append({"rater": "_unanimous rate", "kappa": f"{m.get('unanimous_rate', 0):.3f}"})
    rows.append({"rater": "_n", "kappa": str(m.get("n", 0))})
    return pd.DataFrame(rows)


# ── TABLE 7: Cross-Benchmark (held-out tasks) ────────────────────────────────

def table_7_crossbench(xb_csv: str, orig_p2_csv: str) -> pd.DataFrame | None:
    if not os.path.exists(xb_csv):
        return None
    xb = pd.read_csv(xb_csv)
    if not os.path.exists(orig_p2_csv):
        return None
    op = pd.read_csv(orig_p2_csv)
    # filter to semantic_wrong if that column exists
    if "injection_type" in op.columns:
        op = op[op["injection_type"].isin(["semantic_wrong", "p2_semantic_wrong"])]
    rows = []
    for model in sorted(set(xb["model"].unique()) | set(op["model"].unique())):
        xm = xb[xb["model"] == model]
        om = op[op["model"] == model]
        def block(df):
            if len(df) == 0:
                return None, None, None
            h = (1 - df["final_correct"].astype(int)).values
            hpt, hlo, hhi = bootstrap_ci(h)
            return hpt, hlo, hhi
        xpt, xlo, xhi = block(xm)
        opt, olo, ohi = block(om)
        rows.append({
            "model": model,
            "n_original": len(om),
            "halluc_original": ci_cell(opt or 0, olo or 0, ohi or 0) if opt is not None else "—",
            "n_heldout": len(xm),
            "halluc_heldout": ci_cell(xpt or 0, xlo or 0, xhi or 0) if xpt is not None else "—",
        })
    return pd.DataFrame(rows)


# ── TABLE 8: Cross-Domain Replication (Retail) ────────────────────────────────

def table_8_retail(baseline_csv: str, p2_csv: str) -> pd.DataFrame | None:
    if not (os.path.exists(baseline_csv) and os.path.exists(p2_csv)):
        return None
    b = pd.read_csv(baseline_csv)
    p = pd.read_csv(p2_csv)
    rows = []
    for model in sorted(set(b["model"]) | set(p["model"])):
        bm = b[b["model"] == model]; pm = p[p["model"] == model]
        bc = bm["final_correct"].astype(int).values if len(bm) else np.array([0])
        pc = pm["final_correct"].astype(int).values if len(pm) else np.array([0])
        eps = pm["eps"].astype(float).values if "eps" in pm and len(pm) else np.array([0.0])
        rows.append({
            "model": model,
            "n": len(pm),
            "baseline_correct": ci_cell(*bootstrap_ci(bc)),
            "p2_correct": ci_cell(*bootstrap_ci(pc)),
            "p2_eps": ci_cell(*bootstrap_ci(eps), pct=False),
        })
    return pd.DataFrame(rows)


# ── TABLE 9: Tuned Interceptor (threshold sweep) ──────────────────────────────

def table_9_tuned_interceptor(sweep_json: str) -> pd.DataFrame | None:
    if not os.path.exists(sweep_json):
        return None
    with open(sweep_json) as f:
        d = json.load(f)
    rows = [{
        "metric": "best_cfg",
        "value": json.dumps(d.get("best_cfg", {})),
    }, {
        "metric": "best_F1",
        "value": f"{d.get('best_f1', 0):.3f}",
    }, {
        "metric": "best_precision",
        "value": f"{d.get('best_precision', 0):.3f}",
    }, {
        "metric": "best_recall",
        "value": f"{d.get('best_recall', 0):.3f}",
    }]
    return pd.DataFrame(rows)


# ── TABLE 10: Human Inter-Rater Agreement (Fix 3) ─────────────────────────────

def table_10_human_kappa(json_path: str) -> pd.DataFrame | None:
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    rows = [{"metric": "n_common", "value": str(d.get("n_common"))},
            {"metric": "n_annotators", "value": str(d.get("n_annotators"))},
            {"metric": "Fleiss κ (humans)", "value": f"{d.get('fleiss_kappa_humans', 0):.3f}"},
            {"metric": "Ensemble vs heuristic κ", "value": f"{d.get('ensemble_vs_heuristic_kappa', 0):.3f}"}]
    for a, m in (d.get("per_annotator") or {}).items():
        rows.append({"metric": f"κ({a}, ensemble)", "value": f"{m['kappa_vs_ensemble']:.3f}"})
        rows.append({"metric": f"κ({a}, heuristic)", "value": f"{m['kappa_vs_heuristic']:.3f}"})
    return pd.DataFrame(rows)


# ── TABLE A4: Power ──────────────────────────────────────────────────────────

def table_power(power_json: str) -> pd.DataFrame | None:
    if not os.path.exists(power_json):
        return None
    with open(power_json) as f:
        data = json.load(f)
    claims = data.get("claims", [])
    rows = []
    for c in claims:
        rows.append({
            "claim": c["claim"],
            "n_a": c["n_a"],
            "n_b": c["n_b"],
            "p_a": f"{c['group_a_p']:.3f}",
            "p_b": f"{c['group_b_p']:.3f}",
            "h": f"{c['effect_h']:.2f}",
            "power": f"{c['power']:.2f}",
            "underpowered": c["underpowered"],
        })
    return pd.DataFrame(rows)


# ── Output ───────────────────────────────────────────────────────────────────

def df_to_md(df: pd.DataFrame, title: str) -> str:
    if df is None or len(df) == 0:
        return f"## {title}\n\n_(no data)_\n\n"
    out = [f"## {title}", ""]
    out.append(df.to_markdown(index=False))
    out.append("")
    return "\n".join(out)


def df_to_latex(df: pd.DataFrame, title: str, label: str) -> str:
    if df is None or len(df) == 0:
        return f"% {title}: no data\n"
    try:
        body = df.to_latex(index=False, escape=True)
    except Exception as e:
        body = f"% to_latex failed: {e}\n" + df.to_string(index=False)
    return (
        f"% {title}\n"
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{title}}}\n"
        f"\\label{{tab:{label}}}\n{body}\n\\end{{table}}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/results/baseline_all.csv")
    parser.add_argument("--p2", default="data/results/p2_all.csv")
    parser.add_argument("--interceptor", default="data/results/interceptor_results.csv")
    parser.add_argument("--seeds", default="data/results/multi_seed_results.csv")
    parser.add_argument("--ablation", default="data/results/ablation_results.csv")
    parser.add_argument("--power", default="data/results/power_analysis.json")
    parser.add_argument("--sc", default="data/results/self_consistency_results.csv")
    parser.add_argument("--ensemble", default="data/results/ensemble_judge_metrics.json")
    parser.add_argument("--crossbench", default="data/results/p2_crossbench.csv")
    parser.add_argument("--retail-baseline", default="data/results/baseline_retail.csv")
    parser.add_argument("--retail-p2", default="data/results/p2_retail.csv")
    parser.add_argument("--sweep", default="analysis/interceptor_tuned_thresholds.json")
    parser.add_argument("--human-kappa", default="data/results/human_kappa.json")
    parser.add_argument("--output-md", default="analysis/tables_day8.md")
    parser.add_argument("--output-tex", default="analysis/tables_day8.tex")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")

    console.rule("[bold cyan]Day 8 STEP 8 — Generate Tables[/bold cyan]")

    t1 = table_1_main(os.path.join(base, args.baseline), os.path.join(base, args.p2))
    t2 = table_2_interceptor(os.path.join(base, args.interceptor))
    t3 = table_3_seeds(os.path.join(base, args.seeds))
    t4 = table_4_ablation(os.path.join(base, args.ablation))
    t5 = table_5_self_consistency(os.path.join(base, args.sc))
    t6 = table_6_ensemble_judge(os.path.join(base, args.ensemble))
    t7 = table_7_crossbench(os.path.join(base, args.crossbench), os.path.join(base, args.p2))
    t8 = table_8_retail(os.path.join(base, args.retail_baseline), os.path.join(base, args.retail_p2))
    t9 = table_9_tuned_interceptor(os.path.join(base, args.sweep))
    t10 = table_10_human_kappa(os.path.join(base, args.human_kappa))
    ta4 = table_power(os.path.join(base, args.power))

    for name, t in [("T1 main", t1), ("T2 interceptor", t2),
                    ("T3 seeds", t3), ("T4 ablation", t4),
                    ("T5 self-consistency", t5), ("T6 ensemble judge", t6),
                    ("T7 cross-benchmark", t7), ("T8 retail", t8),
                    ("T9 tuned-interceptor", t9), ("T10 human-κ", t10),
                    ("A4 power", ta4)]:
        console.print(f"{name}: {len(t) if t is not None else 0} rows")

    md = []
    md.append("# AgentHallu-Bench Results Tables (Day 8)\n")
    md.append(df_to_md(t1, "Table 1: Main Results (Baseline vs P2)"))
    md.append(df_to_md(t2, "Table 2: Interceptor Performance"))
    md.append(df_to_md(t3, "Table 3: Multi-Seed Stability"))
    md.append(df_to_md(t4, "Table 4: Interceptor Ablation"))
    md.append(df_to_md(t5, "Table 5: Self-Consistency Voting (M2)"))
    md.append(df_to_md(t6, "Table 6: Judge-Agreement Ensemble (M1)"))
    md.append(df_to_md(t7, "Table 7: Cross-Benchmark Validation (A2)"))
    md.append(df_to_md(t8, "Table 8: Cross-Domain Replication — Retail (Fix 4)"))
    md.append(df_to_md(t9, "Table 9: Tuned Interceptor — Threshold Sweep (Fix 1)"))
    md.append(df_to_md(t10, "Table 10: Human-Annotator Agreement (Fix 3)"))
    md.append(df_to_md(ta4, "Table A4: Achieved Power per Claim"))
    out_md = os.path.join(base, args.output_md)
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    console.print(f"[green]Markdown tables: {out_md}[/green]")

    tex = []
    tex.append("% AgentHallu-Bench tables (auto-generated Day 8)\n")
    tex.append(df_to_latex(t1, "Main Results", "main"))
    tex.append(df_to_latex(t2, "Interceptor Performance", "interceptor"))
    tex.append(df_to_latex(t3, "Multi-Seed Stability", "seeds"))
    tex.append(df_to_latex(t4, "Interceptor Ablation", "ablation"))
    tex.append(df_to_latex(t5, "Self-Consistency Voting", "self_consistency"))
    tex.append(df_to_latex(t6, "Ensemble-Judge Agreement", "ensemble_judge"))
    tex.append(df_to_latex(t7, "Cross-Benchmark Validation", "crossbench"))
    tex.append(df_to_latex(t8, "Cross-Domain Replication (Retail)", "retail"))
    tex.append(df_to_latex(t9, "Tuned Interceptor", "tuned_int"))
    tex.append(df_to_latex(t10, "Human-Annotator Agreement", "human_kappa"))
    tex.append(df_to_latex(ta4, "Achieved Power", "power"))
    out_tex = os.path.join(base, args.output_tex)
    with open(out_tex, "w") as f:
        f.write("\n".join(tex))
    console.print(f"[green]LaTeX tables: {out_tex}[/green]")


if __name__ == "__main__":
    main()
