"""Per-stage reach probabilities and hop-conditional propagation rates.

Reads p2_all.csv (P2 injection results), computes for each model:
  S1 = injection applied          (eps >= 1)
  S2 = injection executed visibly (eps >= 2, independent of final_correct)
  S3 = final answer wrong         (final_correct == False, independent of eps)

Stage indicators are computed INDEPENDENTLY — S2 uses only the eps column,
S3 uses only the final_correct column.  This avoids the prior bug where
classify_stage() mapped reached_output→stage=3 and then S2=(stage>=2)
included those rows, making p_S2 == p_S3 by construction.

Hop-conditional rates:
  r_12 = p_S2 / p_S1   (fraction of applied injections that executed visibly)
  r_23 = P(S3 | S2)    (fraction of visibly-executed injections that caused wrong answer)

Outputs:
  - analysis/hop_rates.md     (Markdown + LaTeX table)
  - data/results/hop_rates.json (raw metrics)
  - analysis/figures/fig_sankey.png  (headline Sankey figure)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from analysis.bootstrap import bootstrap_ci

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2", default="data/results/p2_all.csv")
    parser.add_argument("--output-md", default="analysis/hop_rates.md")
    parser.add_argument("--output-json", default="data/results/hop_rates.json")
    parser.add_argument("--figure", default="analysis/figures/fig_sankey.png")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    p = pd.read_csv(os.path.join(base, args.p2))
    console.print(f"Loaded {len(p)} P2 rows, models: {sorted(p['model'].unique())}")

    # Filter to semantic_wrong only (column is 'error_type', not 'injection_type')
    type_col = "error_type" if "error_type" in p.columns else "injection_type"
    if type_col in p.columns:
        before = len(p)
        p = p[p[type_col].isin(["semantic_wrong", "p2_semantic_wrong"])]
        console.print(f"Filtered {type_col}=semantic_wrong: {before} → {len(p)} rows")
    else:
        console.print("[yellow]No error_type or injection_type column — using all rows[/yellow]")

    # ---------- Independent stage indicators ----------
    # S1: injection applied (eps >= 1)
    # S2: injection executed with observable effect (eps >= 2)
    # S3: final answer wrong (final_correct == False)
    #
    # These are computed from SEPARATE columns with NO cross-reference.
    # A row can have S3=True without S2=True (injection applied but
    # no observable execution error, yet answer still wrong).
    p["S1"] = (p["eps"].fillna(0).astype(int) >= 1).astype(int)
    p["S2"] = (p["eps"].fillna(0).astype(int) >= 2).astype(int)
    p["S3"] = (~p["final_correct"].astype(bool)).astype(int)

    # Diagnostic: how many rows have S3 without S2?
    s3_no_s2 = int(((p["S3"] == 1) & (p["S2"] == 0)).sum())
    s2_no_s3 = int(((p["S2"] == 1) & (p["S3"] == 0)).sum())
    console.print(f"[dim]Diagnostic: S3=1 & S2=0 (wrong answer w/o visible exec): {s3_no_s2}[/dim]")
    console.print(f"[dim]Diagnostic: S2=1 & S3=0 (visible exec but correct answer): {s2_no_s3}[/dim]")

    # Per-model stage distributions
    metrics = {}
    rows = []
    for model in sorted(p["model"].unique()):
        sub = p[p["model"] == model]
        if len(sub) == 0:
            continue

        pt = {}
        for k in (1, 2, 3):
            arr = sub[f"S{k}"].values
            m, lo, hi = bootstrap_ci(arr)
            pt[k] = {"point": m, "lo": lo, "hi": hi}

        # Hop-conditional rates (marginal)
        r_12 = round(pt[2]["point"] / pt[1]["point"], 4) if pt[1]["point"] > 0 else None

        # r_23: P(S3=1 | S2=1) — computed directly on the S2=1 subset
        s2_rows = sub[sub["S2"] == 1]
        if len(s2_rows) > 0:
            r23_arr = s2_rows["S3"].values
            r23_m, r23_lo, r23_hi = bootstrap_ci(r23_arr)
        else:
            r23_m, r23_lo, r23_hi = None, None, None

        # Also compute: P(S3=1 | S1=1) — injection-to-error rate
        s1_rows = sub[sub["S1"] == 1]
        if len(s1_rows) > 0:
            ite_arr = s1_rows["S3"].values
            ite_m, ite_lo, ite_hi = bootstrap_ci(ite_arr)
        else:
            ite_m, ite_lo, ite_hi = None, None, None

        metrics[model] = {
            "n": len(sub),
            "n_S1": int(sub["S1"].sum()),
            "n_S2": int(sub["S2"].sum()),
            "n_S3": int(sub["S3"].sum()),
            "p_S1": pt[1], "p_S2": pt[2], "p_S3": pt[3],
            "r_1_2": r_12,
            "r_2_3": {"point": r23_m, "lo": r23_lo, "hi": r23_hi} if r23_m is not None else None,
            "injection_to_error": {"point": ite_m, "lo": ite_lo, "hi": ite_hi} if ite_m is not None else None,
        }
        rows.append({
            "model": model,
            "n": len(sub),
            "n_S2": int(sub["S2"].sum()),
            "p_S1": f"{100*pt[1]['point']:.1f} [{100*pt[1]['lo']:.1f}, {100*pt[1]['hi']:.1f}]",
            "p_S2": f"{100*pt[2]['point']:.1f} [{100*pt[2]['lo']:.1f}, {100*pt[2]['hi']:.1f}]",
            "p_S3": f"{100*pt[3]['point']:.1f} [{100*pt[3]['lo']:.1f}, {100*pt[3]['hi']:.1f}]",
            "r_12": f"{r_12:.2f}" if r_12 is not None else "—",
            "r_23": f"{r23_m:.3f} [{r23_lo:.3f}, {r23_hi:.3f}]" if r23_m is not None else "—",
            "inj_to_err": f"{ite_m:.3f} [{ite_lo:.3f}, {ite_hi:.3f}]" if ite_m is not None else "—",
        })

    # Rich table
    tab = Table(title="Stage-Reach and Hop-Conditional Rates (P2 semantic_wrong, independent indicators)")
    for c in ("model", "n", "p(S1) %", "p(S2) %", "p(S3) %", "r_12", "r_23 [CI]", "n_S2", "inj→err [CI]"):
        tab.add_column(c, justify="right")
    for r_ in rows:
        tab.add_row(r_["model"], str(r_["n"]), r_["p_S1"], r_["p_S2"], r_["p_S3"],
                     r_["r_12"], r_["r_23"], str(r_["n_S2"]), r_["inj_to_err"])
    console.print(tab)

    md = ["# Hop-Conditional Propagation Rates (Corrected)\n",
          "Stage indicators are computed **independently**:",
          "- `S1` = injection applied (`eps >= 1`)",
          "- `S2` = injection executed with visible effect (`eps >= 2`, independent of final_correct)",
          "- `S3` = final answer wrong (`final_correct == False`, independent of eps)",
          "",
          "- `r_12` = `p(S2) / p(S1)` — marginal",
          "- `r_23` = `P(S3=1 | S2=1)` — conditional, computed on S2=1 subset only",
          "- `inj→err` = `P(S3=1 | S1=1)` — injection-to-error rate",
          "- 95% CIs from 1,000-iter bootstrap.",
          "",
          "| model | n | p(S1) % | p(S2) % | p(S3) % | r_12 | r_23 [CI] | n_S2 | inj→err [CI] |",
          "|-------|---|---------|---------|---------|------|-----------|------|--------------|"]
    for r_ in rows:
        md.append(f"| {r_['model']} | {r_['n']} | {r_['p_S1']} | {r_['p_S2']} | {r_['p_S3']} | {r_['r_12']} | {r_['r_23']} | {r_['n_S2']} | {r_['inj_to_err']} |")
    md.append("")

    os.makedirs(os.path.join(base, "analysis"), exist_ok=True)
    with open(os.path.join(base, args.output_md), "w") as f:
        f.write("\n".join(md))
    os.makedirs(os.path.dirname(os.path.join(base, args.output_json)), exist_ok=True)
    with open(os.path.join(base, args.output_json), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Bar figure: independent stage reach + conditional r_23 ──
    try:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(os.path.join(base, args.figure)), exist_ok=True)

        fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4.5), squeeze=False)
        axes = axes[0]
        for ax, model in zip(axes, sorted(metrics.keys())):
            m = metrics[model]
            stages = [m["p_S1"]["point"], m["p_S2"]["point"], m["p_S3"]["point"]]
            r23_val = m["r_2_3"]["point"] if m["r_2_3"] is not None else 0
            labels = ["S1\nApplied", "S2\nExecuted", "S3\nWrong"]
            colors = ["#4C72B0", "#C44E52", "#8172B2"]
            bars = ax.bar(range(3), stages, color=colors, edgecolor="black", linewidth=0.5)
            for i, s in enumerate(stages):
                ax.text(i, s + 0.02, f"{100*s:.0f}%", ha="center", fontsize=9)
            ax.set_xticks(range(3))
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_title(f"{model}\nr₂₃={r23_val:.2f}" if r23_val else model, fontsize=9)
            ax.set_ylabel("Rate")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.suptitle("Independent Stage Reach Rates (semantic_wrong, error_type filter)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(base, args.figure), dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved figure → {args.figure}[/green]")
    except Exception as e:
        console.print(f"[yellow]Figure skipped: {e}[/yellow]")

    console.print(f"[green]Tables → {args.output_md}[/green]")
    console.print(f"[green]Metrics → {args.output_json}[/green]")


if __name__ == "__main__":
    main()
