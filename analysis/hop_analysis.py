"""Fix 2: Per-stage reach probabilities and hop-conditional propagation rates.

Reads p2_all.csv (P2 injection results), computes for each model:
  p_S1 = P(parameter stage reached)
  p_S2 = P(observation stage reached | S1)   (from injection_applied + param_errors)
  p_S3 = P(reasoning stage reached)           (derived from tool_steps post-injection)
  p_S4 = P(final answer wrong)                (from final_correct)

And hop-conditional rates r_k = p_{k+1} / p_k.

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


def classify_stage(row) -> int:
    """Map one P2 row to the deepest stage reached (0..3).
      0 = injection did not apply
      1 = parameter stage entered (eps>=1, injector modified the planned call)
      2 = tool-execution stage reached (eps>=2, bad params actually executed)
      3 = output stage corrupted (reached_output, injection effect visible in final)
    """
    stage = 0
    eps = row.get("eps")
    reached = bool(row.get("reached_output", False))

    if pd.notna(eps) and int(eps) >= 1:
        stage = 1
    if pd.notna(eps) and int(eps) >= 2:
        stage = max(stage, 2)
    if reached:
        stage = max(stage, 3)
    return stage


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

    # For this paper we focus on semantic_wrong as the canonical injection
    if "injection_type" in p.columns:
        p = p[p["injection_type"].isin(["semantic_wrong", "p2_semantic_wrong"])]
    console.print(f"Semantic-wrong rows: {len(p)}")

    p["stage"] = p.apply(classify_stage, axis=1)

    # Per-model stage distributions
    metrics = {}
    rows = []
    for model in sorted(p["model"].unique()):
        sub = p[p["model"] == model]
        if len(sub) == 0:
            continue
        # Reach indicators S_k = 1 iff stage >= k
        S = {k: (sub["stage"].astype(int) >= k).astype(int).values for k in (1, 2, 3)}
        pt = {}
        for k, arr in S.items():
            m, lo, hi = bootstrap_ci(arr)
            pt[k] = {"point": m, "lo": lo, "hi": hi}
        # Hop-conditional rates
        r = {}
        for k in (1, 2):
            denom = pt[k]["point"]
            numer = pt[k+1]["point"]
            r[f"r_{k}_{k+1}"] = round(numer / denom, 4) if denom > 0 else None
        metrics[model] = {
            "n": len(sub),
            "p_S1": pt[1], "p_S2": pt[2], "p_S3": pt[3],
            **r,
        }
        rows.append({
            "model": model,
            "n": len(sub),
            "p_S1": f"{100*pt[1]['point']:.1f} [{100*pt[1]['lo']:.1f}, {100*pt[1]['hi']:.1f}]",
            "p_S2": f"{100*pt[2]['point']:.1f} [{100*pt[2]['lo']:.1f}, {100*pt[2]['hi']:.1f}]",
            "p_S3": f"{100*pt[3]['point']:.1f} [{100*pt[3]['lo']:.1f}, {100*pt[3]['hi']:.1f}]",
            "r_12": f"{r['r_1_2']:.2f}" if r['r_1_2'] is not None else "—",
            "r_23": f"{r['r_2_3']:.2f}" if r['r_2_3'] is not None else "—",
        })

    # Rich table
    tab = Table(title="Stage-Reach and Hop-Conditional Rates (P2 semantic_wrong)")
    for c in ("model", "n", "p(S1) %", "p(S2) %", "p(S3) %", "r_12", "r_23"):
        tab.add_column(c, justify="right")
    for r_ in rows:
        tab.add_row(r_["model"], str(r_["n"]), r_["p_S1"], r_["p_S2"], r_["p_S3"], r_["r_12"], r_["r_23"])
    console.print(tab)

    md = ["# Hop-Conditional Propagation Rates\n",
          "Columns:",
          "- `p(S_k)` = probability an injection reaches stage k (1=param, 2=obs, 3=final)",
          "- `r_{k,k+1}` = conditional probability of advancing given stage k reached",
          "- 95% CIs from 1,000-iter bootstrap.",
          "",
          "| model | n | p(S1) % | p(S2) % | p(S3) % | r_12 | r_23 |",
          "|-------|---|---------|---------|---------|------|------|"]
    for r_ in rows:
        md.append(f"| {r_['model']} | {r_['n']} | {r_['p_S1']} | {r_['p_S2']} | {r_['p_S3']} | {r_['r_12']} | {r_['r_23']} |")
    md.append("")

    os.makedirs(os.path.join(base, "analysis"), exist_ok=True)
    with open(os.path.join(base, args.output_md), "w") as f:
        f.write("\n".join(md))
    os.makedirs(os.path.dirname(os.path.join(base, args.output_json)), exist_ok=True)
    with open(os.path.join(base, args.output_json), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Sankey figure ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        os.makedirs(os.path.dirname(os.path.join(base, args.figure)), exist_ok=True)

        fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4), squeeze=False)
        axes = axes[0]
        for ax, model in zip(axes, sorted(metrics.keys())):
            m = metrics[model]
            # 4 stages: S0=injected (always 100%), S1, S2, S3
            stages = [1.0, m["p_S1"]["point"], m["p_S2"]["point"], m["p_S3"]["point"]]
            labels = ["Inject", "Param", "Obs", "Final"]
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
            for i, (s, lbl, col) in enumerate(zip(stages, labels, colors)):
                ax.bar(i, s, color=col, edgecolor="black", linewidth=0.5)
                ax.text(i, s + 0.02, f"{100*s:.0f}%", ha="center", fontsize=9)
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, rotation=0, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_title(model, fontsize=10)
            ax.set_ylabel("Reach probability")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.suptitle("Error Propagation by Stage (semantic_wrong injection)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(base, args.figure), dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved figure → {args.figure}[/green]")
    except Exception as e:
        console.print(f"[yellow]Figure skipped: {e}[/yellow]")

    console.print(f"[green]Tables → {args.output_md}[/green]")
    console.print(f"[green]Metrics → {args.output_json}[/green]")


if __name__ == "__main__":
    main()
