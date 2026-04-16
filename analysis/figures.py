"""Figure generation for AgentHallu-Bench paper.

Figure 1: EPS heatmap (5 models × 4 error types)
Figure 2: Cascade depth by domain (3-panel violin)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rich.console import Console

console = Console()

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ERROR_TYPES = ["type_mismatch", "out_of_range", "missing_required", "semantic_wrong"]
ERROR_LABELS = ["Type\nMismatch", "Out of\nRange", "Missing\nRequired", "Semantic\nWrong"]

MODEL_ORDER = [
    "gpt41_nano", "gpt35_turbo", "gpt4o_mini", "gpt41_mini",
    "gemini_20_flash", "deepseek_v3", "o3_mini", "gpt4o", "gemini_25_flash",
]
MODEL_LABELS = [
    "GPT-4.1-nano", "GPT-3.5-Turbo", "GPT-4o-mini", "GPT-4.1-mini",
    "Gemini-2.0-Flash", "DeepSeek-V3", "o3-mini (50)", "GPT-4o (50)", "Gemini-2.5-Flash (50)",
]


def load_p2_data(path: str) -> pd.DataFrame:
    """Load P2 results CSV."""
    df = pd.read_csv(path)
    console.print(f"Loaded {len(df)} rows from {path}")
    return df


def load_frontier_data(path: str) -> pd.DataFrame:
    """Load frontier comparison CSV if it exists."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        console.print(f"Loaded {len(df)} frontier rows from {path}")
        return df
    return pd.DataFrame()


def figure_1_eps_heatmap(p2_df: pd.DataFrame, frontier_df: pd.DataFrame = None):
    """Figure 1: 5×4 heatmap of mean EPS (models × error types).

    Rows: gpt4o_mini, gemini_20_flash, deepseek_v3, gpt4o(50), gemini25(50)
    Columns: type_mismatch, out_of_range, missing_required, semantic_wrong
    Values: mean EPS per cell
    Color: Blues colormap (dark=high EPS=bad)
    """
    # Combine main + frontier data
    df = p2_df.copy()
    if frontier_df is not None and not frontier_df.empty:
        df = pd.concat([df, frontier_df], ignore_index=True)

    # Build heatmap matrix
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    labels_present = [MODEL_LABELS[MODEL_ORDER.index(m)] for m in models_present]

    matrix = np.zeros((len(models_present), len(ERROR_TYPES)))
    counts = np.zeros((len(models_present), len(ERROR_TYPES)), dtype=int)

    for i, model in enumerate(models_present):
        for j, et in enumerate(ERROR_TYPES):
            subset = df[(df["model"] == model) & (df["error_type"] == et)]
            if len(subset) > 0:
                matrix[i, j] = subset["eps"].mean()
                counts[i, j] = len(subset)

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(3, matrix.max()))

    # Annotate each cell
    for i in range(len(models_present)):
        for j in range(len(ERROR_TYPES)):
            val = matrix[i, j]
            n = counts[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}\nn={n}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(ERROR_TYPES)))
    ax.set_xticklabels(ERROR_LABELS, fontsize=10)
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels(labels_present, fontsize=10)
    ax.set_xlabel("Error Type", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("Figure 1: Mean Error Propagation Score (EPS) by Model and Error Type", fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean EPS (0=caught immediately, 3=reached output)", fontsize=9)

    plt.tight_layout()

    pdf_path = os.path.join(FIGURES_DIR, "fig1_eps_heatmap.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig1_eps_heatmap.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure 1 saved: {pdf_path} + {png_path}[/green]")

    # Print ASCII summary
    console.print("\n[bold]Figure 1 — EPS Heatmap Summary:[/bold]")
    for i, model in enumerate(labels_present):
        vals = " | ".join(f"{matrix[i,j]:.2f}" for j in range(len(ERROR_TYPES)))
        console.print(f"  {model:20s}: {vals}")


def figure_2_cascade_by_domain(p2_df: pd.DataFrame):
    """Figure 2: 3-panel violin plots of EPS distribution by domain.

    Uses only gpt4o_mini + gemini_20_flash (largest n).
    Each panel: one domain, violins for each error type.
    """
    df = p2_df[p2_df["model"].isin(["gpt4o_mini", "gemini_20_flash"])].copy()
    domains = [d for d in ["weather", "calendar", "medical"] if d in df["domain"].unique()]

    if not domains:
        console.print("[yellow]No domain data for Figure 2 — skipping[/yellow]")
        return

    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 5), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]
        domain_df = df[df["domain"] == domain]

        data_by_et = []
        labels = []
        for et in ERROR_TYPES:
            subset = domain_df[domain_df["error_type"] == et]["eps"]
            if len(subset) > 0:
                data_by_et.append(subset.values)
                labels.append(et.replace("_", "\n"))
            else:
                data_by_et.append(np.array([0]))
                labels.append(et.replace("_", "\n"))

        parts = ax.violinplot(data_by_et, positions=range(len(ERROR_TYPES)),
                               showmeans=True, showmedians=True)

        for pc_idx, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[pc_idx % len(colors)])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(ERROR_TYPES)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{domain.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("EPS" if ax_idx == 0 else "")
        ax.set_ylim(-0.5, 4)
        ax.grid(axis="y", alpha=0.3)

        # Add n counts
        for i, et in enumerate(ERROR_TYPES):
            n = len(domain_df[domain_df["error_type"] == et])
            ax.text(i, -0.3, f"n={n}", ha="center", fontsize=7, color="gray")

    fig.suptitle("Figure 2: EPS Distribution by Domain and Error Type\n(GPT-4o-mini + Gemini-2.0-Flash)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    pdf_path = os.path.join(FIGURES_DIR, "fig2_cascade_domain.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig2_cascade_domain.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure 2 saved: {pdf_path} + {png_path}[/green]")


def figure_3_detection_vs_eps(p2_df: pd.DataFrame):
    """Figure 3: Scatter — self-detection rate vs mean EPS per (model, error_type).

    X: proportion of sessions where tool validation caught the error
    Y: mean EPS
    Each point labeled with model+error_type abbreviation.
    Fit regression line, show R² in legend.
    """
    from scipy import stats as scipy_stats

    records = []
    for (model, et), group in p2_df.groupby(["model", "error_type"]):
        if len(group) < 5:
            continue
        # Detection rate: sessions where EPS=0 (error caught at origin)
        detection_rate = (group["eps"] == 0).mean()
        mean_eps = group["eps"].mean()
        label = f"{model[:6]}_{et[:4]}"
        records.append({"model": model, "error_type": et, "detection_rate": detection_rate,
                         "mean_eps": mean_eps, "label": label})

    if not records:
        console.print("[yellow]No data for Figure 3[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.array([r["detection_rate"] for r in records])
    y = np.array([r["mean_eps"] for r in records])

    # Color by model
    models = sorted(set(r["model"] for r in records))
    cmap = plt.cm.get_cmap("tab10", len(models))
    model_colors = {m: cmap(i) for i, m in enumerate(models)}

    for r in records:
        ax.scatter(r["detection_rate"], r["mean_eps"], color=model_colors[r["model"]],
                   s=60, zorder=5)
        ax.annotate(r["label"], (r["detection_rate"], r["mean_eps"]),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    # Regression line
    if len(x) > 2:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        x_line = np.linspace(0, 1, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"R²={r_value**2:.3f}, p={p_value:.4f}")
        ax.legend(fontsize=9)

    # Legend for models
    for m in models:
        ax.scatter([], [], color=model_colors[m], label=m, s=40)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax.set_xlabel("Self-Detection Rate (proportion EPS=0)", fontsize=11)
    ax.set_ylabel("Mean EPS", fontsize=11)
    ax.set_title("Figure 3: Detection Rate vs Error Propagation Score", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(FIGURES_DIR, "fig3_detection_vs_eps.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig3_detection_vs_eps.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure 3 saved: {pdf_path} + {png_path}[/green]")


def figure_4_mtas_curves(p3_df: pd.DataFrame):
    """Figure 4: MTAS survival curves — proportion of sessions with active false belief per turn.

    X: turn number 1-8
    Y: proportion of sessions where false belief still active
    One line per model (3 lines)
    Vertical dashed line at injection turn (turn 2)
    """
    if p3_df.empty:
        console.print("[yellow]No P3 data for Figure 4[/yellow]")
        return

    models = sorted(p3_df["model"].unique())
    num_turns = int(p3_df["num_turns"].max()) if "num_turns" in p3_df.columns else 8
    injection_turn = int(p3_df["injection_turn"].iloc[0]) if "injection_turn" in p3_df.columns else 2

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, model in enumerate(models):
        model_df = p3_df[p3_df["model"] == model]
        n_sessions = len(model_df)

        # For each turn after injection, compute survival rate
        # Use propagation_depth: if depth >= (turn - injection_turn), belief still active
        turns = list(range(1, num_turns + 1))
        survival = []
        for t in turns:
            if t <= injection_turn:
                survival.append(0.0)  # Before injection
            else:
                # Proportion of sessions where propagation_depth >= (t - injection_turn)
                turns_since_inj = t - injection_turn
                active = (model_df["propagation_depth"] >= turns_since_inj).mean()
                survival.append(active)

        ax.plot(turns, survival, color=colors[i % len(colors)], marker="o",
                linewidth=2, label=f"{model} (n={n_sessions})", markersize=5)

    # Injection point
    ax.axvline(x=injection_turn, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(injection_turn + 0.1, 0.95, "injection", fontsize=9, color="gray", ha="left")

    ax.set_xlabel("Turn Number", fontsize=11)
    ax.set_ylabel("Proportion with Active False Belief", fontsize=11)
    ax.set_title("Figure 4: Multi-Turn False Belief Persistence (MTAS Curves)", fontsize=12)
    ax.set_xlim(0.5, num_turns + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, num_turns + 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(FIGURES_DIR, "fig4_mtas_curves.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig4_mtas_curves.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure 4 saved: {pdf_path} + {png_path}[/green]")


def figure_5_cross_agent(p4_df: pd.DataFrame):
    """Figure 5: Cross-agent error flow — 3 horizontal panels (one per injection type).

    Each panel: 3 boxes (Agent1, Agent2, Agent3) with arrows.
    Arrow width = propagation rate. Box color = error presence.
    """
    if p4_df.empty:
        console.print("[yellow]No P4 data for Figure 5[/yellow]")
        return

    injection_types = sorted(p4_df["injection_type"].unique())
    n_panels = len(injection_types)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax_idx, inj_type in enumerate(injection_types):
        ax = axes[ax_idx]
        subset = p4_df[p4_df["injection_type"] == inj_type]
        n = len(subset)

        # Compute propagation rates
        rate_to_a2 = subset["reached_agent2"].mean() if n > 0 else 0
        rate_to_a3 = subset["reached_agent3"].mean() if n > 0 else 0

        # Draw boxes
        box_positions = [(0.1, 0.3), (0.4, 0.3), (0.7, 0.3)]
        box_w, box_h = 0.2, 0.3
        labels = ["Agent 1\n(Planner)", "Agent 2\n(Executor)", "Agent 3\n(Synthesizer)"]

        # Agent 1 always has error (red), others based on rate
        error_rates = [1.0, rate_to_a2, rate_to_a3]

        for i, (bx, by) in enumerate(box_positions):
            intensity = error_rates[i]
            color = (1.0, 1.0 - intensity * 0.7, 1.0 - intensity * 0.7)  # white to red
            rect = plt.Rectangle((bx, by), box_w, box_h, facecolor=color,
                                  edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(bx + box_w / 2, by + box_h / 2, labels[i],
                    ha="center", va="center", fontsize=8, fontweight="bold")

        # Draw arrows
        arrow_pairs = [
            (0.3, 0.4, rate_to_a2, f"{rate_to_a2*100:.0f}%"),
            (0.6, 0.7, rate_to_a3, f"{rate_to_a3*100:.0f}%"),
        ]
        for x_start, x_end, rate, label in arrow_pairs:
            width = max(0.5, rate * 5)
            ax.annotate("", xy=(x_end, 0.45), xytext=(x_start, 0.45),
                        arrowprops=dict(arrowstyle="->", lw=width, color="darkred", alpha=0.7))
            ax.text((x_start + x_end) / 2, 0.52, label,
                    ha="center", fontsize=9, color="darkred", fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"{inj_type.replace('_', ' ').title()}\n(n={n})", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Figure 5: Cross-Agent Error Propagation Flow", fontsize=12, y=0.98)
    plt.tight_layout()

    pdf_path = os.path.join(FIGURES_DIR, "fig5_cross_agent.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig5_cross_agent.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Figure 5 saved: {pdf_path} + {png_path}[/green]")


def figure_6_interceptor_bars(intercept_csv: str):
    """Hallucination & abstention rates per model × condition with bootstrap CIs."""
    from analysis.bootstrap import bootstrap_ci
    if not os.path.exists(intercept_csv):
        console.print(f"[red]Missing {intercept_csv}[/red]")
        return
    df = pd.read_csv(intercept_csv)
    models = sorted(df["model"].unique())
    conds = sorted(df["condition"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(models))
    width = 0.35

    for i, cond in enumerate(conds):
        halluc_pt, halluc_lo, halluc_hi = [], [], []
        abs_pt, abs_lo, abs_hi = [], [], []
        for m in models:
            sub = df[(df["model"] == m) & (df["condition"] == cond)]
            if len(sub) == 0:
                for arr in (halluc_pt, halluc_lo, halluc_hi, abs_pt, abs_lo, abs_hi):
                    arr.append(0)
                continue
            p, lo, hi = bootstrap_ci(sub["hallucination"].astype(int).values)
            halluc_pt.append(p); halluc_lo.append(p - lo); halluc_hi.append(hi - p)
            p, lo, hi = bootstrap_ci(sub["abstained"].astype(int).values)
            abs_pt.append(p); abs_lo.append(p - lo); abs_hi.append(hi - p)
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, halluc_pt, width, yerr=[halluc_lo, halluc_hi],
                    label=cond, capsize=3)
        axes[1].bar(x + offset, abs_pt, width, yerr=[abs_lo, abs_hi],
                    label=cond, capsize=3)

    axes[0].set_title("Hallucination Rate by Interceptor Condition")
    axes[0].set_ylabel("Hallucination rate")
    axes[1].set_title("Abstention Rate")
    axes[1].set_ylabel("Abstain rate")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_6_interceptor.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved Figure 6: {out}[/green]")
    plt.close()


def figure_7_ablation_bars(ablation_csv: str):
    """Ablation: hallucination reduction vs abstention cost by L-level."""
    from analysis.bootstrap import bootstrap_ci
    if not os.path.exists(ablation_csv):
        console.print(f"[red]Missing {ablation_csv}[/red]")
        return
    df = pd.read_csv(ablation_csv)
    conds = ["l1_only", "l1_l2", "l1_l2_l3"]
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(conds))
    width = 0.25
    for i, m in enumerate(models):
        halluc = []
        halluc_err = []
        for cond in conds:
            sub = df[(df["model"] == m) & (df["condition"] == cond)]
            if len(sub) == 0:
                halluc.append(0); halluc_err.append(0); continue
            p, lo, hi = bootstrap_ci(sub["hallucination"].astype(int).values)
            halluc.append(p); halluc_err.append(max(p - lo, hi - p))
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, halluc, width, yerr=halluc_err, label=m, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["L1 only", "L1+L2", "L1+L2+L3"])
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Interceptor Ablation by Check-Level Stack")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_7_ablation.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved Figure 7: {out}[/green]")
    plt.close()


def figure_9_seed_stability(seed_csv: str):
    """Seed-level mean with std bars — shows reproducibility."""
    if not os.path.exists(seed_csv):
        console.print(f"[red]Missing {seed_csv}[/red]")
        return
    df = pd.read_csv(seed_csv)
    groups = df.groupby(["model", "condition", "seed"]).agg(
        halluc=("hallucination", "mean"),
        eps=("eps", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in [(axes[0], "halluc", "Hallucination rate"),
                               (axes[1], "eps", "EPS")]:
        agg = groups.groupby(["model", "condition"])[metric].agg(["mean", "std"]).reset_index()
        labels = [f"{r['model']}\n{r['condition']}" for _, r in agg.iterrows()]
        x = np.arange(len(labels))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=5, color="steelblue", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"Seed-to-seed {label} (mean ± std across 3 seeds)")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "figure_9_seeds.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved Figure 9: {out}[/green]")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--input", type=str, default="data/results/p2_all.csv")
    parser.add_argument("--frontier", type=str, default="data/results/frontier_50.csv")
    parser.add_argument("--p3", type=str, default="data/results/p3_results.csv")
    parser.add_argument("--p4", type=str, default="data/results/p4_results.csv")
    parser.add_argument("--interceptor", type=str, default="data/results/interceptor_results.csv")
    parser.add_argument("--ablation", type=str, default="data/results/ablation_results.csv")
    parser.add_argument("--seeds", type=str, default="data/results/multi_seed_results.csv")
    parser.add_argument("--figs", type=str, default="1,2", help="Comma-separated figure numbers")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    p2_path = os.path.join(base, args.input)
    frontier_path = os.path.join(base, args.frontier)
    p3_path = os.path.join(base, args.p3)
    p4_path = os.path.join(base, args.p4)

    figs = [int(f.strip()) for f in args.figs.split(",")]

    p2_df = load_p2_data(p2_path) if os.path.exists(p2_path) else pd.DataFrame()
    frontier_df = load_frontier_data(frontier_path) if os.path.exists(frontier_path) else pd.DataFrame()
    p3_df = pd.read_csv(p3_path) if os.path.exists(p3_path) else pd.DataFrame()
    p4_df = pd.read_csv(p4_path) if os.path.exists(p4_path) else pd.DataFrame()

    if 1 in figs:
        if not p2_df.empty:
            figure_1_eps_heatmap(p2_df, frontier_df)
        else:
            console.print("[red]No P2 data for Figure 1[/red]")

    if 2 in figs:
        if not p2_df.empty:
            figure_2_cascade_by_domain(p2_df)

    if 3 in figs:
        if not p2_df.empty:
            figure_3_detection_vs_eps(p2_df)
        else:
            console.print("[red]No P2 data for Figure 3[/red]")

    if 4 in figs:
        if not p3_df.empty:
            figure_4_mtas_curves(p3_df)
        else:
            console.print("[red]No P3 data for Figure 4[/red]")

    if 5 in figs:
        if not p4_df.empty:
            figure_5_cross_agent(p4_df)
        else:
            console.print("[red]No P4 data for Figure 5[/red]")

    if 6 in figs:
        figure_6_interceptor_bars(os.path.join(base, args.interceptor))

    if 7 in figs:
        figure_7_ablation_bars(os.path.join(base, args.ablation))

    if 9 in figs:
        figure_9_seed_stability(os.path.join(base, args.seeds))

    console.print("\n[bold green]All requested figures generated.[/bold green]")


if __name__ == "__main__":
    main()
