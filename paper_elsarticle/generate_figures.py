#!/usr/bin/env python3
"""Generate all figures for the AgentProp-Bench paper submission."""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")
OUTDIR = os.path.join(BASE, "paper_elsarticle", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
CB_PALETTE = sns.color_palette("colorblind", 10)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def bootstrap_ci(arr, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def cohens_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else (1.0 if po >= 1 else 0.0)


def bootstrap_kappa(a, b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    kappas = []
    for _ in range(n):
        idx = rng.choice(len(a), len(a), replace=True)
        kappas.append(cohens_kappa(a[idx], b[idx]))
    return np.percentile(kappas, 2.5), np.percentile(kappas, 97.5)


MODEL_ORDER = [
    "gpt35_turbo", "gpt4o_mini", "gpt41_mini", "gpt41_nano",
    "gpt4o", "o3_mini", "gemini_20_flash", "gemini_25_flash", "deepseek_v3",
    "qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b",
]
MODEL_LABELS = {
    "gpt35_turbo": "GPT-3.5",
    "gpt4o_mini": "GPT-4o-mini",
    "gpt41_mini": "GPT-4.1-mini",
    "gpt41_nano": "GPT-4.1-nano",
    "gpt4o": "GPT-4o",
    "o3_mini": "o3-mini",
    "gemini_20_flash": "Gemini-2.0-Flash",
    "gemini_25_flash": "Gemini-2.5-Flash",
    "deepseek_v3": "DeepSeek-V3",
    "qwen25_3b": "Qwen2.5-3B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_14b": "Qwen2.5-14B",
    "hermes3_llama8b": "Hermes-3-Llama-8B",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_all = pd.read_csv(os.path.join(DATA, "p2_all_rejudged.csv"))
sw = df_all[df_all["error_type"] == "semantic_wrong"].copy()

human = pd.read_csv(os.path.join(DATA, "human_labeling_sheet_Bhaskar_labeled.csv"))
ctrl = pd.read_csv(os.path.join(DATA, "interceptor_control_600.csv"))
intc = pd.read_csv(os.path.join(DATA, "interceptor_heldout_600.csv"))

# Merge human labels with rejudged verdicts
hm = human.merge(
    df_all[["session_id", "ensemble_correct", "heuristic_correct",
            "gpt4o_verdict", "gemini_25_flash_verdict", "gpt4o_mini_verdict"]],
    on="session_id", how="left",
)

# ==========================================================================
# FIGURE 1 -- Stage reach per model (S1, S2, S3 bars)
# ==========================================================================
def fig_stage_reach():
    models_present = [m for m in MODEL_ORDER if m in sw["model"].unique()]
    n_models = len(models_present)

    rates = {m: {} for m in models_present}
    cis = {m: {} for m in models_present}

    for m in models_present:
        sub = sw[sw["model"] == m]
        n = len(sub)
        s1 = (sub["eps"] >= 1).astype(int).values
        s2 = (sub["eps"] >= 2).astype(int).values
        s3 = sub["ensemble_correct"].astype(int).values

        rates[m]["S1"] = s1.mean()
        rates[m]["S2"] = s2.mean()
        rates[m]["S3"] = s3.mean()

        cis[m]["S1"] = bootstrap_ci(s1)
        cis[m]["S2"] = bootstrap_ci(s2)
        cis[m]["S3"] = bootstrap_ci(s3)

    x = np.arange(n_models)
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (stage, color) in enumerate(zip(["S1", "S2", "S3"],
                                            [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2]])):
        vals = [rates[m][stage] for m in models_present]
        lo = [rates[m][stage] - cis[m][stage][0] for m in models_present]
        hi = [cis[m][stage][1] - rates[m][stage] for m in models_present]
        label = {"S1": r"$p_{S1}$ (eps $\geq$ 1)",
                 "S2": r"$p_{S2}$ (eps $\geq$ 2)",
                 "S3": r"$p_{S3}$ (ensemble correct)"}[stage]
        ax.bar(x + i * w, vals, w, yerr=[lo, hi], label=label,
               color=color, capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], rotation=35, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Model Stage-Reach Rates (semantic_wrong only)\n"
                 "[Human-calibrated ensemble judge for S3]")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.savefig(os.path.join(OUTDIR, "fig_stage_reach.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_stage_reach.png")


# ==========================================================================
# FIGURE 2 -- Rejection vs Recovery scatter
# ==========================================================================
def fig_rejection_vs_recovery():
    models_present = [m for m in MODEL_ORDER if m in sw["model"].unique()]

    rej_rates, rec_rates, n_s2_list = [], [], []
    labels = []

    for m in models_present:
        sub = sw[sw["model"] == m]
        s1_rate = (sub["eps"] >= 1).mean()
        rej = 1.0 - s1_rate

        s2_sub = sub[sub["eps"] >= 2]
        n_s2 = len(s2_sub)
        if n_s2 > 0:
            r23 = 1.0 - s2_sub["ensemble_correct"].mean()
            rec = 1.0 - r23
        else:
            rec = np.nan

        rej_rates.append(rej)
        rec_rates.append(rec)
        n_s2_list.append(max(n_s2, 5))
        labels.append(MODEL_LABELS.get(m, m))

    rej_rates = np.array(rej_rates)
    rec_rates = np.array(rec_rates)
    n_s2_arr = np.array(n_s2_list)

    mask = ~np.isnan(rec_rates)
    rho, pval = sp_stats.spearmanr(rej_rates[mask], rec_rates[mask])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(rej_rates, rec_rates,
                         s=n_s2_arr * 8, alpha=0.75,
                         c=[CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(rej_rates))],
                         edgecolors="black", linewidths=0.6)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (rej_rates[i], rec_rates[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)

    ax.set_xlabel("Rejection rate (1 - S1 rate)")
    ax.set_ylabel("Recovery rate (ensemble_correct | S2)")
    ax.set_title(f"Rejection vs Recovery (Spearman $\\rho$={rho:.3f}, p={pval:.3f})")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # legend for bubble size
    for sz_val in [10, 30, 60]:
        ax.scatter([], [], s=sz_val * 8, c="grey", alpha=0.5, edgecolors="black",
                   linewidths=0.5, label=f"n_S2 = {sz_val}")
    ax.legend(title="Point size = n(S2)", loc="lower right", fontsize=7)

    fig.savefig(os.path.join(OUTDIR, "fig_rejection_vs_recovery.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_rejection_vs_recovery.png")


# ==========================================================================
# FIGURE 3 -- Judge agreement (kappa vs human)
# ==========================================================================
def fig_judge_agreement():
    # Binary: human says correct = 1
    human_bin = (hm["YOUR_LABEL"] == "YES").astype(int).values

    judges = {}

    # Heuristic
    heur_bin = hm["heuristic_correct"].astype(int).values
    judges["Heuristic"] = (cohens_kappa(human_bin, heur_bin),
                           bootstrap_kappa(human_bin, heur_bin))

    # Individual LLM judges -- verdict=True means correct
    for col, name in [("gpt4o_mini_verdict", "GPT-4o-mini judge"),
                      ("gpt4o_verdict", "GPT-4o judge"),
                      ("gemini_25_flash_verdict", "Gemini-2.5-Flash judge")]:
        j_bin = hm[col].astype(int).values
        judges[name] = (cohens_kappa(human_bin, j_bin),
                        bootstrap_kappa(human_bin, j_bin))

    # Ensemble
    ens_bin = hm["ensemble_correct"].astype(int).values
    judges["Ensemble"] = (cohens_kappa(human_bin, ens_bin),
                          bootstrap_kappa(human_bin, ens_bin))

    # Order from bottom to top
    order = ["Heuristic", "GPT-4o-mini judge", "GPT-4o judge",
             "Gemini-2.5-Flash judge", "Ensemble"]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(order))
    kappas = [judges[k][0] for k in order]
    lo_err = [judges[k][0] - judges[k][1][0] for k in order]
    hi_err = [judges[k][1][1] - judges[k][0] for k in order]

    colors = [CB_PALETTE[i] for i in range(len(order))]
    ax.barh(y_pos, kappas, xerr=[lo_err, hi_err], capsize=4,
            color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel("Cohen's $\\kappa$ vs Human Labels")
    ax.set_title("Judge Agreement with Human Annotations (n=100)")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlim(-0.15, 0.75)

    # annotate values
    for i, k in enumerate(order):
        ax.text(kappas[i] + hi_err[i] + 0.02, i,
                f"{kappas[i]:.3f}", va="center", fontsize=8)

    fig.savefig(os.path.join(OUTDIR, "fig_judge_agreement.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_judge_agreement.png")


# ==========================================================================
# FIGURE 4 -- Confusion matrix (ensemble vs human)
# ==========================================================================
def fig_confusion_matrix():
    human_correct = (hm["YOUR_LABEL"] == "YES").values
    ens_correct = hm["ensemble_correct"].values

    # Cells: rows = Human (Y/N), cols = Ensemble (Y/N)
    cm = np.array([
        [(human_correct & ens_correct).sum(),
         (human_correct & ~ens_correct).sum()],
        [(~human_correct & ens_correct).sum(),
         (~human_correct & ~ens_correct).sum()],
    ])
    # Expected: [[19, 19], [6, 56]]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=60)

    labels_rc = ["Correct (Y)", "Incorrect (N)"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_rc)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels_rc)
    ax.set_xlabel("Ensemble Verdict")
    ax.set_ylabel("Human Label")

    n_total = cm.sum()
    for i in range(2):
        for j in range(2):
            rate = cm[i, j] / n_total
            ax.text(j, i, f"{cm[i, j]}\n({rate:.0%})",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > 30 else "black")

    # Marginals
    ax.set_title(f"Ensemble vs Human (n={n_total})\n"
                 f"Human Y={cm[0].sum()}, N={cm[1].sum()} | "
                 f"Ens Y={cm[:,0].sum()}, N={cm[:,1].sum()}")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.savefig(os.path.join(OUTDIR, "fig_confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_confusion_matrix.png")


# ==========================================================================
# FIGURE 5 -- Interceptor effect
# ==========================================================================
def fig_interceptor_effect():
    models_int = ["gpt4o_mini", "gemini_20_flash"]
    model_labels_int = {"gpt4o_mini": "GPT-4o-mini", "gemini_20_flash": "Gemini-2.0-Flash"}

    data_bars = {m: {} for m in models_int}
    data_ci = {m: {} for m in models_int}

    for m in models_int:
        # Control
        c = ctrl[ctrl["model"] == m]
        hallu_c = c["hallucination"].astype(int).values
        data_bars[m]["ctrl"] = hallu_c.mean()
        data_ci[m]["ctrl"] = bootstrap_ci(hallu_c)

        # Interceptor
        t = intc[intc["model"] == m]
        hallu_t = t["hallucination"].astype(int).values
        data_bars[m]["int"] = hallu_t.mean()
        data_ci[m]["int"] = bootstrap_ci(hallu_t)

        # Abstention
        abst = t["abstained"].astype(int).values
        data_bars[m]["abst"] = abst.mean()
        data_ci[m]["abst"] = bootstrap_ci(abst)

    x = np.arange(len(models_int))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_groups = [("ctrl", "Control halluc. rate", CB_PALETTE[0]),
                  ("int", "Interceptor halluc. rate", CB_PALETTE[1]),
                  ("abst", "Abstention rate", CB_PALETTE[2])]

    for i, (key, label, color) in enumerate(bar_groups):
        vals = [data_bars[m][key] for m in models_int]
        lo = [data_bars[m][key] - data_ci[m][key][0] for m in models_int]
        hi = [data_ci[m][key][1] - data_bars[m][key] for m in models_int]
        ax.bar(x + i * w, vals, w, yerr=[lo, hi], label=label,
               color=color, capsize=4, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([model_labels_int[m] for m in models_int])
    ax.set_ylabel("Rate")
    ax.set_title("Interceptor Effect on Hallucination & Abstention\n(600 tasks per arm)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, max(data_bars[m][k] for m in models_int for k in ["ctrl", "int", "abst"]) * 1.4)

    # Annotate null result for Gemini
    gem_ctrl = data_bars["gemini_20_flash"]["ctrl"]
    gem_int = data_bars["gemini_20_flash"]["int"]
    ax.annotate("n.s.", xy=(1 + 0.5 * w, max(gem_ctrl, gem_int) + 0.02),
                fontsize=9, ha="center", fontstyle="italic", color="grey")

    fig.savefig(os.path.join(OUTDIR, "fig_interceptor_effect.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_interceptor_effect.png")


# ==========================================================================
# FIGURE 6 -- Domain breakdown (supplementary)
# ==========================================================================
def fig_domain_breakdown():
    domains = sorted(sw["domain"].unique())
    n_dom = len(domains)

    rates = {d: {} for d in domains}
    cis_d = {d: {} for d in domains}

    for d in domains:
        sub = sw[sw["domain"] == d]
        s1 = (sub["eps"] >= 1).astype(int).values
        s2 = (sub["eps"] >= 2).astype(int).values
        s3 = sub["ensemble_correct"].astype(int).values

        rates[d]["S1"] = s1.mean()
        rates[d]["S2"] = s2.mean()
        rates[d]["S3"] = s3.mean()
        cis_d[d]["S1"] = bootstrap_ci(s1)
        cis_d[d]["S2"] = bootstrap_ci(s2)
        cis_d[d]["S3"] = bootstrap_ci(s3)

    x = np.arange(n_dom)
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (stage, color) in enumerate(zip(["S1", "S2", "S3"],
                                            [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2]])):
        vals = [rates[d][stage] for d in domains]
        lo = [rates[d][stage] - cis_d[d][stage][0] for d in domains]
        hi = [cis_d[d][stage][1] - rates[d][stage] for d in domains]
        label = {"S1": r"$p_{S1}$", "S2": r"$p_{S2}$", "S3": r"$p_{S3}$"}[stage]
        ax.bar(x + i * w, vals, w, yerr=[lo, hi], label=label,
               color=color, capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylabel("Rate")
    ax.set_title("Stage-Reach Rates by Domain (semantic_wrong)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    fig.savefig(os.path.join(OUTDIR, "fig_domain_breakdown.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_domain_breakdown.png")


# ==========================================================================
# FIGURE 7 -- Calibration (supplementary)
# ==========================================================================
def fig_calibration():
    human_correct = (hm["YOUR_LABEL"] == "YES").astype(int).values
    ens_correct = hm["ensemble_correct"].values

    # P(human=correct | ensemble=correct)
    ens_y_mask = ens_correct == True
    ens_n_mask = ens_correct == False
    p_h_given_ey = human_correct[ens_y_mask].mean()  # expected ~0.76
    p_h_given_en = human_correct[ens_n_mask].mean()  # expected ~0.25

    ci_ey = bootstrap_ci(human_correct[ens_y_mask])
    ci_en = bootstrap_ci(human_correct[ens_n_mask])

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = [p_h_given_ey, p_h_given_en]
    lo_err = [p_h_given_ey - ci_ey[0], p_h_given_en - ci_en[0]]
    hi_err = [ci_ey[1] - p_h_given_ey, ci_en[1] - p_h_given_en]
    x_pos = [0, 1]
    colors = [CB_PALETTE[0], CB_PALETTE[3]]

    ax.bar(x_pos, bars, 0.5, yerr=[lo_err, hi_err], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Ensemble = Correct", "Ensemble = Incorrect"])
    ax.set_ylabel("P(Human = Correct)")
    ax.set_title("Calibration: P(Human agrees | Ensemble verdict)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    for i in range(2):
        ax.text(x_pos[i], bars[i] + hi_err[i] + 0.03,
                f"{bars[i]:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.savefig(os.path.join(OUTDIR, "fig_calibration.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig_calibration.png")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig_stage_reach()
    fig_rejection_vs_recovery()
    fig_judge_agreement()
    fig_confusion_matrix()
    fig_interceptor_effect()
    fig_domain_breakdown()
    fig_calibration()
    print(f"\nAll figures saved to {OUTDIR}")
