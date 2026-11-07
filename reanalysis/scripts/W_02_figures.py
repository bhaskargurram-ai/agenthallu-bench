import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""W_02: figures for the JIIS manuscript (vector PDF). Reads only result files; writes JIIS_submission/figures/.
  fig_accuracy.pdf      baseline accuracy per model: validator (Wilson CI) vs substring heuristic vs 3-LLM ensemble,
                        all three on the same verifiable baseline traces (V_validator/results.json).
  fig_forest.pdf        paired injected-minus-baseline error difference per model, validator vs ensemble on the
                        same 1,085 pairs (W_writeup/W01_results.json).
  fig_outcomes.pdf      validator outcome codes, baseline vs injected arm, paired set (W01_results.json is not
                        per-pair, so this recomputes from V_validator/labels_all.csv with the X/W01 pairing).
  fig_risk_coverage.pdf selective risk vs coverage for GPT-4o-mini (C_interceptor/risk_coverage_data.json).
Also writes W_writeup/W02_numbers.json (Wilson CIs, Kendall tau, flip rates).
Run: python3 -I W_02_figures.py
"""
import json, csv, sys, math, collections
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
csv.field_size_limit(sys.maxsize)
ROOT = _REPO + "/reanalysis"
FIG = ROOT + "/JIIS_submission/figures"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "font.family": "DejaVu Sans"})
C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY = "#8a8985"
NAME = {"o3_mini": "o3-mini", "deepseek_v3": "DeepSeek-V3", "gpt41_mini": "GPT-4.1-mini", "gpt41_nano": "GPT-4.1-nano",
        "gpt35_turbo": "GPT-3.5-Turbo", "gpt4o_mini": "GPT-4o-mini", "gemini_20_flash": "Gemini-2.0-Flash"}

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

out = {}
# ---------------- fig_accuracy
V = json.load(open(ROOT + "/V_validator/results.json"))["main"]["baseline_accuracy_by_model"]
ms = sorted([m for m in V if V[m].get("n")], key=lambda m: V[m]["acc"])
fig, ax = plt.subplots(figsize=(6.2, 2.9))
for i, m in enumerate(ms):
    v = V[m]; lo, hi = wilson(v["correct"], v["n"])
    ax.plot([lo * 100, hi * 100], [i, i], color=C1, lw=2, solid_capstyle="round")
    ax.plot(v["acc"] * 100, i, "o", color=C1, ms=6.5, mec="white", mew=1.2, label="Structured validator (95% CI)" if i == 0 else None)
    ax.plot(v["old_heuristic_same_verifiable"] * 100, i, "s", color=C2, ms=6, mec="white", mew=1.2, label="Substring heuristic" if i == 0 else None)
    ax.plot(v["old_ensemble_same_verifiable"] * 100, i, "^", color=C3, ms=6.5, mec="white", mew=1.2, label="3-LLM ensemble" if i == 0 else None)
    out.setdefault("validator_wilson", {})[m] = [100 * lo, 100 * hi]
ax.set_yticks(range(len(ms))); ax.set_yticklabels([f"{NAME[m]} (n={V[m]['n']})" for m in ms])
ax.set_xlim(0, 100); ax.set_xlabel("Baseline accuracy on the same verifiable traces (%)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG + "/fig_accuracy.pdf"); plt.close(fig)
tv = [V[m]["acc"] for m in ms]
out["kendall_validator_vs_heuristic"] = kendalltau(tv, [V[m]["old_heuristic_same_verifiable"] for m in ms]).statistic
out["kendall_validator_vs_ensemble"] = kendalltau(tv, [V[m]["old_ensemble_same_verifiable"] for m in ms]).statistic

# ---------------- fig_forest
W = json.load(open(ROOT + "/W_writeup/W01_results.json"))
order = sorted(W["validator"]["per_model"], key=lambda m: W["validator"]["per_model"][m]["rd"])
rows = [("All pairs", W["validator"]["all"], W["ensemble"]["all"]),
        ("Injected call executed cleanly", W["validator"]["executed_clean"], W["ensemble"]["executed_clean"])]
rows += [(NAME[m], W["validator"]["per_model"][m], W["ensemble"]["per_model"][m]) for m in order]
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ys = list(range(len(rows)))[::-1]
for y, (lab, v, e) in zip(ys, rows):
    ax.plot(v["ci"], [y + 0.14] * 2, color=C1, lw=2, solid_capstyle="round")
    ax.plot(v["rd"], y + 0.14, "o", color=C1, ms=6.5, mec="white", mew=1.2)
    ax.plot(e["ci"], [y - 0.14] * 2, color=C3, lw=2, solid_capstyle="round")
    ax.plot(e["rd"], y - 0.14, "^", color=C3, ms=6.5, mec="white", mew=1.2)
ax.plot([], [], "o-", color=C1, label="Structured validator"); ax.plot([], [], "^-", color=C3, label="3-LLM ensemble (same pairs)")
ax.axvline(0, color=GREY, lw=0.8); ax.axhline(len(rows) - 2.5, color="#d0cfca", lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels([f"{lab} (n={v['n']})" for lab, v, _ in rows])
ax.set_xlabel("Injected minus clean error rate (percentage points)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG + "/fig_forest.pdf"); plt.close(fig)

# ---------------- fig_outcomes (paired set)
R = list(csv.DictReader(open(ROOT + "/V_validator/labels_all.csv")))
ok = lambda r: r["verifiable"].lower() in ("true", "1") and r["outcome"] != "infra_error"
B = {(r["task_id"], r["model"]): r for r in R if r["source"] == "baseline" and r["condition"] == "baseline" and ok(r)}
P = {}
for r in R:
    if r["condition"] in ("p2_semantic_wrong", "semantic_wrong") and ok(r):
        P.setdefault((r["task_id"], r["model"]), r)
keys = sorted(set(B) & set(P))
codes = ["correct", "wrong_params", "wrong_answer_values", "incomplete", "schema_error", "refused", "fabricated", "wrong_state"]
cols = [C3, C2, C4, GREY, C7, C5, C8, "#52514e"]
labels = ["correct", "wrong parameters", "wrong answer values", "incomplete", "schema error", "refused", "fabricated", "wrong state"]
mod = ["o3_mini", "deepseek_v3", "gpt41_mini", "gpt41_nano", "gpt35_turbo", "gpt4o_mini", "gemini_20_flash"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
yt, yl = [], []; y = 0
comp = {}
for m in mod[::-1]:
    km = [k for k in keys if k[1] == m]
    for arm, D, tag in (("injected", P, "inj."), ("clean", B, "clean")):
        cnt = collections.Counter(D[k]["outcome"] for k in km); n = len(km); left = 0
        comp[f"{m}|{arm}"] = dict(cnt)
        for c, col in zip(codes, cols):
            w = 100 * cnt.get(c, 0) / n
            if w > 0:
                ax.barh(y, w, left=left, color=col, height=0.78, edgecolor="white", linewidth=0.8)
            left += w
        yt.append(y); yl.append(f"{NAME[m]} {tag}"); y += 1
    y += 0.5
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5); ax.set_xlim(0, 100)
ax.set_xlabel("Share of paired traces (%)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols[:7]]
ax.legend(handles, labels[:7], ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.17), frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "/fig_outcomes.pdf"); plt.close(fig)
out["outcome_composition_paired"] = comp
bc = [k for k in keys if B[k]["outcome"] == "correct"]
fl = [k for k in bc if P[k]["outcome"] != "correct"]
ex = [k for k in keys if int(P[k]["n_exec"] or 0) > 0 and int(P[k]["n_schema_err"] or 0) == 0]
bce = [k for k in ex if B[k]["outcome"] == "correct"]; fle = [k for k in bce if P[k]["outcome"] != "correct"]
out["flip_all"] = [len(fl), len(bc), 100 * len(fl) / len(bc), [100 * x for x in wilson(len(fl), len(bc))]]
out["flip_executed_clean"] = [len(fle), len(bce), 100 * len(fle) / len(bce), [100 * x for x in wilson(len(fle), len(bce))]]
wp = [k for k in fl if P[k]["outcome"] == "wrong_params"]
out["wrong_params_flips"] = {"n": len(wp), "A_false": sum(1 for k in wp if P[k]["A"] == "False"),
                             "P_emitted_true": sum(1 for k in wp if P[k]["P_emitted"] == "True")}
out["flip_outcomes"] = dict(collections.Counter(P[k]["outcome"] for k in fl))

# ---------------- fig_risk_coverage
RC = json.load(open(ROOT + "/C_interceptor/risk_coverage_data.json"))
panels = [("clean_r1__scoreable", "(a) Original harness (no parameter schemas)"),
          ("clean_schema__scoreable", "(b) Harness exposing parameter schemas")]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
pts = [("ORIG (L1>=2 | L3; as run)", "o", "Monitor as run"), ("L1>=1", "s", "Execution-error layer only"), ("L3", "v", "Numeric-provenance layer only")]
for ax, (k, title) in zip(axes, panels):
    d = RC[k]
    ax.plot(d["cov"], d["risk_interceptor_score"], color=C1, lw=1.6, label="Monitor score ranking")
    ax.axhline(d["base_risk"], color=C2, lw=1.4, ls="--", label="Random abstention")
    ax.plot(d["cov"], d["oracle_risk"], color=GREY, lw=1, ls=":", label="Oracle")
    for name, mk, lab in pts:
        if name in d["points"]:
            c, r = d["points"][name]
            ax.plot(c, r, mk, color="#0b0b0b", mfc="white", ms=7, mew=1.3, label=lab)
    ax.set_title(title + f"\nn={len(d['cov'])}", fontsize=8)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 0.8); ax.set_xlabel("Coverage (fraction answered)")
    ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("Selective risk (wrong among answered)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(FIG + "/fig_risk_coverage.pdf"); plt.close(fig)
out["risk_coverage_points"] = {k: {**RC[k]["points"], "base_risk": RC[k]["base_risk"]} for k, _ in panels}

json.dump(out, open(ROOT + "/W_writeup/W02_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "outcome_composition_paired"}, indent=1, default=float))
