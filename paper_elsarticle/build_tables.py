"""Emit per-table .tex files referenced by paper/main.tex.

Outputs to paper/tables/.
"""
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.bootstrap import bootstrap_ci

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)


def ci(arr, pct=True):
    import numpy as np
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return "--"
    m, lo, hi = bootstrap_ci(arr)
    if pct:
        return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write(name, content):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(content)


# ============ TABLE 1: Main baseline + P2 ============
b = pd.read_csv(os.path.join(BASE, "data/results/baseline_all.csv"))
p = pd.read_csv(os.path.join(BASE, "data/results/p2_all.csv"))
rows = []
for m in sorted(set(b["model"]) | set(p["model"])):
    bm = b[b["model"] == m]
    pm = p[p["model"] == m]
    rows.append((
        m, len(bm), len(pm),
        ci(bm["final_correct"].astype(int).values) if len(bm) else "--",
        ci(pm["final_correct"].astype(int).values) if len(pm) else "--",
        ci(pm["eps"].astype(float).values, pct=False) if len(pm) else "--",
    ))
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Baseline and P2-semantic-wrong results by model. $n_B$: baseline traces; $n_{P2}$: P2 traces. Values are point [95\% CI]. Correctness judged by 3-LLM ensemble.}",
    r"\label{tab:main}",
    r"\begin{tabular}{lrrccc}",
    r"\toprule",
    r"Model & $n_B$ & $n_{P2}$ & Baseline \% & P2 \% & P2 $\bar{\mathrm{EPS}}$ \\",
    r"\midrule",
]
for m, nb, npp, bc, pc, eps in rows:
    lines.append(f"{tex_escape(m)} & {nb} & {npp} & {bc} & {pc} & {eps} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table1_main.tex", "\n".join(lines))


# ============ TABLE 2: Hop-conditional rates ============
with open(os.path.join(BASE, "data/results/hop_rates.json")) as f:
    hop = json.load(f)
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Stage-reach probabilities $p_k=\Pr[S_k]$ and hop-conditional propagation rates $r_k=p_{k+1}/p_k$ under P2 semantic-wrong injection. 95\% CIs from 1,000-iter bootstrap.}",
    r"\label{tab:hop}",
    r"\begin{tabular}{lrccccc}",
    r"\toprule",
    r"Model & $n$ & $p_1$ \% & $p_2$ \% & $p_3$ \% & $r_1$ & $r_2$ \\",
    r"\midrule",
]
for m in sorted(hop.keys()):
    h = hop[m]
    def cell(d):
        return f"{100*d['point']:.1f} [{100*d['lo']:.1f}, {100*d['hi']:.1f}]"
    r1 = h.get("r_1_2"); r2 = h.get("r_2_3")
    lines.append(f"{tex_escape(m)} & {h['n']} & {cell(h['p_S1'])} & {cell(h['p_S2'])} & {cell(h['p_S3'])} & {r1:.2f} & {r2:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table2_hop.tex", "\n".join(lines))


# ============ TABLE 3: Judge agreement ============
with open(os.path.join(BASE, "data/results/ensemble_judge_metrics.json")) as f:
    ej = json.load(f)
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Per-rater Cohen's $\kappa$ agreement with 3-LLM ensemble majority on 960 baseline traces. Heuristic is substring/keyword match; LLM judges use identical prompts.}",
    r"\label{tab:judge}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Rater & $\kappa$ vs.\ majority & Interpretation \\",
    r"\midrule",
    f"Heuristic (substring) & \\textbf{{{ej['heuristic_kappa']:.3f}}} & slight \\\\",
    f"GPT-4o-mini & {ej['gpt4o_mini_kappa']:.3f} & substantial \\\\",
    f"GPT-4o & {ej['gpt4o_kappa']:.3f} & substantial \\\\",
    f"Gemini-2.5-Flash & \\textbf{{{ej['gemini_25_flash_kappa']:.3f}}} & almost perfect \\\\",
    r"\midrule",
    f"Unanimous 3/3 rate & \\multicolumn{{2}}{{l}}{{{100*ej['unanimous_rate']:.1f}\\% ($n$={ej['n']})}} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
write("table3_judge.tex", "\n".join(lines))


# ============ TABLE 4: Threshold sweep ============
with open(os.path.join(BASE, "analysis/interceptor_tuned_thresholds.json")) as f:
    sw = json.load(f)
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Interceptor threshold sweep on validation split. $\tau_{L1}$: parameter-error count threshold; $\tau_{L2}$: thought-keyword count threshold; L3: output-consistency layer on/off. Selected config in \textbf{bold}.}",
    r"\label{tab:sweep}",
    r"\begin{tabular}{cccccccc}",
    r"\toprule",
    r"$\tau_{L1}$ & $\tau_{L2}$ & L3 & $F_1$ & Prec & Recall & Abstain \% \\",
    r"\midrule",
]
best = sw["best_cfg"]
for s in sw["all_scores"]:
    c = s["cfg"]
    bold = (c == best)
    def fmt(x, _bold=bold):
        return f"\\textbf{{{x}}}" if _bold else str(x)
    f1s = f"{s['f1']:.3f}"
    ps = f"{s['precision']:.3f}"
    rs = f"{s['recall']:.3f}"
    abs_ = f"{100*s['abstain_pct']:.1f}"
    l3 = "on" if c["l3_enabled"] else "off"
    lines.append(
        f"{fmt(c['l1_threshold'])} & {fmt(c['l2_keywords'])} & {fmt(l3)} & "
        f"{fmt(f1s)} & {fmt(ps)} & {fmt(rs)} & {fmt(abs_)} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table4_sweep.tex", "\n".join(lines))


# ============ TABLE 5: Retail cross-domain ============
rb = pd.read_csv(os.path.join(BASE, "data/results/baseline_retail.csv"))
rp = pd.read_csv(os.path.join(BASE, "data/results/p2_retail.csv"))
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Cross-domain replication on held-out retail tasks (300 tasks). Model rank ordering on baseline correctness and P2 $\bar{\eps}$ is preserved relative to the original-domain results in Table~\ref{tab:main}.}",
    r"\label{tab:retail}",
    r"\begin{tabular}{lrccc}",
    r"\toprule",
    r"Model & $n$ & Baseline \% & P2 \% & P2 $\bar{\mathrm{EPS}}$ \\",
    r"\midrule",
]
for m in sorted(set(rb["model"]) | set(rp["model"])):
    bm = rb[rb["model"] == m]
    pm = rp[rp["model"] == m]
    lines.append(
        f"{tex_escape(m)} & {len(pm)} & "
        f"{ci(bm['final_correct'].astype(int).values)} & "
        f"{ci(pm['final_correct'].astype(int).values)} & "
        f"{ci(pm['eps'].astype(float).values, pct=False)} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("table5_retail.tex", "\n".join(lines))


# ============ TABLE A4: Power ============
power_path = os.path.join(BASE, "data/results/power_analysis.json")
if os.path.exists(power_path):
    with open(power_path) as f:
        pw = json.load(f)
    claims = pw.get("claims", [])
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Achieved statistical power for headline claims. $h$ = Cohen's effect size; 80\% is the conventional target. Both held-out interceptor comparisons (at $n\!=\!600$) now meet the threshold. Three residual rows remain under-powered (o3-mini P2 and the weather/medical domain contrasts) and are treated as directional in the text (see Limitations).}",
        r"\label{tab:power}",
        r"\begin{tabular}{p{5cm}rrccc}",
        r"\toprule",
        r"Claim & $n_A$ & $n_B$ & $h$ & Power & Underpwr? \\",
        r"\midrule",
    ]
    for c in claims:
        lines.append(
            f"{tex_escape(c['claim'])} & {c['n_a']} & {c['n_b']} & "
            f"{c['effect_h']:.2f} & {c['power']:.2f} & "
            f"{'yes' if c['underpowered'] else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tableA4_power.tex", "\n".join(lines))
else:
    write("tableA4_power.tex", "% No power_analysis.json found.\n")


print("All paper tables written to paper/tables/")
