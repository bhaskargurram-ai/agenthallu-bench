"""Write LaTeX booktabs tables and combined results.json for D_fabrication.
Usage: python3 -I D_make_tables.py <D_fabrication_dir>"""
import json, os, sys
D = sys.argv[1]
F = json.load(open(os.path.join(D, "fabrication_results.json")))
PL = json.load(open(os.path.join(D, "pilot_results.json")))
pc = lambda x: f"{100*x:.1f}"
ci = lambda c: f"[{100*c[0]:.1f}, {100*c[1]:.1f}]"
L = []
L.append("% Requires \\usepackage{booktabs}")
L.append(r"""\begin{table*}[t]
\centering\small
\caption{Fabricated tool use on P2 semantic-wrong traces for all 13 models (\%%, Wilson 95\%% CIs).
Fabricated = no tool output was obtained (verified against the step logs) \emph{and} the final answer asserts a tool-derived action or result.
\emph{Rule}: reconstructed rule-based classifier (reproduces the published rates closely);
\emph{LLM-adj.}: GPT-4.1-mini judge (temperature 0) with pure bracketed placeholders reclassified as not fabricated per the labeling guidance;
\emph{LLM}: raw judge label (a second judge, GPT-4o, agrees with $\kappa=%.2f$).
LLM-based columns are provisional pending human labels. $P(\text{fab}\mid\text{no call})$ uses the LLM-adj.\ label.}
\label{tab:fab_all}
\begin{tabular}{llrrrrrrr}
\toprule
Model & Type & $n$ & Tool-call & Published & Rule [95\%% CI] & LLM-adj. [95\%% CI] & LLM [95\%% CI] & $P(\text{fab}\mid\text{no call})$ \\
\midrule""" % F["classifier_validity"]["population"]["judge1_vs_judge2_kappa_no_tool"])
prev = None
for m, v in F["per_model"].items():
    if prev and prev != v["kind"]: L.append(r"\midrule")
    prev = v["kind"]
    a = v["judge1_adjusted_no_placeholder"]; nt = v["n"] - v["tool_call"]["k"]
    # conditional using adjusted label
    cond = f"{a['k']}/{nt}" if nt else "--"
    L.append(f"{v['name']} & {'open' if v['kind']=='open-weight' else 'prop.'} & {v['n']} & {pc(v['tool_call']['rate'])} & {v['published_pct']:.1f} & "
             f"{pc(v['rule']['rate'])} {ci(v['rule']['ci95'])} & {pc(a['rate'])} {ci(a['ci95'])} & "
             f"{pc(v['judge1_gpt41mini']['rate'])} {ci(v['judge1_gpt41mini']['ci95'])} & {cond} \\\\")
po = F["prop_vs_open"]
L.append(r"\midrule")
for key, lab in (("rule", "Rule"), ("j1adj", "LLM-adj.")):
    x = po[key]
    L.append(f"\\multicolumn{{9}}{{l}}{{{lab}: model-level mean prop.\\ {pc(x['model_level_mean_prop'])} vs open {pc(x['model_level_mean_open'])} "
             f"(diff {100*x['model_level_diff_open_minus_prop']:+.1f}\\,pp, exact permutation $p={x['model_level_exact_permutation_p']:.2f}$, 9 vs 4 models)}} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")
# classifier validity table
cv = F["classifier_validity"]["population"]
L.append(r"""\begin{table}[t]
\centering\small
\caption{Provisional validity of the rule-based fabrication classifier against two independent LLM judges over all 734 no-tool-call P2 traces
(tool-call traces are not fabricated by definition). Wilson 95\% CIs. Provisional pending human labels on the 150-trace stratified sample.}
\label{tab:fab_validity}
\begin{tabular}{lrrr}
\toprule
Reference & TP/FP/FN & Precision [95\% CI] & Recall [95\% CI] \\
\midrule""")
for k, lab in (("rule_vs_judge1", "GPT-4.1-mini judge"), ("rule_vs_judge2", "GPT-4o judge"), ("rule_vs_judge1_adjusted", "GPT-4.1-mini, adj."),
               ("rule_vs_judge1_proprietary", "\\quad proprietary only"), ("rule_vs_judge1_open-weight", "\\quad open-weight only")):
    x = cv[k]
    L.append(f"{lab} & {x['tp']}/{x['fp']}/{x['fn']} & {pc(x['precision'])} {ci(x['precision_ci95'])} & {pc(x['recall'])} {ci(x['recall_ci95'])} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
# pilot table
L.append(r"""\begin{table*}[t]
\centering\small
\caption{Paired simulated-vs-real tool pilot (80 tasks per model, same task order and seed; fabricated = no successful tool call and a confident, non-hedged answer).
Differences are real $-$ simulated in percentage points with Newcombe (method 10) paired CIs; McNemar exact test on discordant pairs ($b$: real-only, $c$: simulated-only).
Equivalence margin pre-stated at $\pm10$\,pp (TOST, $\alpha=0.05$ $\Leftrightarrow$ 90\% CI inside the margin). Pooled CIs use a task-clustered bootstrap.}
\label{tab:pilot}
\begin{tabular}{lrrrrrrrl}
\toprule
Model & $n$ & Sim. fab. [95\% CI] & Real fab. [95\% CI] & $\Delta$ (pp) & 95\% CI & 90\% CI & $b$/$c$, $p$ & TOST $\pm10$ \\
\midrule""")
for m, v in PL["per_model"].items():
    f = v["fabrication"]
    L.append(f"{m} & {f['n']} & {pc(f['sim_rate'])} {ci(f['sim_ci95'])} & {pc(f['real_rate'])} {ci(f['real_ci95'])} & {100*f['diff_real_minus_sim']:+.1f} & "
             f"{ci(f['diff_ci95'])} & {ci(f['diff_ci90'])} & {f['real_only']}/{f['sim_only']}, {f['mcnemar_exact_p']:.2f} & {'yes' if f['tost_equivalent_at_10pp'] else 'no'} \\\\")
f = PL["pooled"]["fabrication"]
L.append(r"\midrule")
L.append(f"Pooled (4 models) & {f['n']} & {pc(f['sim_rate'])} {ci(f['sim_ci95'])} & {pc(f['real_rate'])} {ci(f['real_ci95'])} & {100*f['diff_real_minus_sim']:+.1f} & "
         f"{ci(f['diff_ci95_task_cluster_bootstrap'])} & {ci(f['diff_ci90_task_cluster_bootstrap'])} & {f['real_only']}/{f['sim_only']}, {f['mcnemar_exact_p']:.2f} & {'yes' if f['tost_equivalent_at_10pp_cluster'] else 'no'} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")
open(os.path.join(D, "tables.tex"), "w").write("\n".join(L).replace("gpt-", "GPT-"))
json.dump({"fabrication": F, "real_tool_pilot": PL}, open(os.path.join(D, "results.json"), "w"), indent=2)
print("ok")
