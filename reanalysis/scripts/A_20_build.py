"""Assemble results.json and LaTeX (booktabs) tables from the A_* JSON outputs.
Run order: A_02, A_04, A_05, A_10, A_11, A_12, A_13, then this script."""
import sys, json, os
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from A_common import OUT
from A_10_paired import MODEL_ORDER

L = lambda f: json.load(open(os.path.join(OUT, f)))
pr, rs, cal, ref, mech, s2t = L("paired_rd.json"), L("r23_spearman.json"), L("calibration.json"), L("reference_audit.json"), L("eps_mechanism.json"), L("s2_by_first_tool.json")
rerun = L("rerun_trace_mechanism.json") if os.path.exists(os.path.join(OUT, "rerun_trace_mechanism.json")) else None
NAMES = {"deepseek_v3": "DeepSeek-V3", "gpt35_turbo": "GPT-3.5-Turbo", "gpt41_mini": "GPT-4.1-mini", "gpt41_nano": "GPT-4.1-nano",
         "gpt4o_mini": "GPT-4o-mini", "gemini_20_flash": "Gemini-2.0-Flash", "o3_mini": "o3-mini", "qwen25_3b": "Qwen2.5-3B",
         "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "hermes3_llama8b": "Hermes-3-Llama-8B", "gpt4o": "GPT-4o",
         "gemini_25_flash": "Gemini-2.5-Flash"}

def _m(s): return s.replace("-", "$-$")
def pp(x): return _m(f"{100*x:+.1f}")
def ci(c): return _m(f"[{100*c[0]:+.1f}, {100*c[1]:+.1f}]")
def pv(p):
    if p < 1e-4: return "$<10^{-4}$"
    if p >= 0.001: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"

# ---------------- Table A: pooled paired RD
STR = [("all", "All paired traces"), ("placebo_both_no_tool_query_nonempty", "Placebo: no tool call in either arm$^{a}$"),
       ("applied", "Injector invoked (valid first tool call)"), ("applied_T0", "\\quad corrupted call executed, no error ($T{=}0$)"),
       ("applied_T1", "\\quad trace contains a tool/validator error ($T{=}1$)"), ("S1", "Paper's $S_1$ ($\\varepsilon\\geq1$)"),
       ("S2", "Paper's $S_2$ ($\\varepsilon\\geq2$)$^{b}$")]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Causal effect of semantic-wrong injection on final-answer error. Each P2 trace is paired 1:1 with the baseline (no-injection) run of the same model on the same task ($n=2{,}041$ pairs, 11 models; GPT-4o and Gemini-2.5-Flash have no baseline runs). RD $=P(\text{wrong}\mid\text{injected})-P(\text{wrong}\mid\text{not injected})$ in percentage points; 95\% CI: Newcombe hybrid-score interval for paired proportions; $p$: exact McNemar test. Strata are defined on the P2 trace.}",
         r"\label{tab:paired_rd}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l r r r l r r l r l}", r"\toprule",
         r" & & \multicolumn{4}{c}{Ensemble judge (primary)} & \multicolumn{2}{c}{GPT-4o-mini judge} & \multicolumn{2}{c}{Heuristic} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Stratum & $n$ & Inj.\% & Base\% & RD [95\% CI] & $p$ & RD [95\% CI] & $p$ & RD [95\% CI] & $p$ \\", r"\midrule"]
for k, lab in STR:
    e, g, h = pr["pooled"]["ensemble"][k], pr["pooled"]["gpt4o_mini"][k], pr["pooled"]["heuristic"][k]
    lines.append(f"{lab} & {e['n']} & {100*e['p_wrong_inj']:.1f} & {100*e['p_wrong_base']:.1f} & {pp(e['RD'])} {ci(e['RD_ci_newcombe'])} & {pv(e['mcnemar_exact_p'])} & "
                 f"{pp(g['RD'])} {ci(g['RD_ci_newcombe'])} & {pv(g['mcnemar_exact_p'])} & {pp(h['RD'])} {ci(h['RD_ci_newcombe'])} & {pv(h['mcnemar_exact_p'])} \\\\")
tok = pr["verbatim_injected_token"]
lines += [r"\midrule", r"\multicolumn{10}{l}{\emph{Judge-free check: final answer contains the injected substitute value (e.g.\ the swapped city), injected vs.\ baseline}} \\"]
for k, lab in [("all", "All paired traces"), ("applied_T0", "\\quad corrupted call executed, no error ($T{=}0$)"), ("applied_T1", "\\quad trace contains an error ($T{=}1$)"), ("S2", "Paper's $S_2$")]:
    v = tok[k]
    lines.append(f"{lab} & {v['n']} & {100*v['p_wrong_inj']:.1f} & {100*v['p_wrong_base']:.1f} & {pp(v['RD'])} {ci(v['RD_ci_newcombe'])} & {pv(v['mcnemar_exact_p'])} & & & & \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip\footnotesize $^{a}$Neither arm executed a tool, so the injector could not act; rows whose P2 re-judging received an empty question are excluded (with them: RD " + pp(pr['pooled']['ensemble']['placebo_both_no_tool']['RD']) + r" pp, " + ci(pr['pooled']['ensemble']['placebo_both_no_tool']['RD_ci_newcombe']) + r"). "
          r"$^{b}$$S_2$ requires the heuristic to mark the P2 answer wrong (Sec.~\ref{sec:eps_def}), so the heuristic RD in this row is inflated by construction and the ensemble RD is subject to the same selection. "
          r"Pooled task-cluster bootstrap CIs (tasks recur across models) are in the supplementary JSON and agree to within 1 pp.",
          r"\end{table}"]
open(os.path.join(OUT, "tab_paired_rd.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- Table B: per-model paired RD (ensemble)
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Per-model paired risk difference (ensemble judge): all paired traces; the stratum in which the corrupted call actually executed without error ($T{=}0$); and the paper's $S_2$ stratum (selected on heuristic failure, see Sec.~\ref{sec:eps_def}). Newcombe 95\% CIs (pp); exact McNemar $p$. Single-digit $n$ cells are uninformative.}",
         r"\label{tab:paired_rd_model}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r l r r l r r l r}", r"\toprule",
         r"Model & $n$ & RD all [95\% CI] & $p$ & $n_{T=0}$ & RD$_{T=0}$ [95\% CI] & $p$ & $n_{S_2}$ & RD$_{S_2}$ [95\% CI] & $p$ \\", r"\midrule"]
for m in MODEL_ORDER:
    if m not in pr["per_model"]: continue
    a, t = pr["per_model"][m]["ensemble"]["all"], pr["per_model"][m]["ensemble"]["applied_T0"]
    tt = (f"{pp(t['RD'])} {ci(t['RD_ci_newcombe'])} & {pv(t['mcnemar_exact_p'])}") if t["n"] else "--- & ---"
    s2 = pr["per_model"][m]["ensemble"]["S2"]
    ss = (f"{pp(s2['RD'])} {ci(s2['RD_ci_newcombe'])} & {pv(s2['mcnemar_exact_p'])}") if s2["n"] else "--- & ---"
    lines.append(f"{NAMES[m]} & {a['n']} & {pp(a['RD'])} {ci(a['RD_ci_newcombe'])} & {pv(a['mcnemar_exact_p'])} & {t['n']} & {tt} & {s2['n']} & {ss} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_paired_rd_per_model.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- Table C: per-model r23 with CIs
r23e, r23g = rs["r23"]["ensemble"], rs["r23"]["gpt4o_mini"]; calb = cal["calibrated"]["bhaskar"]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Per-model $r_{23}=P(S_3\mid S_2)$ with 95\% Wilson and bootstrap ($B{=}10{,}000$, percentile) intervals, ensemble judge; GPT-4o-mini-judge point estimate; and human-calibrated value with an interval that propagates calibration uncertainty (joint bootstrap over the 100 human labels and the $S_2$ traces). Admission $=P$(first action is a valid tool call). Note: $S_2$ is not a clean ``injection executed'' stage (Sec.~\ref{sec:eps_def}).}",
         r"\label{tab:r23_ci}", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l r r r r r l l r l}", r"\toprule",
         r"Model & $n$ & Adm.\% & $S_1$\% & $S_2$\% & $n_{S_2}$ & $r_{23}$ [Wilson] & [Bootstrap] & $r_{23}^{\text{4o-mini}}$ & $r_{23}^{\text{cal}}$ [95\% CI] \\", r"\midrule"]
for m in MODEL_ORDER:
    v, g, c = r23e[m], r23g[m], calb.get(f"r23|{m}")
    lines.append(f"{NAMES[m]} & {v['n']} & {100*v['admission']:.1f} & {100*v['p_S1']:.1f} & {100*v['p_S2']:.1f} & {v['n_S2']} & "
                 f"{v['r23']:.2f} [{v['r23_wilson'][0]:.2f}, {v['r23_wilson'][1]:.2f}] & [{v['r23_boot'][0]:.2f}, {v['r23_boot'][1]:.2f}] & {g['r23']:.2f} & "
                 f"{c['calibrated']:.2f} [{c['ci_labels_and_traces'][0]:.2f}, {c['ci_labels_and_traces'][1]:.2f}] \\\\")
P = r23e["_pooled"]; Pg = r23g["_pooled"]; Pc = calb["pooled_r23"]
lines += [r"\midrule",
          f"Pooled & 2150 & & & & {P['n_S2']} & {P['r23']:.2f} [{P['wilson'][0]:.2f}, {P['wilson'][1]:.2f}] & [{P['task_cluster_boot'][0]:.2f}, {P['task_cluster_boot'][1]:.2f}]$^{{c}}$ & {Pg['r23']:.2f} & {Pc['calibrated']:.2f} [{Pc['ci_labels_and_traces'][0]:.2f}, {Pc['ci_labels_and_traces'][1]:.2f}] \\\\",
          r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize $^{c}$Task-cluster bootstrap. Under the heuristic judge $r_{23}\equiv1$ for every model by construction.", r"\end{table}"]
open(os.path.join(OUT, "tab_r23_ci.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- EPS formal definition paragraph
sp = rs["spearman"]
eps_tex = r"""\paragraph{Formal definition of the EPS stage indicators as implemented.}\label{sec:eps_def}
Let $\tau$ be a trace of the ReAct loop with at most five model turns. Let $\mathcal{T}$ be the tool registry.
Define three binary trace properties:
(i)~\emph{admission} $A(\tau)=1$ iff the agent's first emitted \textsf{Action} names a tool in $\mathcal{T}$ \emph{and} the same completion does not contain ``Final Answer:'' (that branch takes precedence and no tool is executed). Only when $A(\tau)=1$ is the injector invoked; it corrupts the parameters of that first call, immediately before validation and execution, so the agent never has an opportunity to inspect or reject the corrupted arguments.
(ii)~\emph{error observation} $D(\tau)=1$ iff some \textsf{Observation} in $\tau$ (from any call, injected or not) carries a non-empty schema-validator report or a tool return value with an \texttt{error} key.
(iii)~\emph{heuristic failure} $W_h(\tau)=1$ iff the run-time string-match check fails, i.e.\ neither the first 20 characters of the reference occur in the first 200 characters of the answer nor any of the first three reference words longer than three characters occurs in the answer (case-insensitive).
The scorer (\texttt{metrics/eps\_scorer.py}) then yields
\[
\varepsilon(\tau)=A(\tau)\,\bigl[D(\tau)+W_h(\tau)\bigr]\in\{0,1,2\},\qquad
\varepsilon\geq1 \iff A\wedge(D\vee W_h),\qquad
\varepsilon\geq2 \iff A\wedge D\wedge W_h .
\]
(The scorer's literal search for the injected value never fires, because the injected parameter dictionary is rendered as a Python \texttt{repr} and compared with JSON-serialised fields; the planning stage precedes the injection origin and is never counted.) This identity holds for all 2{,}150 semantic-wrong traces with zero exceptions; independently, the detectability weight in $w\varepsilon$ equals 0.6 exactly when $D=1$.
Two consequences follow. First, $1-P(\varepsilon\geq1)$ is not an input-rejection rate: it is dominated by $1-P(A)$, the fraction of traces in which no executable tool call was emitted (""" + f"{mech['total']['no_tool']}" + r""" of 2{,}150 traces made no tool call at all, mostly because the model wrote an \textsf{Action} and a template ``Final Answer:'' in the same completion), plus admitted traces with $D=0$ and a heuristic pass. Second, $\varepsilon\geq2$ is not an ``injection executed visibly'' stage. A semantic-wrong substitution preserves parameter names and types, so it cannot cause a validator error; $D=1$ instead arises from the agent's own malformed calls (the prompt lists tool names but not parameter schemas, e.g.\ \texttt{\{"param": ...\}} or \texttt{duration} for \texttt{duration\_minutes}). When the injected first call fails validation, the corrupted values are never executed, and the agent typically retries with correct, un-injected parameters. $S_2$ therefore selects traces in which the injection was most often \emph{inert}, and additionally conditions on the heuristic judging the answer wrong ($W_h=1$), which makes $r_{23}=P(S_3\mid S_2)$ partly an agreement rate between the heuristic and the ensemble judge (and identically 1 under the heuristic judge).
"""
open(os.path.join(OUT, "eps_definitions.tex"), "w").write(eps_tex)

# ---------------- flip rates: P(injected wrong | baseline right) with Wilson CIs (derived from paired counts)
from A_stats import wilson
flip = {}
for j in ["ensemble", "gpt4o_mini", "heuristic"]:
    flip[j] = {}
    for st in ["all", "placebo_both_no_tool_query_nonempty", "applied", "applied_T0", "applied_T1", "S1", "S2"]:
        v = pr["pooled"][j][st]; b, d, a, c = v["inj_wrong_base_right"], v["both_right"], v["both_wrong"], v["inj_right_base_wrong"]
        flip[j][st] = {"right_to_wrong": b, "base_right": b + d, "rate": b / (b + d) if b + d else None, "wilson": list(wilson(b, b + d)) if b + d else None,
                       "wrong_to_right": c, "base_wrong": a + c, "rate_wrong_to_right": c / (a + c) if a + c else None}
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Injection-induced failure among tasks the same model solves without injection: $P(\text{wrong}\mid\text{injected},\ \text{baseline correct})$, paired traces, Wilson 95\% CIs. The placebo stratum (no tool executed in either arm) measures run-to-run and re-judging noise.}",
         r"\label{tab:flip}", r"\begin{tabular}{l r l r l}", r"\toprule",
         r"Stratum & \multicolumn{2}{c}{Ensemble judge} & \multicolumn{2}{c}{GPT-4o-mini judge} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & $k/n$ & rate [95\% CI] & $k/n$ & rate [95\% CI] \\", r"\midrule"]
for st, lab in [("placebo_both_no_tool_query_nonempty", "Placebo (no tool call in either arm)"), ("all", "All paired traces"),
                ("applied_T1", "Injector invoked, trace has an error ($T{=}1$)"), ("applied_T0", "Corrupted call executed, no error ($T{=}0$)"), ("S2", "Paper's $S_2$")]:
    e, g = flip["ensemble"][st], flip["gpt4o_mini"][st]
    lines.append(f"{lab} & {e['right_to_wrong']}/{e['base_right']} & {e['rate']:.2f} [{e['wilson'][0]:.2f}, {e['wilson'][1]:.2f}] & {g['right_to_wrong']}/{g['base_right']} & {g['rate']:.2f} [{g['wilson'][0]:.2f}, {g['wilson'][1]:.2f}] \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_flip_rates.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- results.json
results = {
    "description": "AgentProp-Bench revision, workstream A (propagation statistics). All numbers computed from data/results/*.csv by scripts/A_*.py.",
    "data_files": {"p2": "data/results/p2_all_rejudged.csv (contains all 13 models; the four open-weight p2_*_rejudged.csv files are identical subsets, verified cell-by-cell)",
                   "baseline": "data/results/baseline_all_rejudged.csv (same; open-weight baseline files are identical subsets)",
                   "human_labels": ["data/results/human_labeling_sheet_Bhaskar_labeled.csv", "data/results/human_labeling_sheet_Yoshitha_labeled.csv"]},
    "eps_mechanism": mech, "s2_by_first_tool": s2t, "rerun_trace_mechanism_workstreamC_gpt4o_mini": rerun,
    "paired_risk_difference": pr, "flip_rates_injected_wrong_given_baseline_right": flip, "r23_and_spearman": rs, "calibration": cal, "reference_audit": ref}
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
print("wrote results.json and tables")
