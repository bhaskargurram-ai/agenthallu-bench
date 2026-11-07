import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""W_04: LaTeX tables for the JIIS manuscript generated from result files (writes JIIS_submission/tables/).
  tab_accuracy.tex  per-model baseline accuracy: validator vs heuristic vs ensemble on the same traces
                    (V_validator/results.json; Wilson CIs from W_writeup/W02_numbers.json)
  tab_paired.tex    paired effect by stratum and instrument (W_writeup/W01_results.json) + per-model validator rows
  tab_monitor.tex   monitor as selective prediction (C2_openweight/results.json via REPORT numbers, see below)
                    and validator re-scoring (W_writeup/W03_monitor_validator.json)
  tab_fabrication.tex  fabrication rates (D_fabrication/tables.tex values) + validator 'fabricated' code
Run: python3 -I W_04_tables.py
"""
import json
ROOT = _REPO + "/reanalysis"
T = ROOT + "/JIIS_submission/tables/"
NAME = {"o3_mini": "o3-mini", "deepseek_v3": "DeepSeek-V3", "gpt41_mini": "GPT-4.1-mini", "gpt41_nano": "GPT-4.1-nano",
        "gpt35_turbo": "GPT-3.5-Turbo", "gpt4o_mini": "GPT-4o-mini", "gemini_20_flash": "Gemini-2.0-Flash",
        "gpt4o": "GPT-4o", "gemini_25_flash": "Gemini-2.5-Flash"}
V = json.load(open(ROOT + "/V_validator/results.json"))["main"]
W1 = json.load(open(ROOT + "/W_writeup/W01_results.json"))
W2 = json.load(open(ROOT + "/W_writeup/W02_numbers.json"))
W3 = json.load(open(ROOT + "/W_writeup/W03_monitor_validator.json"))
f1 = lambda x: f"{x:.1f}"

# ---- accuracy
A = V["baseline_accuracy_by_model"]
ms = sorted([m for m in A if A[m].get("n")], key=lambda m: -A[m]["acc"])
L = [r"\begin{table}[!tbp]", r"\centering",
     r"\caption{Baseline accuracy (\%) of the seven proprietary models with stored baseline traces, scored by three instruments on the same verifiable traces. Validator: structured validator with Wilson 95\% CI and component pass rates ($T$ tools, $P$ parameters, $A$ answer faithfulness; $S$ is reported in the released results). Fab.: share of traces with outcome code \texttt{fabricated}. Heuristic and ensemble: the labels used in the earlier version of this study, restricted to the same traces. Kendall's $\tau$ between the validator ranking and the heuristic ranking is " + f"{W2['kendall_validator_vs_heuristic']:.2f}" + r"; with the ensemble ranking it is " + f"{W2['kendall_validator_vs_ensemble']:.2f}" + r".}",
     r"\label{tab:accuracy}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lrcrrrrrr@{}}", r"\toprule",
     r"Model & $n$ & Validator [95\% CI] & $T$ & $P$ & $A$ & Fab. & Heuristic & Ensemble \\", r"\midrule"]
for m in ms:
    a = A[m]; lo, hi = W2["validator_wilson"][m]
    fb = W1["baseline_outcomes_by_model"][m].get("fabricated", 0) / W1["baseline_outcomes_by_model"][m]["n"]
    L.append(f"{NAME[m]} & {a['n']} & {f1(100*a['acc'])} [{f1(lo)}, {f1(hi)}] & {f1(100*a['T_rate'])} & {f1(100*a['P_rate'])} & {f1(100*a['A_rate'])} & {f1(100*fb)} & {f1(100*a['old_heuristic_same_verifiable'])} & {f1(100*a['old_ensemble_same_verifiable'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(T + "tab_accuracy.tex", "w").write("\n".join(L) + "\n")

# ---- paired
def row(lab, s):
    if s["b"] + s["c"] == 0 and s["rd"] == 0 and s["err_base"] in (0.0, 100.0):
        pv = "--"
    else:
        pv = f"{s['p']:.1e}".replace("e-0", "e-").replace("e+00", "") if s["p"] < 0.001 else (f"{s['p']:.3f}" if s["p"] < 0.01 else f"{s['p']:.2f}")
        if "e" in pv:
            mant, ex = pv.split("e"); pv = f"${mant}\\times10^{{{int(ex)}}}$"
    return f"{lab} & {s['n']} & {f1(s['err_base'])} & {f1(s['err_inj'])} & {s['rd']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}] & {s['b']}/{s['c']} & {pv} \\\\"
L = [r"\begin{table}[!tbp]", r"\centering",
     r"\caption{Paired effect of injecting one schema-valid wrong argument into the first tool call. Each pair is the same task run by the same model without and with injection. Err.: error rate (\%) in the clean and injected arm; RD: risk difference in percentage points with a 95\% task-cluster bootstrap CI ($B=2000$); $b/c$: pairs wrong only when injected / wrong only when clean; $p$: exact McNemar test. The executed-cleanly stratum (the injected run executed at least one call and no call failed validation) is defined on the injected arm and is therefore descriptive. Under the validator the no-tool stratum is wrong in both arms by construction and is not an informative placebo; the LLM-judge rows provide that check.}",
     r"\label{tab:paired}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrlrr@{}}", r"\toprule",
     r"Instrument / stratum & $n$ & Err.\ clean & Err.\ inj. & RD [95\% CI] & $b/c$ & $p$ \\", r"\midrule"]
labs = {"validator": "Structured validator", "ensemble": "3-LLM ensemble", "gpt4o_mini": "GPT-4o-mini judge", "heuristic": "Substring heuristic"}
for j in ("validator", "ensemble", "gpt4o_mini", "heuristic"):
    L.append(r"\multicolumn{7}{@{}l}{\emph{" + labs[j] + r"}} \\")
    L.append(row(r"\quad All pairs", W1[j]["all"]))
    L.append(row(r"\quad Injected call executed cleanly", W1[j]["executed_clean"]))
    L.append(row(r"\quad No tool executed in either arm", W1[j]["placebo"]))
L.append(r"\midrule")
L.append(r"\multicolumn{7}{@{}l}{\emph{Structured validator, per model (all pairs)}} \\")
order = sorted(W1["validator"]["per_model"], key=lambda m: -W1["validator"]["per_model"][m]["rd"])
for m in order:
    L.append(row(r"\quad " + NAME[m], W1["validator"]["per_model"][m]))
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(T + "tab_paired.tex", "w").write("\n".join(L) + "\n")

# ---- monitor (validator re-scoring rows)
g = W3["gpt4o_mini"]; gm = W3["gemini_20_flash"]
open(T + "monitor_validator_numbers.tex", "w").write(
    "% generated by W_04 from W_writeup/W03_monitor_validator.json\n"
    f"\\newcommand{{\\MonValN}}{{{g['n']}}}\n\\newcommand{{\\MonValCtrlCorrect}}{{{g['control_correct']}}}\n"
    f"\\newcommand{{\\MonValIntCorrect}}{{{g['interceptor_correct']}}}\n\\newcommand{{\\MonValAbst}}{{{g['abstained']}}}\n"
    f"\\newcommand{{\\MonValCaught}}{{{g['abstained_control_wrong']}}}\n\\newcommand{{\\MonValPrec}}{{{g['precision']:.2f}}}\n"
    f"\\newcommand{{\\MonValBase}}{{{g['base_error']:.2f}}}\n\\newcommand{{\\MonValExp}}{{{g['expected_caught_random']:.0f}}}\n")
print(open(T + "tab_accuracy.tex").read()); print(open(T + "tab_paired.tex").read())
