import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""V_05: merge all validator outputs into V_validator/results.json and write LaTeX booktabs tables."""
import json
import os

OUT = _REPO + "/reanalysis/V_validator"
J = lambda f: json.load(open(f"{OUT}/{f}"))

MODEL_NAMES = {"deepseek_v3": "DeepSeek-V3", "gemini_20_flash": "Gemini-2.0-Flash", "gpt35_turbo": "GPT-3.5-Turbo",
               "gpt41_mini": "GPT-4.1-mini", "gpt41_nano": "GPT-4.1-nano", "gpt4o_mini": "GPT-4o-mini",
               "o3_mini": "o3-mini", "qwen25_3b": "Qwen2.5-3B", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
               "hermes3_llama8b": "Hermes-3-Llama-8B", "mistral_7b": "Mistral-7B", "phi35_mini": "Phi-3.5-mini",
               "gemini_15_flash": "Gemini-1.5-Flash", "gemini_20_flash_lite": "Gemini-2.0-Flash-Lite"}


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def main():
    main_ = J("results_main.json")
    hum = J("human_agreement.json")
    adj = J("adjudication_summary.json")
    rob = J("robustness.json")
    res = {"validator_version": main_["validator_version"], "main": main_, "human_agreement": hum,
           "adjudication": adj, "robustness": rob}
    # agreement with the adjudicated reference (human majority, with the adjudicated disagreements replaced)
    n = hum["n_verifiable"]
    agree = n - adj["n_disagreements"] + adj["verdicts"].get("validator", 0)
    res["agreement_with_adjudicated_reference"] = {"n": n, "agree": agree, "rate": agree / n,
                                                   "note": "ambiguous cases counted as disagreements"}
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    # ---------------- Table V1: validator definition
    t1 = r"""\begin{table}[t]
\centering
\caption{Structured correctness validator (V1.0). All components are computed programmatically from the query-grounded
gold specification, deterministic simulator replay and the logged trace; no LLM or human judgement is involved.}
\label{tab:validator-def}
\small
\begin{tabular}{@{}lp{0.74\linewidth}@{}}
\toprule
Component & Definition \\
\midrule
$T$ (tools) & Every required gold call is matched one-to-one by an \emph{executed} (schema-valid) call of the same tool; per-template order constraints hold (e.g.\ \texttt{get\_events} before a conditional \texttt{create\_event}). \\
$P$ (params) & Every matched call agrees with the gold call on every scored slot, using the parameters actually executed (post-injection). Normalisation: case/whitespace/accents, ISO dates, 24h times, unordered sets (attendees, drug pairs), enum spelling; celsius call + correct $^\circ$F answer accepted for Fahrenheit queries. \\
$S$ (state) & Write tasks only. Multiset of state-changing successful writes equals the gold writes (strict, $\tau$-bench style); conditional writes (slot free, in stock, cheaper item) are resolved from the simulator or the received observation; a gold delete is satisfied by an executed attempt with the gold id. \\
$A$ (answer) & Typed values in the final answer (temperatures, humidity, sky condition, IDs, prices, dates, times, drugs, conditions, severity, product and event names) vs the \emph{gold} observation: recall of required key values $=1$ and precision of typed mentions $=1$ (derived aggregates and unit conversions are grounded, tolerance $\pm1^\circ$); write-outcome claims must match $S$. \\
CORRECT & $T\wedge P\wedge S\wedge A$ ($S$ vacuous for read-only tasks; for write-only tasks $A$ reduces to outcome-claim faithfulness and precision). \\
\midrule
Outcome code & First matching rule: \texttt{infra\_error} (runner/API error text) $\to$ \texttt{correct} $\to$ \texttt{fabricated} (asserts values/success with no executed call, or claims a write that never succeeded) $\to$ \texttt{schema\_error} (all calls failed validation) $\to$ \texttt{refused} (inability/clarification, no gold content) $\to$ \texttt{incomplete} ($\neg T$) $\to$ \texttt{wrong\_params} ($\neg P$) $\to$ \texttt{wrong\_state} ($\neg S$) $\to$ \texttt{wrong\_answer\_values} ($\neg A$). Tasks without recoverable gold are \texttt{not\_verifiable} and excluded. \\
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{OUT}/tab_V1_validator_definition.tex", "w").write(t1)

    # ---------------- Table V2: agreement with humans
    ci = lambda a: f"[{a[0]:.2f}, {a[1]:.2f}]"
    cm = hum["confusion_vs_majority"]
    t2 = rf"""\begin{{table}}[t]
\centering
\caption{{Agreement of the structured validator with the three human annotators on the {hum['n_total']} labelled P2
(semantic\_wrong) traces; {hum['n_not_verifiable']} traces are not verifiable (underspecified queries) and excluded,
leaving $n={hum['n_verifiable']}$. 95\% CIs: 10{{,}}000-sample bootstrap. Humans saw only the descriptive reference;
every disagreement was adjudicated against the gold simulator observations.}}
\label{{tab:validator-human}}
\small
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
Comparison & Cohen's $\kappa$ [95\% CI] & Raw agreement \\
\midrule
Validator vs human majority & {hum['kappa_vs_majority']:.2f} {ci(hum['kappa_vs_majority_ci95'])} & {pct(hum['raw_agreement_vs_majority'])}\% \\
Validator vs annotator 1 & {hum['kappa_vs_bhaskar']:.2f} {ci(hum['kappa_vs_bhaskar_ci95'])} & -- \\
Validator vs annotator 2 & {hum['kappa_vs_yoshitha']:.2f} {ci(hum['kappa_vs_yoshitha_ci95'])} & -- \\
Validator vs annotator 3 & {hum['kappa_vs_third']:.2f} {ci(hum['kappa_vs_third_ci95'])} & -- \\
Human vs human (range of 3 pairs) & {min(hum['kappa_human_bhaskar_yoshitha'], hum['kappa_human_bhaskar_third'], hum['kappa_human_yoshitha_third']):.2f}--{max(hum['kappa_human_bhaskar_yoshitha'], hum['kappa_human_bhaskar_third'], hum['kappa_human_yoshitha_third']):.2f} & -- \\
\midrule
\multicolumn{{3}}{{@{{}}l}}{{Confusion (validator/majority): YES/YES {cm['both_yes']}, YES/NO {cm['validator_yes_human_no']}, NO/YES {cm['validator_no_human_yes']}, NO/NO {cm['both_no']}}} \\
\multicolumn{{3}}{{@{{}}l}}{{Adjudication of the {adj['n_disagreements']} disagreements: validator right {adj['verdicts'].get('validator', 0)}, human right {adj['verdicts'].get('human', 0)}, ambiguous {adj['verdicts'].get('ambiguous', 0)}}} \\
\multicolumn{{3}}{{@{{}}l}}{{\quad ({adj['n_disagreements_where_agent_executed_no_tool']} disagreements are traces in which the agent executed no tool call at all)}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(f"{OUT}/tab_V2_validator_vs_human.tex", "w").write(t2)

    # ---------------- Table V3: per-model accuracy
    base = main_["baseline_accuracy_by_model"]
    lines = []
    for m, d in sorted(base.items(), key=lambda kv: -(kv[1]["acc"] or -1)):
        if not d["n"]:
            continue
        oc = d["outcomes"]
        fab = oc.get("fabricated", 0) / d["n"]
        lines.append(f"{MODEL_NAMES.get(m, m)} & {d['n']} & {pct(d['acc'])} [{pct(d['ci95'][0])}, {pct(d['ci95'][1])}] & "
                     f"{pct(d['T_rate'])} & {pct(d['P_rate'])} & {pct(d['S_rate_write_tasks'])} & {pct(d['A_rate'])} & "
                     f"{pct(fab)} & {pct(d.get('old_heuristic_all_sessions'))} & {pct(d.get('old_ensemble_all_sessions'))} \\\\")
    excl = [MODEL_NAMES.get(m, m) for m, d in base.items() if not d["n"]]
    ao = main_["answer_only_accuracy"]
    lines_ao = []
    for m in ["qwen25_7b", "qwen25_3b", "qwen25_14b", "hermes3_llama8b", "phi35_mini", "mistral_7b"]:
        d = ao.get(f"open_baseline|{m}")
        if d:
            lines_ao.append(f"{MODEL_NAMES[m]} & {d['n_scored']} & \\multicolumn{{6}}{{c}}{{$A$-only (answer-only mode): "
                            f"{pct(d['acc'])} [{pct(d['ci95'][0])}, {pct(d['ci95'][1])}]}} & & \\\\")
    cal = main_["answer_only_calibration_on_proprietary_baseline"]
    t3 = rf"""\begin{{table*}}[t]
\centering
\caption{{Baseline accuracy under the structured validator (verifiable tasks only; knowledge and underspecified tasks
excluded). $T,P,A$: component pass rates; $S$: pass rate on write tasks; Fab.: share of traces coded
\texttt{{fabricated}}. Old: the paper's string-heuristic and 3-LLM-ensemble labels on all 450 sessions of the same model
(incl.\ knowledge tasks). Open-weight models have no stored traces (only final answers), so only the answer-only
component can be computed (read tasks with deterministic tools); on proprietary traces answer-only $A$ over-estimates
full CORRECT ({pct(cal['AO_rate'])}\% vs {pct(cal['CORRECT_rate'])}\% on the same traces), so the two blocks are not comparable.
{('Excluded (all runs are API errors): ' + ', '.join(excl) + '.') if excl else ''}}}
\label{{tab:validator-models}}
\small
\begin{{tabular}}{{@{{}}lrccccccrr@{{}}}}
\toprule
Model & $n$ & CORRECT [95\% CI] & $T$ & $P$ & $S$ & $A$ & Fab. & Old heur. & Old ens. \\
\midrule
{chr(10).join(lines)}
\midrule
{chr(10).join(lines_ao)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    open(f"{OUT}/tab_V3_per_model_accuracy.tex", "w").write(t3)

    # ---------------- Table V4: robustness
    rv = rob["variants"]
    names = {"default": "Default (V1.0)", "no_casefold": "No case folding", "no_accent_fold": "No accent folding",
             "no_date_norm": "No date normalisation", "no_time_norm": "No time normalisation",
             "temp_tol_0": r"Temp.\ tolerance 0", "temp_tol_0.5": r"Temp.\ tolerance 0.5",
             "temp_tol_2": r"Temp.\ tolerance 2", "num_tol_1": r"Numeric tolerance $\pm1$",
             "no_unit_conversion": r"No $^\circ$C/$^\circ$F conversion", "no_derived_aggregates": "No derived aggregates",
             "precision_ignored": "Precision ignored", "precision_ge_0.8": r"Precision $\ge 0.8$",
             "recall_ge_0.5": r"Recall $\ge 0.5$", "state_lenient": "Lenient state (subset)",
             "no_equivalent_obs": "No equivalent-call observations", "no_condition_synonyms": "No sky-condition synonyms",
             "hist_lenient": "Lenient `last January' range", "no_aux_obs": "No auxiliary on-task observations",
             "title_lenient": "Lenient event titles", "gold_from_taskfile": "Gold = raw task-file sequence"}
    rl = []
    for k, lab in names.items():
        d = rv[k]
        rl.append(f"{lab} & {pct(d['baseline_pooled'])} & {pct(d['p2_pooled'])} & {pct(d['flip_rate_vs_default'])} & "
                  f"{d['kappa_vs_human_majority']:.2f} & {d['kendall_tau_model_ranking_vs_default']:.2f} \\\\")
        if k == "title_lenient":
            rl.append(r"\midrule")
    t4 = rf"""\begin{{table}}[t]
\centering
\caption{{Sensitivity of validator scores to each normalisation/rubric choice (one switch at a time). Accuracy pooled
over verifiable baseline / P2 traces; flips: share of CORRECT labels that change vs default (baseline+P2+retail);
$\kappa$ vs human majority ($n={hum['n_verifiable']}$); $\tau$: Kendall rank correlation of the per-model baseline
ranking with the default.}}
\label{{tab:validator-robustness}}
\small
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Variant & Baseline & P2 & Flips & $\kappa$ & $\tau$ \\
\midrule
{chr(10).join(rl)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(f"{OUT}/tab_V4_robustness.tex", "w").write(t4)

    # ---------------- Table V5: gold audit
    ga = main_["gold_audit"]
    slot_lines = "\n".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {v} \\\\" for k, v in ga["by_slot"].items())
    nvt = main_["not_verifiable_tasks"]["by_template"]
    t5 = rf"""\begin{{table}}[t]
\centering
\caption{{Audit of the task-file gold sequences against the query text, over the {ga['n_tasks_in_traces']} distinct tasks
that appear in any trace. {ga['n_verifiable_tasks']} are verifiable; in {ga['n_tasks_taskfile_contradicts_query']}
of them the task-file \texttt{{correct\_tool\_sequence}} contradicts the query (generator draws slot values independently
of the values rendered into the query, ignores conditions such as `cheaper' or `in stock', or, for ToolBench-style
tasks, uses a fixed placeholder call). The validator uses the query-grounded gold. Not verifiable:
{', '.join(f'{k.replace("_", chr(92) + "_")} ({v})' for k, v in nvt.items())}.}}
\label{{tab:gold-audit}}
\small
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
Contradicted slot (task file vs query) & Tasks \\
\midrule
{slot_lines}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(f"{OUT}/tab_V5_gold_audit.tex", "w").write(t5)
    print("ok")


if __name__ == "__main__":
    main()
