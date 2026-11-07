"""C_07: Risk-coverage figure (PDF) + LaTeX booktabs tables + consolidated results.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C_common import OUT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a", "#333333"

rc = json.load(open(f"{OUT}/risk_coverage_data.json"))
rep = json.load(open(f"{OUT}/replay_results.json"))
orig = json.load(open(f"{OUT}/original_csv_results.json"))
cons = json.load(open(f"{OUT}/reported_table_consistency.json"))
lat = json.load(open(f"{OUT}/latency_results.json"))

PANELS = [("clean_r1__scoreable", "(a) Clean tasks, original harness"),
          ("clean_schema__scoreable", "(b) Clean tasks, schema-exposed harness"),
          ("p2sem_schema__scoreable", "(c) P2 semantic-wrong, schema-exposed"),
          ("retail_r1", "(d) Unseen retail domain, original harness")]
MARK = {"ORIG (L1>=2 | L3; as run)": ("o", "As run (L1≥2 ∨ L3)"),
        "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)": ("s", "Paper Coverage mode"),
        "L1>=1": ("^", "L1 only (Precision)"),
        "L3": ("v", "L3 only"),
        "L2paper>=1": ("D", "L2 only"),
        "No tool call": ("P", "Abstain if no tool call")}
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
for ax, (name, title) in zip(axes, PANELS):
    if name not in rc:
        ax.set_visible(False); continue
    d = rc[name]
    ax.plot(d["cov"], d["risk_interceptor_score"], color=BLUE, lw=2, label="Interceptor score ranking")
    ax.plot(d["cov"], d["risk_notool_score"], color=AQUA, lw=2, label="No-tool-call ranking")
    ax.plot(d["cov"], d["oracle_risk"], color=GREY, lw=1, ls=":", label="Oracle")
    ax.axhline(d["base_risk"], color=ORANGE, lw=2, ls="--", label="Random abstention (any coverage)")
    for p, (m, lab) in MARK.items():
        if p in d["points"]:
            c, r = d["points"][p]
            if r is not None:
                ax.plot([c], [r], marker=m, ms=8, color=INK, mfc="white", mew=1.5, ls="none", label=lab)
    ax.set_title(f"{title}\n(n={len(d['cov'])}, label={'LLM judge' if d['label']=='j_correct' else 'heuristic'})", fontsize=8.5)
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel("Selective risk (wrong among answered)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=8)
fig.savefig(f"{OUT}/fig_risk_coverage.pdf"); fig.savefig(f"{OUT}/fig_risk_coverage.png", dpi=160)


def f1(x): return "--" if x is None else f"{100*x:.1f}"
def f2(x): return "--" if x is None else f"{x:.2f}"
def pv(x):
    if x is None: return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


# Table C1: original CSV experiment, selective-prediction metrics
L = [r"% C_interceptor/tab_C1_original.tex -- source: interceptor_control_600.csv, interceptor_heldout_600.csv (C_01)",
     r"\begin{table}[t]\centering\small",
     r"\caption{Original interceptor experiment re-analysed as selective prediction (heuristic labels, $n{=}600$ tasks per model, paired by task). Sv/Bk = abstained on a task the control arm got wrong/right. Precision = Sv/abstentions; recall = Sv/control errors. ``Random'' = expected hallucination when abstaining on the same number of control tasks uniformly at random; $p$ = one-sided hypergeometric test that the Interceptor's abstentions hit more control errors than random.}",
     r"\label{tab:C1}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
     r"Model & Ctrl hall.\% & Int.\ hall.\% & Cov.\% & Sel.\ acc.\% (Int) & Ctrl acc.\% & Sv & Bk & Prec. & Rec. & Random hall.\% & $p$ \\", r"\midrule"]
for m, nm in [("gpt4o_mini", "GPT-4o-mini"), ("gemini_20_flash", "Gemini-2.0-Flash")]:
    o = orig[m]
    L.append(f"{nm} & {f1(o['ctrl_halluc'])} & {f1(o['int_halluc'])} & {f1(o['coverage'])} & {f1(o['selective_accuracy_int_answered'])} & {f1(o['ctrl_accuracy'])} & {o['saved_A(abst & ctrl_h)']} & {o['broken_A(abst & !ctrl_h)']} & {f2(o['abstention_precision_proxy'])} & {f2(o['abstention_recall_proxy'])} & {f1(o['random_abstention_matched']['expected_halluc'])} & {pv(o['random_abstention_matched']['p_one_sided_int_better(hypergeom)'])} \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{12}{l}{\emph{Per layer (GPT-4o-mini, from \texttt{abstain\_reason})}} \\")
for lay in ["L1", "L3"]:
    b = orig["gpt4o_mini"]["by_layer"][lay]
    L.append(f"\\quad {lay} abstentions & \\multicolumn{{5}}{{r}}{{{b['abstentions']} abstentions}} & {b['ctrl_hallucinated']} & {b['ctrl_correct']} & {f2(b['precision_proxy'])} & -- & -- & {pv(b['p_vs_random(hypergeom, one-sided)'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/tab_C1_original.tex", "w").write("\n".join(L) + "\n")

# Table C2: offline replay, policies at matched coverage (judge labels, scoreable subset)
POLS = ["ORIG (L1>=2 | L3; as run)", "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)", "L1>=1", "L2paper>=1", "L3",
        "No tool call", "No tool call | inline-fabricated", "Arg-grounding (diag. L4)"]
NAMES = {"ORIG (L1>=2 | L3; as run)": "As run: $L_1{\\ge}2 \\lor L_3$",
         "PAPER-COVERAGE (L1>=1 | L2paper>=1 | L3)": "Paper Coverage: $L_1{\\ge}1 \\lor L_2 \\lor L_3$",
         "L1>=1": "$L_1$ only (Precision mode)", "L2paper>=1": "$L_2$ only", "L3": "$L_3$ only",
         "No tool call": "Abstain if no tool executed", "No tool call | inline-fabricated": "No tool $\\lor$ inline-fabricated",
         "Arg-grounding (diag. L4)": "Arg-grounding check (diagnostic)"}
RUNS = [("clean_r1__scoreable", "Clean, original harness"), ("clean_schema__scoreable", "Clean, schema-exposed harness"),
        ("p2sem_schema__scoreable", "P2 semantic-wrong, schema-exposed"), ("retail_r1", "Retail (unseen domain), original harness"),
        ("retail_schema", "Retail (unseen domain), schema-exposed")]
L = [r"% C_interceptor/tab_C2_replay.tex -- source: C_02 full-trace GPT-4o-mini re-runs, C_03 offline replay, C_04 judge",
     r"\begin{table}[t]\centering\small",
     r"\caption{Offline replay of Interceptor layers and baselines on full GPT-4o-mini traces (no abstention applied at run time; GPT-4o-mini judge labels; halueval/toolbench\_syn excluded from clean/P2 rows because their references are empty or echo the query). Sel.\ risk = wrong among answered. Random hall.\% = expected hallucination of random abstention at the same rate. $p$: one-sided hypergeometric test vs.\ random.}",
     r"\label{tab:C2}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
     r"Run & Policy & Cov.\% & Hall.\% after & Random hall.\% & Sel.\ risk\% & Sv & Bk & Prec. & $p$ \\", r"\midrule"]
for run, rn in RUNS:
    R = rep["runs"].get(run)
    if not R: continue
    lab = "j_correct" if "j_correct" in R else "h_correct"
    first = True
    base = R[lab]["L1>=1"]["halluc_before"]
    L.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{rn}}} ($n{{=}}{R['n']}$, no-abstention hallucination {f1(base)}\\%)}} \\\\")
    for p in POLS:
        e = R[lab][p]
        L.append(f" & {NAMES[p]} & {f1(e['coverage'])} & {f1(e['halluc_after'])} & {f1(e['random_matched_expected_halluc'])} & {f1(e['selective_risk'])} & {e['saved']} & {e['broken']} & {f2(e['precision'])} & {pv(e['p_better_than_random(hypergeom one-sided)'])} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/tab_C2_replay.tex", "w").write("\n".join(L) + "\n")

# Table C3: reported Table 8 audit
L = [r"% C_interceptor/tab_C3_reported_audit.tex -- source: reported Table 8 numbers (no raw open-weight data exists); C_06",
     r"\begin{table}[t]\centering\small",
     r"\caption{Audit of the submitted interceptor table. $n$ = the only task count reproducing every reported percentage. Random = hallucination expected from abstaining at random at the reported rate, $\mathrm{Ctrl}\times(1-\mathrm{Abst})$. Positive Int$-$Random means the Interceptor is \emph{worse} than random abstention. Precision under both Sv/Bk orderings; base rate = Ctrl\%. Unpaired 95\% CI = the paper's own CI method.}",
     r"\label{tab:C3}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrrrl}", r"\toprule",
     r"Model & $n$ & Ctrl\% & Int\% & Abst\% & Random\% & Int$-$Random (pp) & Prec.\ (submitted Sv/Bk) & Prec.\ (arXiv v2 Sv/Bk) & Unpaired 95\% CI $\Delta$ \\", r"\midrule"]
ROWS = {"GPT-4o-mini": (55.8, 32.8, 33.5), "Gemini-2.0-Flash": (44.5, 43.2, 2.8), "Qwen2.5-7B": (55.3, 31.3, 55.3),
        "Qwen2.5-14B": (45.0, 37.0, 27.0), "Hermes-3-Llama-8B": (43.0, 39.0, 13.3)}
for m, (c, i, a) in ROWS.items():
    r = cons[m]
    s, v = r["submitted"], r["arxiv_v2"]
    ci = r["unpaired_95CI_delta_pp"]
    L.append(f"{m} & {r['n']} & {c:.1f} & {i:.1f} & {a:.1f} & {r['random_abstention_expected_int_pct']:.1f} & {r['int_minus_random_pp']:+.1f} & {s['precision']:.2f} ({s['Sv']}/{s['Bk']}) & {v['precision']:.2f} ({v['Sv']}/{v['Bk']}) & [{ci[0]:.1f}, {ci[1]:.1f}] \\\\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/tab_C3_reported_audit.tex", "w").write("\n".join(L) + "\n")

# Table C4: latency
L = [r"% C_interceptor/tab_C4_latency.tex -- source: C_05 (real traces from C_02)",
     r"\begin{table}[t]\centering\small",
     r"\caption{Interceptor overhead measured on real GPT-4o-mini traces vs.\ the agent's own LLM latency recorded in the same traces.}",
     r"\label{tab:C4}", r"\begin{tabular}{lr}", r"\toprule",
     f"Traces timed & {lat['n_traces']} \\\\",
     f"Monitor CPU time, median / p95 ($\\mu$s per trace) & {lat['monitor_us_per_trace']['median']:.1f} / {lat['monitor_us_per_trace']['p95']:.1f} \\\\",
     f"Agent LLM wall-clock, median / mean (ms per trace) & {lat['agent_llm_ms_per_trace']['median']:.0f} / {lat['agent_llm_ms_per_trace']['mean']:.0f} \\\\",
     f"LLM calls per trace (mean) & {lat['agent_llm_calls_per_trace_mean']:.2f} \\\\",
     f"Overhead / LLM latency, median (max) & {100*lat['overhead_fraction_of_llm_latency']['median']:.4f}\\% ({100*lat['overhead_fraction_of_llm_latency']['max']:.4f}\\%) \\\\",
     r"Additional LLM calls / tokens & 0 / 0 \\",
     r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/tab_C4_latency.tex", "w").write("\n".join(L) + "\n")

json.dump({"original_csv": orig, "original_csv_judged": json.load(open(f"{OUT}/original_csv_judged.json")), "replay": rep, "propagation_detection": json.load(open(f"{OUT}/propagation_detection.json")), "reported_table_audit": cons, "latency": lat},
          open(f"{OUT}/results.json", "w"), indent=1, default=str)
print("ok")
