"""C2_06: LaTeX tables + consolidated results.json for C2_openweight."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from C2_common import OUT, PRETTY, OPEN
R = json.load(open(f"{OUT}/open_results.json"))
L1C = json.load(open(f"{OUT}/l1_categories.json"))
TF = json.load(open(f"{OUT}/trace_failure_types.json"))
V2 = json.load(open(f"{OUT}/v2_trace_checks.json"))


def fp(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def row(name, x):
    r = x["random_abstention"]
    lo, hi = x["delta_paired_boot95_pp"]
    m = x["mcnemar"]
    return (f"{name} & {x['n']} & {x['ctrl_halluc_pct']:.1f} & {x['int_halluc_pct']:.1f} & "
            f"{x['delta_pp(int-ctrl)']:+.1f} [{lo:+.1f}, {hi:+.1f}] & {m['b_ctrlH_intNotH']}/{m['c_ctrlNotH_intH']}, {fp(m['p_exact'])} & "
            f"{x['coverage_pct']:.1f} & {x['acc_answered_int_pct']:.1f} & {x['ctrl_acc_pct']:.1f} & {x['saved']} & {x['broken']} & "
            f"{x['precision']:.2f} ({x['base_rate']:.2f}) & {x['recall']:.2f} & {r['expected_halluc_pct']:.1f} & "
            f"{r['int_minus_random_pp']:+.1f} & {fp(r['p_sim_one_sided(saved>=obs)'])} \\\\")


hdr = (r"Model & $n$ & Ctrl\% & Int\% & $\Delta$ pp [paired 95\% CI] & McNemar $b/c$, $p$ & Cov.\% & Acc.\ ans.\% & Ctrl acc.\% & Sv & Bk & Prec.\ (base) & Rec. & Random\% & Int$-$Rand & $p_{\mathrm{rand}}$ \\")
cap1 = (r"Interceptor re-analysed as selective prediction, all five models, from raw per-task CSVs (heuristic labels). "
        r"Open-weight rows: $n{=}300$ tasks (seed-42 sample of the 450 test tasks in \texttt{full\_500\_tasks.json}, 66 of them HaluEval with empty references), "
        r"both arms run concurrently in one job; proprietary rows: $n{=}600$ different tasks, control run two days later. "
        r"$\Delta$ = Int$-$Ctrl hallucination (abstentions count as non-hallucinated); CI = paired task-level bootstrap (10{,}000). "
        r"McNemar $b$ = ctrl hallucinated \& int not, $c$ = reverse. Acc.\ ans.\ = accuracy of the interceptor arm on the tasks it answered. "
        r"Sv/Bk = abstained on a task the control got wrong/right; Prec.\ = Sv/abstentions (base = control error rate); Rec.\ = Sv/control errors. "
        r"Random = exact expected hallucination when abstaining on the same number of tasks uniformly at random; Int$-$Rand $>0$ means worse than random; "
        r"$p_{\mathrm{rand}}$ = one-sided simulation p-value (20{,}000 draws) that random abstention saves at least as many errors.")
L = [r"% C2_openweight/tab_C2_1_selective.tex -- source: original project data/results/interceptor_{qwen25_7b,qwen25_14b,hermes3_llama8b}[_schemaonly].csv, interceptor_{control,heldout}_600.csv; scripts/C2_01",
     r"\begin{table}[t]\centering\small", r"\caption{" + cap1 + "}", r"\label{tab:C2_1}", r"\resizebox{\linewidth}{!}{%",
     r"\begin{tabular}{lrrrlrrrrrrrrrrr}", r"\toprule", hdr, r"\midrule",
     r"\multicolumn{16}{l}{\emph{Three-layer interceptor as run (L1 $\geq$2 schema/tool errors; L3 unsupported numbers)}} \\"]
for m in ["gpt4o_mini", "gemini_20_flash"]:
    L.append(row(PRETTY[m], R["proprietary"][m]["all"]))
for m in OPEN:
    L.append(row(PRETTY[m], R["open"][m]["full"]["all"]))
L += [r"\midrule", r"\multicolumn{16}{l}{\emph{Schema-only ``precision mode'' (\texttt{INT\_NO\_L3=1}; separate re-run of both arms)}} \\"]
for m in OPEN:
    L.append(row(PRETTY[m], R["open"][m]["schemaonly"]["all"]))
L += [r"\midrule", r"\multicolumn{16}{l}{\emph{Open-weight, excluding the 66 HaluEval tasks (empty reference $\Rightarrow$ always scored wrong)}} \\"]
for m in OPEN:
    L.append(row(PRETTY[m] + " (full)", R["open"][m]["full"]["excl_halueval"]))
for m in OPEN:
    L.append(row(PRETTY[m] + " (schema)", R["open"][m]["schemaonly"]["excl_halueval"]))
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/tab_C2_1_selective.tex", "w").write("\n".join(L) + "\n")

# per-layer table
L = [r"% C2_openweight/tab_C2_2_layers.tex -- source: abstain_reason in the interceptor-arm rows; scripts/C2_01, C2_02",
     r"\begin{table}[t]\centering\small",
     r"\caption{Per-layer attribution of the interceptor's abstentions (from \texttt{abstain\_reason}), scored against the paired control arm. "
     r"Prec.\ = fraction of that layer's abstentions on tasks the control got wrong; base = control error rate; $p$ = one-sided hypergeometric test that the layer hits control errors more often than random. "
     r"L1 sub-types from the logged validator message. Open-weight $n{=}300$, proprietary $n{=}600$.}",
     r"\label{tab:C2_2}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
     r"Model & Run / layer & Abst. & Sv & Bk & Prec.\ (base) & $p$ \\", r"\midrule"]
for grp, m, var in [("proprietary", "gpt4o_mini", "all"), ("proprietary", "gemini_20_flash", "all")] + [("open", m, v) for m in OPEN for v in ("full", "schemaonly")]:
    x = R[grp][m][var] if grp == "proprietary" else R[grp][m][var]["all"]
    for Ly in ("L1", "L3"):
        b = x["by_layer"][Ly]
        if not b["abstentions"]:
            continue
        runname = {"all": "as run", "full": "as run", "schemaonly": "schema-only"}[var]
        L.append(f"{PRETTY[m]} & {runname} / {Ly} & {b['abstentions']} & {b['saved']} & {b['broken']} & {b['precision']:.2f} ({x['base_rate']:.2f}) & {fp(b['p_vs_random_hypergeom'])} \\\\")
L += [r"\midrule", r"\multicolumn{7}{l}{\emph{L1 abstentions by validator message (Sv/Bk), three-layer run}} \\"]
cats = sorted({c for k, v in L1C.items() if isinstance(v, dict) and "/" in k for c in v})
for k in ["qwen25_7b/full", "qwen25_14b/full", "hermes3_llama8b/full", "gpt4o_mini/full", "gemini_20_flash/full"]:
    m = k.split("/")[0]
    parts = [f"{c.split('(')[0].replace('_', ' ')} {v['saved(ctrl_wrong)']}/{v['broken(ctrl_right)']}" for c, v in L1C[k].items()]
    L.append(f"{PRETTY[m]} & \\multicolumn{{6}}{{l}}{{{'; '.join(parts)}}} \\\\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/tab_C2_2_layers.tex", "w").write("\n".join(L) + "\n")

# verdict table
def v(m, var="full"):
    return R["open"][m][var]["all"]
q7, q14, he = v("qwen25_7b"), v("qwen25_14b"), v("hermes3_llama8b")
q7s, q14s, hes = v("qwen25_7b", "schemaonly"), v("qwen25_14b", "schemaonly"), v("hermes3_llama8b", "schemaonly")
VR = [
 ("Qwen2.5-7B $-24.0$ pp, Sv/Bk 75/91 (Table 7, v2)",
  f"Reproduces exactly from raw data. Coverage {q7['coverage_pct']:.1f}\\%; answered accuracy {q7['acc_answered_int_pct']:.1f}\\% vs control {q7['ctrl_acc_pct']:.1f}\\%; "
  f"precision {q7['precision']:.2f} below base {q7['base_rate']:.2f}; random abstention gives {q7['random_abstention']['expected_halluc_pct']:.1f}\\% (worse than random, $p{{=}}${q7['random_abstention']['p_hypergeom_worse_than_random']:.0e}). Report as abstention volume, not detection."),
 ("Qwen2.5-14B $-8.0$ pp, 24/57",
  f"Reproduces. Precision {q14['precision']:.2f} vs base {q14['base_rate']:.2f}; random {q14['random_abstention']['expected_halluc_pct']:.1f}\\% (worse than random, $p{{=}}${q14['random_abstention']['p_hypergeom_worse_than_random']:.0e}); answered accuracy falls {q14['ctrl_acc_pct']:.1f}$\\to${q14['acc_answered_int_pct']:.1f}\\%."),
 ("Hermes-3-8B $-4.0$ pp, 16/24",
  f"Reproduces; paired CI [{he['delta_paired_boot95_pp'][0]:.1f}, {he['delta_paired_boot95_pp'][1]:.1f}] excludes 0 (C's unpaired CI did not). Precision {he['precision']:.2f} vs base {he['base_rate']:.2f}; indistinguishable from random ($p_{{\\mathrm{{rand}}}}{{=}}${he['random_abstention']['p_sim_one_sided(saved>=obs)']:.2f})."),
 ("``All CIs exclude zero'' on tool-calling models",
  "True for the paired bootstrap (the appropriate test), but the CI is on a quantity that any abstention lowers; all three open rows are at or below random abstention."),
 ("``Largest reduction on an open-weight model, matching best proprietary''",
  "Arithmetically true but driven by 55\\% abstention; different task pool ($n{=}300$ from \\texttt{full\\_500}, 22\\% HaluEval with empty references) from the proprietary rows ($n{=}600$ from \\texttt{full\\_2000}); rows not comparable."),
 ("Precision mode, Qwen2.5-7B: $-13.3$ pp at precision 0.62 (39 vs 24)",
  f"Verified exactly (schema-only re-run: Ctrl {q7s['ctrl_halluc_pct']:.1f}\\%, Int {q7s['int_halluc_pct']:.1f}\\%, 39/24, 0.619). But base rate is {q7s['base_rate']:.2f}, so random abstention at 21\\% gives {q7s['random_abstention']['expected_halluc_pct']:.1f}\\%: only {-q7s['random_abstention']['int_minus_random_pp']:.1f} pp better, $p_{{\\mathrm{{rand}}}}{{=}}${q7s['random_abstention']['p_sim_one_sided(saved>=obs)']:.3f}. ``Net-positive'' (Sv$>$Bk) is expected from random abstention alone when base $>$0.5."),
 ("Precision mode, Hermes: $-4.0$ pp at 0.62",
  f"Verified (8/5 of 13 abstentions, 0.615). $n$ too small: $p_{{\\mathrm{{rand}}}}{{=}}${hes['random_abstention']['p_sim_one_sided(saved>=obs)']:.2f}."),
 ("Precision mode, Qwen2.5-14B (not reported in v2)",
  f"Schema-only run: $\\Delta$ {q14s['delta_pp(int-ctrl)']:+.1f} pp, Sv/Bk {q14s['saved']}/{q14s['broken']} (precision {q14s['precision']:.2f}) -- a null/harmful operating point omitted from the paper."),
 ("``Both are exact concurrent-control measurements''",
  "True for the open-weight runs (both arms interleaved in one job, same 300 task ids), but replicate controls disagree on 23/300 (Qwen-7B), 3/300 (Qwen-14B), 13/300 (Hermes) labels."),
 ("``single L40S GPU''",
  "Untraceable: code paths reference Modal A10G/A100-40GB (\\texttt{modal\\_open\\_interceptor.py}) or a RunPod box (\\texttt{run\\_open\\_interceptor\\_local.py}); no run log or trace DB survives (NullTracer)."),
]
L = [r"% C2_openweight/tab_C2_3_verdicts.tex -- replaces the open-weight rows of C_interceptor verdict table",
     r"\begin{table}[t]\centering\small", r"\caption{Updated verdicts for the open-weight interceptor claims (arXiv v2 / TACL Table 7 and \S``tunable operating point''), now from the raw CSVs in the original project.}",
     r"\label{tab:C2_3}", r"\begin{tabular}{p{0.32\linewidth}p{0.62\linewidth}}", r"\toprule", r"Claim & Verdict \\", r"\midrule"]
for a, b in VR:
    L.append(f"{a} & {b} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/tab_C2_3_verdicts.tex", "w").write("\n".join(L) + "\n")

res = {"open_weight": R["open"], "proprietary_same_metrics": R["proprietary"], "l1_categories": L1C,
       "trace_failure_types_proprietary": TF, "v2_trace_checks": V2,
       "notes": {"labels": "harness substring heuristic (kappa 0.049 vs humans); V_validator/labels_all.csv not available at time of analysis",
                 "trace_dbs": "interceptor_day8.db and interceptor_control.db contain only gpt4o_mini and gemini_20_flash; open-weight runs used NullTracer (no traces)",
                 "tasks_open": "seed-42 shuffle of 450 test tasks in full_500_tasks.json, first 300; identical across all 6 open CSVs and both arms",
                 "schemaonly": "INT_NO_L3 env flag added to run_interceptor.py (uncommitted change, mtime 17:03) before the schemaonly CSVs (17:16-17:22); both arms re-run"}}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=str)
print("ok")
