"""Assemble results.json and LaTeX booktabs tables from the B_0x outputs."""
import sys
sys.path.insert(0, "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts")
from B_common import *  # noqa

L = lambda n: json.load(open(os.path.join(OUT, n)))
b1, b2, b4, b7, b8 = (L("B01_interannotator.json"), L("B02_surface_metrics.json"), L("B04_judge_robustness.json"),
                      L("B07_self_preference.json"), L("B08_transfer_provisional.json"))
import B_06_score_round2 as S
r1f = S.round1_factors()

f3 = lambda x: f"{x:.3f}".replace("-0.000", "0.000")
f2 = lambda x: f"{x:.2f}".replace("-0.00", "0.00")
ci = lambda c: f"[{f2(c[0])}, {f2(c[1])}]"
TEX = os.path.join(OUT, "tables")
os.makedirs(TEX, exist_ok=True)

# ---------------- inter-annotator table
P = b1["pairs"]
nm = {"bhaskar": "A1 (first author)", "yoshitha": "A2 (Yoshitha)", "third": "A3 (third pass)"}
rows = []
for k, v in P.items():
    a, b = k.split("-")
    rows.append(f"{nm[a]} -- {nm[b]} & {f3(v['kappa'])} & {ci(v['kappa_ci95'])} & {f2(v['raw_agreement'])} & "
                f"{f3(v['gwet_ac1'])} & {sum(1 for _ in v['disagree_ids'])} \\\\")
fk = b1["fleiss_kappa_3"]
tex = r"""\begin{table}[t]
\centering
\caption{Inter-annotator agreement on the 100 calibration traces (binary correct/incorrect).
95\% CIs: percentile bootstrap over traces ($B{=}10{,}000$). YES rates: A1 0.38, A2 0.44, A3 0.43.
Fleiss' $\kappa$ uses all three annotators; """ + f"{b1['unanimous_items']}" + r""" of 100 traces are unanimous.}
\label{tab:iaa}
\small
\begin{tabular}{lccccc}
\toprule
Pair & Cohen's $\kappa$ & 95\% CI & Raw agr. & Gwet AC1 & \#Disagree \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
All three (Fleiss) & """ + f"{f3(fk['kappa'])} & {ci(fk['ci95'])}" + r""" & -- & -- & """ + f"{b1['items_with_any_disagreement']}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(TEX, "tab_interannotator.tex"), "w").write(tex)

# ---------------- judge reliability table
M = b2["metrics"]
C = b4["configs"]
groups = [
    ("String / surface metrics (reference-based)", [
        ("Exact match", M["exact"]), ("Normalized match (equality)", M["normalized_eq"]),
        ("Normalized containment", M["normalized_contain"]), ("Token-F1 $\\geq$ 0.5", M["token_f1_ge_0.5"]),
        ("Numeric tolerance (1\\%/0.5)", M["numeric_tol"]), ("Reference-entity match", M["ref_entity_match"]),
        ("Substring heuristic (as coded)", M["heuristic_csv"]),
        ("Substring heuristic (as described)", M["heuristic_paper_described"])]),
    ("Query-grounded slot checks (rule-based)", [
        ("Slot presence", M["slot_present"]), ("+ no conflicting entity", M["slot_noconflict"]),
        ("Non-answer filter only", M["non_answer_filter_only"]),
        ("Slot-strict (presence+no-conflict+non-answer)", M["slot_strict"])]),
    ("Original LLM judges (stored verdicts, original prompt)", [
        ("GPT-4o", M["gpt4o"]), ("Gemini-2.5-Flash", M["gemini25flash"]), ("GPT-4o-mini", M["gpt4o_mini"]),
        ("3-LLM ensemble (majority)", M["ensemble"])]),
]
newj = []
for key in ["gpt-4o-mini|P0_original", "gpt-4o-mini|P3_reason_then_verdict", "gpt-4.1-mini|P0_original",
            "gpt-4.1-mini|P3_reason_then_verdict", "gpt-4.1|P0_original", "gpt-4.1|P3_reason_then_verdict",
            "gpt-5-mini|P0_original", "gpt-5-mini|P3_reason_then_verdict"]:
    j, p = key.split("|")
    newj.append((f"{j} ({p.split('_')[0]})", C[key]))
groups.append(("Re-run LLM judges (this revision; P0=original prompt, P3=reason-then-verdict)", newj))

lines = []
for g, items in groups:
    lines.append(r"\multicolumn{7}{l}{\emph{" + g + r"}} \\")
    for name, e in items:
        hb = e["h_bhaskar"] if "h_bhaskar" in e else e["bhaskar"]
        hy = e["h_yoshitha"] if "h_yoshitha" in e else e["yoshitha"]
        ht = e["h_third"] if "h_third" in e else e["third"]
        hm = e["h_majority3"] if "h_majority3" in e else e["majority3"]
        lines.append(f"\\quad {name} & {f2(e['yes_rate'])} & {f3(hb['kappa'])} {ci(hb['ci95'])} & {f3(hy['kappa'])} & "
                     f"{f3(ht['kappa'])} & {f3(hm['kappa'])} & {ci(hm['ci95'])} \\\\")
    lines.append(r"\addlinespace")
tex = r"""\begin{table*}[t]
\centering
\caption{Agreement (Cohen's $\kappa$) of correctness metrics with human labels on the 100 calibration traces.
A1/A2/A3: individual annotators; Maj.: 3-annotator majority (human YES rate 0.42). 95\% CIs: percentile
bootstrap ($B{=}10{,}000$). All 100 references are task descriptions, not gold values (Sec.~X), which caps
every reference-based string metric. Slot checks and prompt P3 were designed by the authors after
inspecting the failure modes in this set and are therefore not out-of-sample estimates. On the 88 traces not
inspected during design, slot-strict reaches $\kappa{=}""" + f"{b2['heldout_88_uninspected']['slot_strict']['kappa']:.2f}" + r"""$.}
\label{tab:judge-expanded}
\small
\begin{tabular}{lcccccc}
\toprule
Metric & YES rate & $\kappa$ vs A1 [95\% CI] & vs A2 & vs A3 & vs Maj. & Maj. 95\% CI \\
\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(TEX, "tab_judge_reliability.tex"), "w").write(tex)

# ---------------- prompt robustness table
prompts = ["P0_original", "P1_rubric", "P2_request_centric", "P3_reason_then_verdict"]
judges = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini"]
lines = []
for j in judges:
    cells = []
    for p in prompts:
        e = C[f"{j}|{p}"]["majority3"]
        cells.append(f"{f2(e['kappa'])} {ci(e['ci95'])}")
    ps = b4["prompt_sensitivity_vs_majority3"][j]
    lines.append(f"{j} & " + " & ".join(cells) + f" & {f2(ps['range'])} \\\\")
cells = []
for p in prompts:
    e = C[f"ens(4.1-mini,4.1,5-mini)|{p}"]["majority3"]
    cells.append(f"{f2(e['kappa'])} {ci(e['ci95'])}")
lines.append(r"\midrule" + "\n" + "Majority(4.1-mini, 4.1, 5-mini) & " + " & ".join(cells) + " & -- \\\\")
tex = r"""\begin{table*}[t]
\centering
\caption{Prompt and judge sensitivity: Cohen's $\kappa$ [95\% CI] vs.\ the 3-annotator majority on the
100 calibration traces (temperature 0; GPT-5-mini at minimal reasoning effort). P0: original ensemble
prompt; P1: explicit rubric; P2: request-centric; P3: brief reasoning then verdict. Re-running the original
prompt reproduces the stored verdicts on 97\% of traces (GPT-4o-mini and GPT-4o).
Total API cost of the re-judging runs: \$""" + f"{b4['total_cost_usd']:.2f}" + r""".}
\label{tab:prompt-robustness}
\small
\begin{tabular}{lccccc}
\toprule
Judge & P0 & P1 & P2 & P3 & Range \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(TEX, "tab_prompt_robustness.tex"), "w").write(tex)

# ---------------- self-preference table
A = b7["A_human_anchored_n100"]
lines = []
for name, lbl in [("gpt4o(stored,P0)", "GPT-4o"), ("gpt4o_mini(stored,P0)", "GPT-4o-mini"),
                  ("gemini25flash(stored,P0)", "Gemini-2.5-Flash"), ("ensemble(stored)", "Ensemble")]:
    for lvl, ll in [("same_vendor", "vendor"), ("exact_model", "self")]:
        if lvl in A[name]:
            x = A[name][lvl]
            d = x["diff_bias_own_minus_other"]
            lines.append(f"{lbl} & {ll} & {x['own']['n']} & {x['own']['bias']:+.2f} & {x['other']['bias']:+.2f} & "
                         f"{d['diff']:+.2f} {ci(d['ci95'])} \\\\")
Cx = b7["C_human_free_exact_self"]
Bx = b7["B_human_free_n2150"]
tex = r"""\begin{table}[t]
\centering
\caption{Self-preference check. Bias = judge YES rate $-$ human-majority YES rate (negative = conservative)
on traces from the judge's own vendor/model vs.\ all other traces (100 calibration traces; CIs: bootstrap).
Human-free check over all 2{,}150 P2 traces (judge minus Gemini-2.5-Flash YES-rate, own-model traces minus
other traces): GPT-4o $""" + f"{Cx['gpt4o_verdict_minus_gemini_25_flash_verdict_on_gpt4o']['did']:+.3f}" + r"""$ """ + ci(Cx['gpt4o_verdict_minus_gemini_25_flash_verdict_on_gpt4o']['ci95']) + r""" ($n_{own}{=}50$),
GPT-4o-mini $""" + f"{Cx['gpt4o_mini_verdict_minus_gemini_25_flash_verdict_on_gpt4o_mini']['did']:+.3f}" + r"""$ """ + ci(Cx['gpt4o_mini_verdict_minus_gemini_25_flash_verdict_on_gpt4o_mini']['ci95']) + r""" ($n_{own}{=}200$);
vendor-level (OpenAI vs Google agents): GPT-4o $""" + f"{Bx['gpt4o_verdict_vs_gemini_25_flash']['did_openaiAgents_minus_googleAgents']:+.3f}" + r"""$ """ + ci(Bx['gpt4o_verdict_vs_gemini_25_flash']['ci95']) + r""",
GPT-4o-mini $""" + f"{Bx['gpt4o_mini_verdict_vs_gemini_25_flash']['did_openaiAgents_minus_googleAgents']:+.3f}" + r"""$ """ + ci(Bx['gpt4o_mini_verdict_vs_gemini_25_flash']['ci95']) + r""".}
\label{tab:selfpref}
\small
\begin{tabular}{llcccc}
\toprule
Judge & Own = & $n_{own}$ & Bias own & Bias other & $\Delta$ [95\% CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(TEX, "tab_self_preference.tex"), "w").write(tex)

# ---------------- transfer table
lines = []
for ref, sec in [("GPT-4.1 (P3)", "vs_strong_judge_sample"), ("Slot-strict", "vs_slot_strict_all2150")]:
    for k in ["open_weight|S2", "proprietary|S2", "open_weight|nonS2", "proprietary|nonS2"]:
        v = b8[sec][k]
        fam, s = k.split("|")
        lines.append(f"{ref} & {fam.replace('_', '-')} & {s.replace('nonS2', 'EPS$<$2').replace('S2', 'EPS$\\geq$2')} & {v['n']} & "
                     f"{f2(v['ens_yes'])} & {f2(v['ref_yes'])} & {f2(v['kappa'])} {ci(v['ci95'])} & "
                     f"{f2(v['P_refYES_given_ensNO'])} & {f2(v['P_refYES_given_ensYES'])} \\\\")
    lines.append(r"\midrule")
tex = r"""\begin{table*}[t]
\centering
\caption{Provisional calibration-transfer check (no new human labels). Agreement of the 3-LLM ensemble with
two non-ensemble references on open-weight vs.\ proprietary P2 traces, excluding the 100 calibration traces.
The last two columns are the reference-based analogues of the human calibration factors
$c_0{=}P(\mathrm{correct}\mid\mathrm{ens. NO})$ and $c_1{=}P(\mathrm{correct}\mid\mathrm{ens. YES})$; round-1 human values
(3-annotator majority) are $c_0{=}""" + f"{r1f['majority3']['all']['c0_P_humanYES_given_ensNO']['p']:.2f}" + r"""$, $c_1{=}""" + f"{r1f['majority3']['all']['c1_P_humanYES_given_ensYES']['p']:.2f}" + r"""$
(only 10 calibration traces have EPS$\geq$2). The two references disagree on the direction of the
open-vs-proprietary difference, so a human round is required.}
\label{tab:transfer}
\small
\begin{tabular}{lllcccccc}
\toprule
Reference & Family & Stratum & $n$ & Ens. YES & Ref. YES & $\kappa$ [95\% CI] & $P(\mathrm{ref YES}\mid\mathrm{ens NO})$ & $P(\mathrm{ref YES}\mid\mathrm{ens YES})$ \\
\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(TEX, "tab_transfer.tex"), "w").write(tex)

# ---------------- results.json
results = {
    "n_calibration_traces": 100,
    "critique1_interannotator": {k: v for k, v in b1.items() if k not in ("disagreement_detail",)},
    "critique2_surface_metrics": {"metrics": M, "paired_kappa_diffs_vs_majority3": b2["paired_kappa_diffs_vs_majority3"],
                                  "heldout_88_uninspected": b2["heldout_88_uninspected"],
                                  "reference_audit": b2["reference_audit"]},
    "critique3_judge_robustness": b4,
    "critique4_self_preference": b7,
    "critique5_transfer_provisional": b8,
    "round1_calibration_factors": r1f,
    "api_spend_usd": {"B03_rejudge": b4["total_cost_usd"], "B08_strong_judge": b8["strong_judge_cost_usd"],
                      "total": b4["total_cost_usd"] + b8["strong_judge_cost_usd"]},
    "provenance": {"scripts": "/Users/challagullayoshitha/Downloads/agentprop_revision/scripts/B_*.py",
                   "bootstrap": f"percentile, B={N_BOOT}, seed={SEED}, resampling traces"},
}
p2 = load_p2()
OPEN = {"qwen25_3b", "qwen25_7b", "qwen25_14b", "hermes3_llama8b"}
ids = sample_ids()
art = {"calibration_answers_over_300_chars": sum(len(p2[s]["final_answer"] or "") > 300 for s in ids),
       "calibration_empty_query": sum(not p2[s]["query"] for s in ids),
       "calibration_eps_ge2": sum(int(p2[s]["eps"]) >= 2 for s in ids)}
for fam_, f in [("open_weight", True), ("proprietary", False)]:
    rs = [r for r in p2.values() if (r["model"] in OPEN) == f]
    art[fam_] = {"n": len(rs), "frac_answers_over_300_chars": sum(len(r["final_answer"] or "") > 300 for r in rs) / len(rs),
                 "frac_empty_query": sum(not r["query"] for r in rs) / len(rs)}
results["input_artifacts"] = art
save_json(results, "results.json")
# wrap tabulars so they fit \linewidth (requires graphicx)
for fn in os.listdir(TEX):
    t = open(os.path.join(TEX, fn)).read()
    if "resizebox" not in t:
        t = t.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}").replace(
            "\\end{tabular}", "\\end{tabular}}")
        open(os.path.join(TEX, fn), "w").write(t)
print("ok", results["api_spend_usd"])
