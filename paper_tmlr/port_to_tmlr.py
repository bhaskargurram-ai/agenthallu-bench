"""Port paper_ieee/main.tex -> paper_tmlr/main.tex (TMLR anonymized submission)."""
import re, os
HERE = os.path.dirname(__file__)
src = open(os.path.join(HERE, "..", "paper_ieee", "main.tex")).read()
lines = src.split("\n")

ab_s = next(i for i, l in enumerate(lines) if "\\begin{abstract}" in l)
ab_e = next(i for i, l in enumerate(lines) if "\\end{abstract}" in l)
abstract = "\n".join(lines[ab_s + 1:ab_e]).strip()

b_s = next(i for i, l in enumerate(lines) if "\\section{Introduction}" in l)
b_e = next(i for i, l in enumerate(lines) if "\\section*{Declarations}" in l)
body = "\n".join(lines[b_s:b_e]).rstrip()

ANON = "https://anonymous.4open.science/r/AgentProp-Bench"
GH = "https://github.com/bhaskargurram-ai/agenthallu-bench"
body = body.replace(GH, ANON); abstract = abstract.replace(GH, ANON)
# natbib author-year: prefer \citep
body = re.sub(r"\\cite\{", r"\\citep{", body)
# single-column: de-star floats
for a, b in [("\\begin{figure*}", "\\begin{figure}"), ("\\end{figure*}", "\\end{figure}"),
             ("\\begin{table*}", "\\begin{table}"), ("\\end{table*}", "\\end{table}")]:
    body = body.replace(a, b)

# ---- keep main <= 12 pages: relocate figures + retail + confusion table to an appendix ----
appendix_blocks = []
# 1) all figures (they duplicate tables); \ref still resolves to the appendix
figs = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}\n?", body, re.DOTALL)
for f in figs:
    body = body.replace(f, "")
appendix_blocks.append("\\section{Figures}\n" + "\n".join(figs))
# 2) cross-domain retail replication subsection (last Results subsection)
m = re.search(r"% 6\.7 Cross-domain replication.*?(?=\n% =+\n\\section\{Discussion|\n\\section\{Discussion)", body, re.DOTALL)
if m:
    body = body.replace(m.group(0), "")
    appendix_blocks.append("\\section{Cross-Domain Replication (Retail)}\n" +
                           re.sub(r"^%.*\n|\\subsection\{[^}]*\}\n|\\label\{[^}]*\}\n", "", m.group(0)))
# 3) confusion-matrix table input -> appendix
if "\\input{tables/table3_confusion}" in body:
    body = body.replace("\\input{tables/table3_confusion}", "")
    appendix_blocks.append("\\section{Ensemble--Human Confusion Matrix}\n\\input{tables/table3_confusion}")
# 4) dataset-breakdown table -> appendix (domain counts remain in prose)
if "\\input{tables/table1_dataset}" in body:
    body = body.replace("\\input{tables/table1_dataset}", "")
    appendix_blocks.append("\\section{Dataset Breakdown}\n\\input{tables/table1_dataset}")
# 5) related-benchmark comparison table (tab:diff) -> appendix
mdiff = re.search(r"\\begin\{table\}\[h\].*?\\label\{tab:diff\}.*?\\end\{table\}\n?", body, re.DOTALL)
if mdiff:
    body = body.replace(mdiff.group(0), "")
    appendix_blocks.append("\\section{Comparison with Related Benchmarks}\n" + mdiff.group(0))
# 6) long Limitations enumerate -> appendix; keep a concise paragraph in main
LIM_SUMMARY = (r"We summarize the main limitations here and give the full list in "
    r"Appendix~\ref{app:limits}. (i)~The ensemble judge has a conservative bias and "
    r"only moderate agreement with humans ($\kappa{=}0.432$), so ensemble-judged rates are "
    r"lower bounds; we therefore also report human-calibrated numbers. (ii)~The 100 human "
    r"labels come from two annotators with almost-perfect agreement ($\kappa{=}0.835$) but a "
    r"similar background. (iii)~Several per-model $n_{S_2}$ samples are small (2--10), so those "
    r"$r_{2,3}$ estimates are directional; the marginal injection-to-error rate is more stable. "
    r"(iv)~Tools are deterministic simulators, injection is single-parameter, and we cover the "
    r"P2 protocol only. (v)~Interceptor absolute rates are unvalidated by human labeling (the "
    r"$-23$~pp delta is robust to shared judging bias).")
mlim = re.search(r"(\\section\{Limitations\}\s*\\label\{sec:limits\}\s*)(\\begin\{enumerate\}.*?\\end\{enumerate\})", body, re.DOTALL)
if mlim:
    body = body.replace(mlim.group(2), LIM_SUMMARY)
    appendix_blocks.append("\\section{Detailed Limitations}\n\\label{app:limits}\n" + mlim.group(2))
APPENDIX = "\n\n\\appendix\n" + "\n\n".join(appendix_blocks) + "\n"

PRE = r"""\documentclass{article}
\usepackage{tmlr}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage{xspace}

\newcommand{\bench}{\textsc{AgentProp-Bench}\xspace}
\newcommand{\eps}{\mathrm{EPS}}
\renewcommand{\Pr}{\mathbb{P}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\interceptor}{\textsc{Interceptor}\xspace}

\title{Evaluating Tool-Using Language Agents: Judge Reliability,\\
Propagation Cascades, and Runtime Mitigation in \bench{}}
\author{}

\begin{document}
\maketitle

\begin{abstract}
""" + abstract + r"""
\end{abstract}

"""
TAIL = r"""

\bibliographystyle{tmlr}
\bibliography{refs}
""" + APPENDIX + r"""
\end{document}
"""
open(os.path.join(HERE, "main.tex"), "w").write(PRE + body + TAIL)
print("wrote paper_tmlr/main.tex; body lines", b_s, "-", b_e, "; anonymized:", GH not in (PRE+body+TAIL))
