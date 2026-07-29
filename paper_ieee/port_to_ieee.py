"""Port paper_elsarticle/manuscript_anonymous.tex -> paper_ieee/main.tex (IEEE Access).

Preserves the body prose verbatim; mechanically converts natbib \citep/\citet to
IEEE \cite; replaces the elsarticle preamble/frontmatter/bibliography with IEEE
equivalents; appends a de-anonymised author bio. Spots that need open-weight-model
number updates are tagged with '% TODO-OPEN'.
"""
import re
import os

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "paper_elsarticle", "manuscript_anonymous.tex")

lines = open(SRC).read().split("\n")
start = next(i for i, l in enumerate(lines) if "\\section{Introduction}" in l)
end = next(i for i, l in enumerate(lines) if "\\bibliographystyle" in l)
body = "\n".join(lines[start:end])

# natbib -> IEEE numeric \cite (drop optional [..] args and \citet/\citep distinction)
body = re.sub(r"\\cite[pt]\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{", r"\\cite{", body)

PREAMBLE = r"""\documentclass{ieeeaccess}
\usepackage{cite}
% natbib compatibility: table/figure captions generated for the Elsevier build
% use \citep/\citet; map them onto IEEE numeric \cite so no re-editing is needed.
\providecommand{\citep}{\cite}
\providecommand{\citet}{\cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{tabularx}
\usepackage{url}
\usepackage{xspace}

\newcommand{\bench}{\textsc{AgentProp-Bench}\xspace}
\newcommand{\eps}{\mathrm{EPS}}
\renewcommand{\Pr}{\mathbb{P}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\interceptor}{\textsc{Interceptor}\xspace}

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2024.0429000}

\title{Evaluating Tool-Using Language Agents: Judge Reliability, Propagation
Cascades, and Runtime Mitigation in \bench{}}
\author{\uppercase{Bhaskar Gurram}\authorrefmark{1}}
\address[1]{Zasti Inc., Ashburn, VA 20147 USA (e-mail: gurrambhaskar.ai@gmail.com)}
\markboth{Gurram: Evaluating Tool-Using Language Agents in AgentProp-Bench}
{Gurram: Evaluating Tool-Using Language Agents in AgentProp-Bench}
\corresp{Corresponding author: Bhaskar Gurram (e-mail: gurrambhaskar.ai@gmail.com).}

\begin{abstract}
Automated evaluation of tool-using large language model (LLM) agents is widely
assumed to be reliable, yet this assumption has rarely been validated against
human annotation. We present AgentProp-Bench, a benchmark of 2,000 tasks across
four domains with a 100-label human-validated subset, and use it to study three
questions: judge reliability, error propagation, and runtime mitigation. We
evaluate nine proprietary LLMs and four open-weight LLMs served locally, all
through an identical prompted tool-use harness. % TODO-OPEN: confirm 9 + 4 = 13 and update counts
Substring-based judging agrees with human annotation at kappa = 0.049
(chance level), whereas a three-LLM ensemble judge reaches kappa = 0.432
(moderate) with a conservative bias. Under validated evaluation, a
parameter-level injection propagates to a wrong final answer with a
human-calibrated probability of about 0.62 (per-model range 0.46 to 0.73).
% TODO-OPEN: update propagation range with open-weight models
Rejection (catching a bad parameter) and recovery (correcting after acceptance)
are statistically independent model capabilities (Spearman rho = 0.126,
p = 0.747). A tuned runtime interceptor reduces hallucination on one model by
23.0 percentage points under a concurrent n = 600 control, but has no
significant effect on a model whose aggressive parameter rejection already
eliminates the target failure mode. All code, data, traces, and human labels
are released.
\end{abstract}

\begin{keywords}
Artificial intelligence, benchmark testing, error analysis, evaluation,
large language models, natural language processing, software agents,
tool-using agents.
\end{keywords}

\titlepgskip=-15pt
\maketitle
"""

TAIL = r"""
\section*{Acknowledgment}
The author thanks Yoshitha Challagulla for independently annotating a subset of
traces used in the inter-annotator agreement analysis.

\bibliographystyle{IEEEtran}
\bibliography{refs}

\begin{IEEEbiographynophoto}{Bhaskar Gurram}
received the M.S. degree in computer science. He is currently a Machine Learning
Scientist with Zasti Inc., Ashburn, VA, USA. His research interests include the
evaluation and reliability of tool-using large language model agents, retrieval
and information systems, and runtime monitoring of autonomous agents.
\end{IEEEbiographynophoto}

\EOD

\end{document}
"""

out = PREAMBLE + "\n" + body + "\n" + TAIL
open(os.path.join(HERE, "main.tex"), "w").write(out)
print("wrote paper_ieee/main.tex")
print("citations converted; body lines", start, "..", end)
print("remaining \\citep/\\citet:", len(re.findall(r"\\cite[pt]\{", out)))
print("TODO-OPEN markers:", out.count("TODO-OPEN"))
