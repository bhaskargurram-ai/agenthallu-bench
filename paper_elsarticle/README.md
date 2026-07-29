# AgentProp-Bench — Elsevier `elsarticle` submission package

> Repository previously published as AgentHallu-Bench; see CHANGELOG.md for rename.

Submission-ready LaTeX source for the paper
**"Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench"**,
formatted with the Elsevier `elsarticle` document class, targeting
Information Processing & Management (IP&M).

## Contents

```
paper_elsarticle/
  manuscript_anonymous.tex      # Anonymized manuscript
  title_page.tex                # Non-anonymized title page
  highlights.tex                # Elsevier highlights (5 items)
  cover_letter.tex              # Cover letter to editor
  Response_to_Editor.tex        # Detailed response to editor feedback
  suplement.tex                 # Supplementary materials
  refs.bib                      # BibTeX references (APA-compatible)
  figures/                      # All figures (300 DPI PNGs)
  tables/                       # LaTeX table files
  elsarticle.dtx                # Elsevier class source
  elsarticle.ins                # Class installer
  elsarticle-num-names.bst      # APA-style BibTeX style
  elsarticle-harv.bst           # Harvard style (alternative)
  elsarticle-num.bst            # Numbered style (alternative)
```

## How to compile

```bash
cd paper_elsarticle
pdflatex manuscript_anonymous
bibtex   manuscript_anonymous
pdflatex manuscript_anonymous
pdflatex manuscript_anonymous
```

## Data repository

All code, data, traces, and human labels:
https://github.com/bhaskargurram-ai/agenthallu-bench
