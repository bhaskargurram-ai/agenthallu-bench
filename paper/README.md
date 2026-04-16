# AgentHallu-Bench — Elsevier `elsarticle` submission package

Submission-ready LaTeX source for the paper
**"AgentHallu-Bench: Measuring and Mitigating Error Propagation in Tool-Using Language Agents"**,
formatted with the Elsevier `elsarticle` document class.

## Contents

```
paper_elsarticle/
  main.tex                    # Manuscript (elsarticle; frontmatter + highlights + keywords)
  refs.bib                    # BibTeX references (peer-reviewed venues where available)
  build_tables.py             # Regenerates tables/*.tex from data/results/*
  tables/
    table1_main.tex
    table2_hop.tex
    table3_judge.tex
    table4_sweep.tex
    table5_retail.tex
    tableA4_power.tex
  figures/
    fig_sankey.png
  elsarticle.dtx              # Source of the elsarticle class (Elsevier Ltd)
  elsarticle.ins              #   (install with `latex elsarticle.ins` to generate .cls)
  elsarticle-num.bst          # Numbered-reference BibTeX style (used here)
  elsarticle-num-names.bst    # Alternative: numbered with author names
  elsarticle-harv.bst         # Alternative: author-year (Harvard)
```

## How to compile

### Option A — Overleaf (recommended, zero install)

1. Create a new Overleaf project and upload the contents of this folder
   (preserving the `tables/` and `figures/` subdirectories).
2. Set the compiler to **pdfLaTeX**.
3. Click **Recompile**. `elsarticle.cls` is already on Overleaf's TeX Live.
4. `bibtex` is run automatically when the `.bib` file is detected.

### Option B — Local TeX Live / MacTeX

```bash
# one-time install if needed
brew install --cask mactex-no-gui

cd paper_elsarticle
# on most TeX Live installations elsarticle.cls is preinstalled; if not:
#   latex elsarticle.ins
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Output: `main.pdf`.

## Journal-layout variants

The current `\documentclass` is `[preprint,11pt]`, which gives single-column
double-spaced output suitable for review. To switch to final journal layout,
change the first line of `main.tex` to one of:

```latex
\documentclass[final,1p,times]{elsarticle}   % single column
\documentclass[final,3p,times]{elsarticle}   % two column, narrow
\documentclass[final,5p,times,twocolumn]{elsarticle}   % 5p format
```

## Regenerating tables from raw results

```bash
venv/bin/python build_tables.py
```

Reads (from the parent `agenthallu_bench/` repo):
- `data/results/baseline_all.csv`, `data/results/p2_all.csv`
- `data/results/hop_rates.json`
- `data/results/ensemble_judge_metrics.json`
- `analysis/interceptor_tuned_thresholds.json`
- `data/results/baseline_retail.csv`, `data/results/p2_retail.csv`
- `data/results/power_analysis.json`

All numerical values in `tables/*.tex` come directly from these files — no
hand-edited numbers.

## Notes on references

`refs.bib` has been curated to cite peer-reviewed venues (NeurIPS, ICLR,
ACL, EMNLP, TACL, ICML, ACM TOIS, etc.) wherever the cited paper has been
published. Entries that remain as `@article{...arXiv preprint...}` are
genuine preprints with no confirmed archival venue at the time of
compilation (2026-04-15); each such entry carries an inline comment
explaining why it was not upgraded.

## Changelog vs.\ the arXiv preprint source

The manuscript contents are identical to the arXiv preprint source
(`paper/main.tex`) with only the following format-level changes:

- `\documentclass[11pt]{article}` → `\documentclass[preprint,11pt]{elsarticle}`
- Title/author/abstract/maketitle block → `\begin{frontmatter}...\end{frontmatter}`
- Added Elsevier-required `highlights` and `keyword` environments
- Switched `\bibliographystyle{plainnat}` → `\bibliographystyle{elsarticle-num}`
- Removed redundant `\usepackage{geometry,xcolor,natbib,caption,subcaption,algorithm,algpseudocode}`
  since `elsarticle` provides or supersedes these
