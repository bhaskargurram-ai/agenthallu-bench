# AgentHallu-Bench — Paper Build

## Structure

```
paper/
  main.tex            # Main manuscript (all sections + appendices)
  refs.bib            # BibTeX (53 curated references, compiled 2026-04-15)
  build_tables.py     # Regenerates tables/*.tex from data/results/*
  tables/
    table1_main.tex   # Baseline + P2 correctness by model (9 models)
    table2_hop.tex    # Stage-reach probabilities + hop-conditional rates
    table3_judge.tex  # Per-rater Cohen's kappa (heuristic vs 3 LLM judges)
    table4_sweep.tex  # Interceptor threshold sweep (12 configs)
    table5_retail.tex # Cross-domain replication on retail tasks
    tableA4_power.tex # Achieved statistical power per headline claim
  figures/
    fig_sankey.png    # 4-stage error cascade Sankey diagram
```

## Rebuild tables from raw results

```bash
venv/bin/python paper/build_tables.py
```

Inputs read:
- `data/results/baseline_all.csv`, `data/results/p2_all.csv`
- `data/results/hop_rates.json`
- `data/results/ensemble_judge_metrics.json`
- `analysis/interceptor_tuned_thresholds.json`
- `data/results/baseline_retail.csv`, `data/results/p2_retail.csv`
- `data/results/power_analysis.json`

## Compile to PDF

### Option A — Overleaf (recommended, no local install)

1. Create a new Overleaf project.
2. Upload everything in `paper/` (preserve directory structure).
3. Set compiler to `pdfLaTeX`.
4. Compile. `bibtex` runs automatically.

### Option B — Local MacTeX

```bash
brew install --cask mactex-no-gui        # one-time
cd paper
pdflatex main
bibtex  main
pdflatex main
pdflatex main
```

Produces `main.pdf`.

## Notes

- All numerical values in tables come directly from the JSON/CSV result
  files under `data/results/` — no hand-edited numbers.
- Bootstrap CIs use `analysis/bootstrap.py::bootstrap_ci` (1,000 resamples).
- `refs.bib` is curated from WebSearch over arXiv / ACL Anthology /
  NeurIPS / ICLR / EMNLP proceedings on 2026-04-15; see the trailing
  `KEY GAPS IN PRIOR WORK` block in the bib for the gap-analysis used
  to frame the introduction.
