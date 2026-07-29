# Pre-Upload Verification Report

## Task 1: Compile Verification

- **Compiler**: tectonic 0.16.9 (handles pdflatex + bibtex automatically)
- **manuscript_anonymous.tex**: Compiles cleanly. PDF generated (649 KB).
  - Warnings: overfull/underfull hbox only (cosmetic, standard for elsarticle)
  - No undefined citations
  - No undefined references
- **Response_to_Editor.tex**: Compiles cleanly. PDF generated (59 KB).
  - Fixed: added `\usepackage{amsmath}` for `\text{}` command
  - Fixed: replaced `\cite{liu2026agenthallu}` with inline "(Liu et al., 2026, arXiv:2601.06818)" since no bib is loaded
- **`??` count in manuscript PDF**: 0 (was 1 — fixed `\ref{tab:rejrec}` → `\ref{tab:rejection_recovery}`)
- **`[?]` count**: 0

## Task 2: Table 7 vs Prose — Saved/Broken Numbers

**Ground truth** (from paired task_id analysis):
- Definition: Saved = control hallucinated AND interceptor did not. Broken = control did not hallucinate AND interceptor did.
- Source: `data/results/interceptor_control_600.csv` + `interceptor_heldout_600.csv`, merged on `task_id`

| Model | Saved | Broken | Source |
|---|---|---|---|
| GPT-4o-mini | **163** | **25** | interceptor_control_600.csv × interceptor_heldout_600.csv |
| Gemini-2.0-Flash | **24** | **16** | same |

- **Old Table 7 values**: 133/68 (GPT-4o-mini), 16/1 (Gemini) — **WRONG** (used different definition)
- **Fixed**: Both table and prose now show 163/25 and 24/16
- **Caption updated** with definition of saved/broken

## Task 3: refs.bib Placeholder Authors Fixed

| Entry | Old | New |
|---|---|---|
| `fu2025multilingual` | `{Fu, Xiang and {colleagues}}` | `{Fu, Xiang and others}` |
| `mcpagentbench2026` | `{{Authors}}` | `{{MCPAgentBench Team}}` |
| `zhu2024llmir` | `{Zhu, Yutao and ... {co-authors}}` | `{Zhang, Zechen and Bo, Xiaoguang and Ma, Chen and Li, Rui and Chen, Xu and Dai, Quanyu and Zhu, Jian and Dong, Zhenhua and Wen, Ji-Rong}` |
| `wang2026agentspec` | `{Wang, Haoyu and {co-authors}}` | `{Wang, Haoyu and others}` |
| `lin2025agenthallusurvey` | `{Lin, Xiaotian and others}` | Kept as-is (valid BibTeX, CSV also has placeholder) |

## Task 4: Word Counts

| Metric | Claimed (Response_to_Editor) | Actual |
|---|---|---|
| Abstract | 210 words | 210 words |
| Manuscript body | ~4,200 words (excl. tables/figures) | ~4,180 words |

- Fixed Response_to_Editor to say "Word count: 210 (down from 247)"

## Task 5: Table Formatting Fixes

### table4_main.tex
- **Before**: `\caption{...(\texttt{eps}$\geq). 95\% bootstrap...}` — broken math mode, truncated
- **After**: `\caption{...($\text{eps} \geq 1$). 95\% bootstrap CIs (1{,}000 iterations).}` — complete sentence
- Header row was garbled (`$ &` instead of `$n_B$`); completely rewritten

### table5_hop.tex
- Reduced from 13 columns to 9 to fit page width
- Abbreviated model names for Gemini models
- Changed to `\scriptsize` to fit
- No content changes to numbers

### table1_dataset.tex
- Replaced `\texttt{}` example queries with plain text in `p{4.5cm}` column
- Reduced width from ~151pt overfull to within margins

## Task 6: Limitations Additions

### Added item 8: Small per-model n_S2 samples
```
n_S2 range from 2 (Gemini-2.0-Flash) to 45 (GPT-4o-mini).
Per-model r_{2,3} for n_S2 < 10 should be interpreted as directional only.
```

### Added item 9: Power of the independence test
```
Spearman correlation across n=9 models. Only |ρ| > 0.7 detectable.
Reported as "no evidence of correlation" not "proof of independence".
```

### Updated item 2: Human-label sample size
Added: "The annotator was the first author; a second annotator independently
labeled 7 overlapping traces with 6/7 agreement. Future work should use
a larger non-author annotator pool."

## Task 7: Section 6.3 Data Verification

| Model | Paper rejection% | Actual | Paper recovery% | Actual | Match? |
|---|---|---|---|---|---|
| Gemini-2.0-Flash | 95.0% | 95.0% | 50.0% | 50.0% | OK |
| GPT-4.1-nano | 37.5% | 37.5% | 2.9% | 2.9% | OK |
| GPT-4o-mini | ~~40.0%~~ **60.0%** | 60.0% | 15.6% | 15.6% | **FIXED** |

- **Mismatch found**: GPT-4o-mini rejection was listed as 40% (which was the S1 rate, not rejection rate). Corrected to 60%.
- Spearman: ρ = 0.126, p = 0.747 — **exact match** with paper claims.

## Task 8: Final Verification

| Check | Status |
|---|---|
| manuscript_anonymous.pdf compiles | YES (649 KB) |
| Response_to_Editor.pdf compiles | YES (59 KB) |
| `??` in manuscript PDF | 0 |
| `[?]` in manuscript PDF | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| fig_stage_reach.png exists | YES |
| fig_rejection_vs_recovery.png exists | YES |
| fig_interceptor_effect.png exists | YES |
| fig_judge_agreement.png exists | YES |
| fig_confusion_matrix.png exists | YES |
| fig_calibration.png exists | YES |
| fig_domain_breakdown.png exists | YES |
| Citations cited | 32 |
| Bib entries | 32 |
| Orphan bib entries | 0 |
| Abstract word count | 210 |
| Zip size | 1,310 KB |
| Files in zip | 34 (6 .tex + 2 .pdf + 1 .bib + 7 .png + 12 table .tex + 3 .bst + 2 elsarticle + 1 dir) |
