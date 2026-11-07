# Phase 2 Status Report

## Completed

1. **Section A: Rename** — AgentHallu-Bench → AgentProp-Bench throughout. CHANGELOG.md created.
2. **refs.bib** — 63 BibTeX entries from literature CSV. APA-compatible. All 19 manuscript citations resolved.
3. **7 figures** — All generated from corrected data at 300 DPI: fig_stage_reach, fig_rejection_vs_recovery, fig_judge_agreement, fig_confusion_matrix, fig_interceptor_effect, fig_domain_breakdown, fig_calibration.
4. **12 tables** — 8 main + 4 supplementary. All generated from CSVs with bootstrap CIs.
5. **manuscript_anonymous.tex** — Full 9-section rewrite. Evaluation-focused positioning per Section D. All corrected numbers. 19 citations. Inline source comments. Transparency note on stage correction.
6. **title_page.tex** — Author info, CRediT, declarations, ethics, acknowledgments.
7. **highlights.tex** — 5 Elsevier-format highlights.
8. **cover_letter.tex** — Addresses editor, summarizes 8 responses + self-disclosed correction.
9. **Response_to_Editor.tex** — All 8 editor points with quoted text + Section 9 on methodological corrections.
10. **suplement.tex** — Threshold sweep, self-consistency, retail, ablation, power analysis, domain breakdown, calibration, judge prompt.
11. **AgentProp_Bench_submission_files.zip** — 687 KB, 32 files.

## Assumptions Made

1. **Abstract word count ~195**: Estimated from character count. Cannot verify without LaTeX compilation.
2. **Manuscript ~9,500 words**: Estimated. Actual count requires compilation.
3. **elsarticle.cls available on Overleaf**: Standard Elsevier class, should be pre-installed.
4. **APA formatting via elsarticle-num-names.bst**: This produces numbered references with author names, which is APA-compatible but not pure APA author-year. The editor said "IP&M uses APA for published articles if this helps" — the .bst choice may need adjustment if the editor specifically requires author-year.

## TODOs (Cannot Be Resolved From Provided Data)

1. **Compile on Overleaf**: No pdflatex on this machine. Must upload zip to Overleaf and verify zero compilation errors before submission.
2. **Generate Response_to_Editor.pdf**: Compile the .tex to get the PDF the editor expects.
3. **Prune orphan bib entries**: refs.bib has 63 entries but only 19 are cited. Run `bibtex` warnings to identify and remove orphans if desired.
4. **Baseline ensemble judging**: Table 4 compares heuristic-judged baseline with ensemble-judged P2. For a fully fair comparison, baseline should also be ensemble-judged. This requires ~2,750 API calls (~$1.50). The manuscript notes this asymmetry.
5. **ORCID**: title_page.tex has a placeholder comment for ORCID. Fill if available.
6. **Second annotator labels**: Only 7 overlapping labels from Yoshitha. More overlap labels would strengthen inter-annotator agreement reporting.

## File Counts

| Category | Count |
|----------|-------|
| .tex files | 6 |
| .bib files | 1 (63 entries) |
| Figures (.png) | 7 |
| Tables (.tex) | 12 |
| .bst files | 3 |
| Other (.dtx, .ins) | 2 |
| **Total in zip** | **32** |
