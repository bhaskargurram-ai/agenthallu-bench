# Resubmission Checklist — AgentProp-Bench

## Editor Requests

| # | Editor Request | Addressed In | Evidence |
|---|----------------|--------------|----------|
| 1 | Abstract: shorter, numbers-first | Abstract (p.1), 195 words | κ=0.049, κ=0.432, r₂₃≈0.62, -23pp interceptor, all in first paragraph |
| 2 | Research objectives as own section | Section 3 | RO1 (judge reliability), RO2 (propagation), RO3 (mitigation) |
| 3 | Dataset description | Section 4 (7 subsections) | Table 1 (domain breakdown), task design, injection protocol, human labels |
| 4 | Discussion and implications | Section 7 (4 subsections) | Theoretical (7.1), agent designers (7.2), IR/IS eval (7.3), differentiation (7.4) |
| 5 | Copy editing | Throughout | Sentences <25 words, no idioms, acronyms defined, active voice |
| 6 | Manuscript length | ~9,500 words | Expanded from ~5,200 (83% increase) |
| 7 | References tied to IR/IS | Section 2, refs.bib | 19 new 2024-2026 citations; 3+ from SIGIR/CIKM/TOIS/IP&M venues |
| 8 | Author guidelines, APA | elsarticle class, elsarticle-num-names.bst | APA formatting, IP&M section structure |

## Self-Disclosed Corrections

| Correction | Disclosed In | Section |
|------------|-------------|---------|
| Stage indicators decoupled (S2≠S3) | Response_to_Editor §9, manuscript §5.2 | Transparency note |
| r₂₃: 1.00 → 0.62 (human-calibrated) | All tables, abstract | Every propagation number |
| Ensemble re-judging of P2 traces | §5.1, §6.1 | 1,350 traces re-judged |
| 100 human labels collected | §4.6, §6.1 | Calibration and validation |
| Concurrent interceptor control | §5.5, §6.4 | n=600 per model |
| Rename: AgentHallu-Bench → AgentProp-Bench | Footnote p.1, cover letter | CHANGELOG.md |

## Doneness Checks

| Check | Status |
|-------|--------|
| All .tex files in zip | 6 files: manuscript, title_page, highlights, cover_letter, Response_to_Editor, suplement |
| 19 citation keys resolved in refs.bib | All 19 found |
| 63 entries in refs.bib | Confirmed |
| 7 figures generated at 300 DPI | All present in figures/ |
| 12 tables generated (8 main + 4 supplementary) | All present in tables/ |
| "Saturated propagation" appears nowhere | Confirmed |
| "r_2 = 1.00" appears nowhere | Confirmed (transparency note references it as the OLD value) |
| AgentProp-Bench used throughout | Confirmed (old name only in footnote and CHANGELOG) |
| Sections 3, 4, 7 exist as named sections | Section 3 (ROs), Section 4 (Dataset), Section 7 (Discussion) |
| Rejection-vs-Recovery has own subsection | Section 6.3 with Figure 2 |
| Stage correction disclosed | Section 5.2 transparency note + Response_to_Editor §9 |
| Abstract 180-220 words | ~195 words |
| Zip built | AgentProp_Bench_submission_files.zip (687 KB) |

## Known Limitations / TODOs

| Item | Status | Notes |
|------|--------|-------|
| LaTeX compilation | NOT VERIFIED | No pdflatex on this machine. Must compile on Overleaf before submission. |
| Word count | ESTIMATED ~9,500 | Cannot verify without compilation. |
| Response_to_Editor.pdf | NOT GENERATED | Compile Response_to_Editor.tex on Overleaf to get PDF. |
| Orphan bib entries | NOT CLEANED | refs.bib has 63 entries but only 19 cited. Orphans are harmless but should be pruned. |
| Baseline ensemble judging | NOT DONE | Baseline column in Table 4 still uses heuristic judging. Note in manuscript. |
