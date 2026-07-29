# Corrected Claims Diff — AgentHallu-Bench
## Old Claim → New Claim → Evidence

Use this as the backbone for the **Response to Editor** document.

---

### ABSTRACT / HIGHLIGHTS

| # | Old Claim | New Claim | Evidence |
|---|-----------|-----------|----------|
| 1 | "2,000 tasks spanning four domains" | "2,000 task specifications across four domains; 1,350 semantic-wrong injection traces evaluated across 9 models" | `full_2000_tasks.json` has 2,000 tasks; `p2_all.csv` has 1,350 semantic_wrong rows (200 unique tasks × 9 models at varying n) |
| 2 | "propagates to a wrong final answer with ~100% conditional probability across every model tested" | "Propagation is substantial but highly variable: human-calibrated r₂₃ ≈ 0.62 (mean), range 0.46–0.73 across models, with wide confidence intervals (n=100 calibration sample). Ensemble-judged r₂₃ = 0.78 overall (range 0.43–0.97). The drop from 1.00 to ~0.62 reflects both the correction of a code bug and the shift from heuristic to ensemble+human-calibrated judging." | `p2_all_rejudged.csv`: 123/158 S2 rows ensemble_correct=False; calibration: P(human=correct\|ens=correct)=0.76, P(human=correct\|ens=wrong)=0.25 from 100 labels |
| 3 | "parameter-stage rejection is the *only* effective filter" | "Parameter-stage rejection is the highest-leverage filter, but models that accept injections still recover 3–57% of the time depending on architecture" | Recovery rates from rejection-vs-recovery analysis; `fig_rejection_vs_recovery.png` |
| 4 | "κ=0.14 agreement with a 3-LLM ensemble judge (κ=0.77–0.82)" | "κ=0.05 (human-vs-heuristic), validating the heuristic critique. Ensemble-vs-human κ=0.43 (moderate), indicating the ensemble is a significant improvement but errs conservative" | 100 human labels: κ_human_heur=0.049, κ_human_ens=0.432; `ensemble_judge_metrics.json` for inter-LLM κ |
| 5 | "F₁=0.842 on a tuned validation sweep" | "F₁=0.842 on a 40-row validation sweep (12 configs); this is a sweep-internal optimum, not a held-out estimate" | `interceptor_tuned_thresholds.json`; sweep size = 20 tasks × 2 models × 2 conditions = 80 rows, ~40 P2 |
| 6 | "GPT-4o-mini: 60.0% → 32.8%, Cohen's h=0.55" | "GPT-4o-mini: 55.8% → 32.8% (−23.0pp) on concurrent no-interceptor control (same 600 tasks, same time). Historical baseline was 60.0% (different task pool, n=450)" | `interceptor_control_600.csv` vs `interceptor_heldout_600.csv`; paired on 600 task_ids |
| 7 | "Gemini-2.0-Flash: 52.4% → 43.2%, h=0.19, power 0.80" | "Gemini-2.0-Flash: 44.5% → 43.2% (−1.3pp) with concurrent control. Not significant." | Same files; concurrent control rate 44.5% vs historical 52.4% |
| 8 | "the first statistically significant per-model hallucination reduction" | "A statistically significant 23pp hallucination reduction for GPT-4o-mini under concurrent control. Gemini-2.0-Flash shows no significant effect." | Power analysis needs recomputation with new n_A=600 |
| 9 | "All code, data, and traces are released" | "All code, data, and traces are released" (now actually true) | `metrics/`, `agent/`, `injector/`, `tracer/`, `config.py` pushed to GitHub 2026-04-17 |

---

### SECTION 4.2 — HOP ANALYSIS (Table 2 in paper)

| Old | New | Evidence |
|-----|-----|----------|
| p_S2 = p_S3 for all 9 models (e.g., deepseek 28.5/28.5) | p_S2 ≠ p_S3 (e.g., deepseek p_S2=7.0%, p_S3=29.5%). Old bug: `classify_stage()` conflated S2 and S3 via `reached_output` override | Fixed `hop_analysis.py`; `p2_all.csv` cross-tab: 339 rows have S3=1 & S2=0 |
| r₂₃ = 1.00 for all models | r₂₃ ≈ 0.62 human-calibrated mean (range 0.46–0.73); 0.78 ensemble-judged (range 0.43–0.97). The 16pp drop from ensemble to calibrated reflects the ensemble's conservative bias (rejects answers humans accept). | `p2_all_rejudged.csv`: 123/158 S2 rows wrong per ensemble; calibration from 100 human labels (wide CIs) |
| "4-stage cascade" S1→S2→S3→S4 | 3-stage model: S1 (injection applied), S2 (visible execution), S3 (wrong answer). Stages independently computed. | Stage indicators use separate columns: eps for S1/S2, ensemble_correct for S3 |
| "Saturated propagation" | "Substantial but highly variable propagation" — human-calibrated r₂₃ mean ~0.62 with per-model range spanning 0.46–0.73. n_S2 is small for some models (2–45 in semantic_wrong), contributing to wide CIs. | Per-model n_S2 in `hop_rates.json`; calibration CIs limited by n=100 human labels |
| Used all 4 error types (filter was broken) | Properly filtered to `error_type == 'semantic_wrong'` (n=1,350) | Fixed `hop_analysis.py` line 54: checks `error_type` column, not `injection_type` |

---

### SECTION 4.1 — MAIN RESULTS (Table 1)

| Old | New | Evidence |
|-----|-----|----------|
| P2 correctness HIGHER than baseline (e.g., gpt4o_mini 40%→51%) | P2 correctness LOWER than baseline under ensemble judging (gpt4o_mini 40%→29%). Paradox was caused by heuristic over-counting. | `p2_all_rejudged.csv`: ensemble correct rate 29.0% vs heuristic 49.0% for gpt4o_mini |
| "Correctness judged by 3-LLM ensemble" (Table 1 caption) | Baseline used heuristic; P2 now has ensemble judging. Baseline needs re-judging for fair comparison, or caption must specify. | `run_p2_experiment.py` line 171 uses substring heuristic; ensemble ran separately |
| Heuristic correct rate ~64% on P2 | Ensemble says 30%, human-calibrated ~40%. Heuristic was accepting paraphrased-wrong answers as correct. | 100 human labels: heuristic correct rate 65%, human 38%, ensemble 25% |

---

### SECTION 4.4 — JUDGE EVALUATION

| Old | New | Evidence |
|-----|-----|----------|
| "κ=0.143 heuristic vs 3-LLM majority" | Confirmed. Human-vs-heuristic κ=0.049 (even worse than ensemble-vs-heuristic). | 100 human labels |
| "3 LLM judges reach substantial-to-almost-perfect agreement (κ=0.77–0.82)" | Inter-LLM κ stands. But ensemble-vs-human κ=0.432 (moderate), so ensemble is not ground truth — it's the best available automated judge. | `ensemble_judge_metrics.json` for inter-LLM; 100 human labels for human-vs-ensemble |
| No human validation of ensemble | 100 human labels provide calibration. Ensemble has conservative bias (rejects 19 answers humans accept per 100 traces). | Confusion matrix: 19 human=YES/ensemble=NO out of 100 |

---

### SECTION 4.5 — INTERCEPTOR

| Old | New | Evidence |
|-----|-----|----------|
| "60.0% → 32.8% (−27.2pp)" for gpt4o_mini | "55.8% → 32.8% (−23.0pp)" with concurrent control on same 600 tasks. 25 "broken" traces are LLM run-to-run variance (0 abstentions, no domain clustering). | `interceptor_control_600.csv` + `interceptor_heldout_600.csv` paired analysis |
| "52.4% → 43.2% (−9.3pp)" for Gemini | "44.5% → 43.2% (−1.3pp)" — not significant | Same files |
| "n_A = 450" in power analysis (historical baseline) | n_A = 600 (concurrent control). Power analysis must be recomputed. | `power_analysis.json` had n_A=450 from baseline_all.csv |
| F₁=0.842 presented as primary result | F₁=0.842 is sweep-internal optimum. Held-out F₁=0.000 under original labels (label mismatch). Primary evidence is now the concurrent hallucination reduction. | `interceptor_tuned_thresholds.json`; held-out analysis |
| (implicit) Absolute hallucination rates are trustworthy | The −23pp delta is measured under heuristic judging applied identically to both arms — robust to shared judging bias. However, the absolute hallucination rates in each arm (55.8% and 32.8%) have not been validated by human labeling. Calibrating these absolutes would require additional human annotation of the interceptor runs, which we leave to future work. | Limitation: P2-trace calibration does not transfer to vanilla agent runs (different heuristic accuracy regime; see +0.6pp calibrated delta as evidence of non-transferability) |

---

### SECTION 4.6 — SELF-CONSISTENCY

| Old | New | Evidence |
|-----|-----|----------|
| "n=30 per condition, directional only" | No change needed — paper already flagged this as underpowered | `self_consistency_results.csv` |

---

### NEW SECTION — Rejection vs Recovery Decomposition (proposed Section 4.3 or 5)

This is a new result that deserves its own subsection, not a bullet in the discussion. It is the strongest structural finding to survive all corrections.

**Proposed section title**: "Rejection and recovery are independent model capabilities"

**Content**:
- Define rejection rate = 1 − P(S1) = fraction of injection attempts the model rejects at the parameter stage
- Define recovery rate = 1 − r₂₃ = fraction of visibly-executed injections where the model still produces a correct final answer
- Plot: 9-model scatter with Spearman ρ = 0.126 (p = 0.747) — axes are statistically independent
- Key examples: gemini_20_flash (best rejector at 95%, decent recovery at 50%); gpt41_nano (mediocre rejector at 37.5%, worst recovery at 2.9%); gpt4o_mini (poor rejector at 40%, poor recovery at 15.6%)
- Implication: "A model's ability to reject bad parameters at the schema gate tells you nothing about its ability to recover from bad observations. These are architecturally distinct failure modes, suggesting that mitigation strategies should target both independently."
- Figure: `fig_rejection_vs_recovery.png` (already generated)

**Why this matters for IP&M**: This gives practitioners a 2D framework for evaluating model robustness to tool-use errors, rather than a single correctness number. It directly answers the editor's request for theoretical and practical implications.

| Claim | Evidence |
|-------|----------|
| "Rejection rate and recovery rate are independent model capabilities (Spearman ρ=0.126, p=0.747)" | `fig_rejection_vs_recovery.png`; per-model data in `hop_rates.json` |
| "Injection-to-error rate P(S3\|S1) ranges from 53–90% across models (ensemble-judged), with human-calibrated estimates of 53–70%" | `p2_all_rejudged.csv` + calibration from 100 human labels |

---

### CONCLUSION / DATA RELEASE

| Old | New | Evidence |
|-----|-----|----------|
| "2,300 agent traces" | Must recount accurately: 1,350 semantic-wrong + 2,750 baseline + 1,200 interceptor + 600 multi-seed + 120 self-consistency + 300 retail baseline + 300 retail P2 = 6,620 primary traces | CSV row counts |
| Code release incomplete | All source code now on GitHub including `metrics/eps_scorer.py` | Pushed 2026-04-17, commit 429bd06 |
