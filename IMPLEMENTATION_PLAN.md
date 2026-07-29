# AgentHallu-Bench: Complete Revision & Implementation Plan
## For IP&M Resubmission

**Status**: Pre-revision plan  
**Date**: 2026-04-17  
**Sources**: Editor feedback (10 points) + Self-audit (15 issues, verified against code/data)

---

## EXECUTIVE SUMMARY

Two classes of work are needed:

1. **Code/Data fixes** — Fix real bugs in analysis code, push missing files to GitHub, re-run experiments where data is structurally flawed
2. **Paper rewrites** — Address editor's 10 revision points AND reframe claims invalidated by the audit

Estimated effort: ~1 week code fixes + ~1 week paper rewriting.

---

## PHASE 1: CRITICAL CODE & DATA FIXES (Do First)

### 1.1 Push missing `metrics/` directory to GitHub [CRITICAL]
**Why**: GitHub repo imports `from metrics.eps_scorer import EPSScorer` but `metrics/` isn't pushed. Paper cannot be reproduced.

**Action**:
- Push `metrics/eps_scorer.py`, `metrics/cascade_detector.py`, `metrics/multi_turn_scorer.py`, `metrics/stage_attribution.py`, `metrics/run_stage_attribution.py`, `metrics/__init__.py`
- Also push other missing directories needed for reproduction: `agent/`, `injector/`, `tracer/`, `config.py`
- Add `smoke_test.py` and `requirements.txt` to root
- Verify the repo can be cloned and `run_p2_experiment.py` imports succeed

**Files**: All files under `metrics/`, `agent/`, `injector/`, `tracer/`, `config.py`

---

### 1.2 Fix `hop_analysis.py` — eliminate S2==S3 bug [CRITICAL]
**Why**: `classify_stage()` maps `reached_output=True` → stage=3, then `S2=(stage>=2)` includes those rows. This makes p_S2 == p_S3 == reached_output/n for all models. The r₂₃=1.00 "saturated propagation" claim is a tautology.

**Root cause (verified)**:
```python
# CURRENT BUG: reached_output overrides eps-based stage
if reached:
    stage = max(stage, 3)  # This makes eps==1 + ro=True → stage=3 → counts for S2
```

**Data proof**:
- 339 rows have eps==1 AND reached_output=True → get stage=3 → counted in S2
- 0 rows have eps>=2 AND reached_output=False
- Result: S2 set == S3 set == {all rows where reached_output=True}

**Fix**:
```python
def classify_stage(row) -> int:
    """Independent stage classification — no cross-contamination."""
    eps = row.get("eps", 0)
    reached = bool(row.get("reached_output", False))
    
    if pd.notna(eps) and int(eps) >= 1:
        # S1: injection applied (parameter stage entered)
        if pd.notna(eps) and int(eps) >= 2:
            if reached:
                return 3  # S3: wrong final answer
            return 2      # S2: executed but answer still correct
        if reached:
            return 3      # eps==1 but wrong answer (skip S2)
        return 1          # eps==1, answer correct
    return 0              # injection didn't apply
```

BUT this still has the problem that S3 > S2 (339 rows bypass S2 to reach S3). The cascade model is broken.

**Better fix — redefine to 3-stage model**:
- S1: injection applied (eps >= 1)  
- S2: injection executed with observable effect (eps >= 2)  
- S3: final answer wrong (reached_output = True)  
- Report S2→S3 rate only for rows WHERE S2=True (eps>=2), NOT for all reached_output rows
- Report S1→S3 "skip rate" for rows where eps==1 but reached_output=True (the 339 anomalous rows)

**After fix**: Re-run `python analysis/hop_analysis.py` to regenerate `hop_rates.json` and `hop_rates.md`.

**Expected new r₂₃ values** (from verified data, computed as reached_output / eps>=2 per model):
| Model | eps>=2 | ro=T | r₂₃ (corrected) |
|---|---|---|---|
| deepseek_v3 | 160 | 228 | 1.425 (incoherent) |
| gpt4o_mini | 169 | 170 | 1.006 |

Since r₂₃ > 1.0 for most models (S3 includes rows that bypass S2), the hop-conditional model itself is broken. Must reframe.

---

### 1.3 Fix `reached_output` semantics [CRITICAL]
**Why**: `reached_output` is a perfect alias for `final_correct == False` (0 exceptions in 5400 rows). It does NOT measure "injection effect reached output stage" — it measures "answer was wrong."

**Root cause** (verified in `metrics/eps_scorer.py` lines 248-251):
```python
# Output generation: if final answer is wrong, error propagated to output
if final_correct is False:
    stages_with_error.add("output_generation")
    reached_output = True
```

**Fix options**:
- Option A: Accept that reached_output = wrong answer and rename it `final_wrong` throughout
- Option B: Implement a true observation-level check (does the tool response contain the injected error?) as the Stage 2 indicator, independent of final_correct
- **Recommended**: Option A for resubmission (honest reframing), note Option B as future work

---

### 1.4 Add ensemble judge post-processing pipeline [MAJOR]
**Why**: Paper says "Correctness judged by 3-LLM ensemble" (Table 1 caption). But `run_p2_experiment.py` uses a substring heuristic for `final_correct`. The ensemble judge (`run_ensemble_judge.py`) ran on 960 baseline traces only.

**Action**:
- Either: run ensemble judge on p2_all.csv and update final_correct values, then re-run all downstream analysis
- Or: honestly state in paper that P2 uses heuristic judging and only baseline uses ensemble

**Recommended**: Run ensemble on P2 traces. This is ~5400 API calls (~$2-3). Then re-compute all tables.

---

### 1.5 Add concurrent no-interceptor control arm [MAJOR]
**Why**: Held-out interceptor comparison uses historical baseline (n=450, different tasks) as control. The 60.0%→32.8% claim rests on distributional similarity assumption.

**Action**:
- Re-run 600 tasks per model WITHOUT interceptor on the SAME held-out task split
- This gives a true concurrent control: same tasks, same time, only difference is interceptor on/off
- Re-compute hallucination reduction with proper paired comparison
- Cost estimate: ~$0.15 (same as original held-out run)

---

### 1.6 Investigate and fix interceptor label mismatch [MAJOR]
**Why**: Held-out F1 = 0.000 for both models. The interceptor abstains on non-hallucinating traces and misses all hallucinations.

**Root cause**: Sweep used `would_hallucinate = not baseline.final_correct` as ground truth. Held-out uses `hallucination = not final_correct` within that run. These are different signals when task pools differ.

**Action**:
- With the concurrent control arm (1.5), redefine ground truth consistently
- Re-tune thresholds if needed on a proper validation set
- Report sweep F1 as "validation estimate" and held-out F1 separately

---

## PHASE 2: PAPER REWRITING

### 2.1 Rewrite Abstract [Editor Point 1]
**Current problem**: Too much background, no specifics.

**New abstract structure** (~200 words):
1. One sentence: problem statement
2. One sentence: what AgentHallu-Bench is
3. Specific numbers: "9 models, 2000 tasks, 4 domains, 5400 P2 injection traces"
4. Key finding 1: injection-to-error rates (replace r₂₃=1.00 with per-model rates)
5. Key finding 2: κ=0.143 heuristic vs ensemble
6. Key finding 3: interceptor reduction (with corrected held-out numbers)
7. One sentence: data/code released

---

### 2.2 Add Research Objectives section [Editor Point 2]
Add after Introduction, before Related Work:

**Section 1.1: Research Objectives**
- RO1: Decompose agent hallucination into measurable stages and quantify propagation rates
- RO2: Evaluate whether heuristic evaluation is sufficient for tool-use benchmarks
- RO3: Benchmark runtime mitigation strategies (interceptor, self-consistency)

---

### 2.3 Add Dataset Description section [Editor Point 3]
Add new **Section 3.1: Dataset Construction**:
- 2000 tasks from `full_2000_tasks.json`: 667 calendar, 674 weather, 559 medical, 100 knowledge
- 300 retail tasks (held-out domain)
- Task structure: query, correct_tool_sequence, correct_final_answer, domain, difficulty
- Split: 209 validation, 1791 test
- Synthetic generation via 4 domain API simulators
- Table with domain × difficulty × split breakdown

---

### 2.4 Reframe the 4-stage cascade as 3-stage [Audit Issues 1, 9]
**Current (broken)**: S1→S2→S3→S4 with r₂₃=1.00

**New framing**:
- S1: Injection applied (eps ≥ 1) — parameter was modified
- S2: Injection executed (eps ≥ 2) — tool ran with bad params AND observable effect
- S3: Final answer wrong (final_correct = False)
- Replace "saturated propagation" with "substantial but highly variable propagation"
- Report human-calibrated r₂₃ ≈ 0.62 mean (range 0.46–0.73) as the primary number
- Report ensemble-judged r₂₃ = 0.78 (range 0.43–0.97) as the automated estimate
- Note: the 16pp gap between ensemble and calibrated reflects the ensemble's conservative bias
- Add limitation: calibration based on n=100 human labels → wide CIs

**New Table 2**: Per-model S1/S2/S3 rates, r₁₂, r₂₃ (ensemble + calibrated), n_S2, injection-to-error rate.

### 2.4b NEW SECTION — Rejection vs Recovery Decomposition [New finding]
**This gets its own subsection** (proposed Section 4.3 or within Section 5).

- Define rejection rate (1 − S1 rate) and recovery rate (1 − r₂₃)
- Present 9-model scatter plot (`fig_rejection_vs_recovery.png`)
- Report Spearman ρ = 0.126 (p = 0.747) — axes statistically independent
- Highlight key examples: gemini (best rejector, decent recovery), gpt41_nano (mediocre rejector, worst recovery), gpt4o_mini (poor at both)
- Frame the practical implication: mitigation strategies should target rejection and recovery independently
- This addresses the editor's request for theoretical/practical implications

---

### 2.5 Fix Table 1 and add conditioned metrics [Audit Issues 2, 3]
- Remove "2,000 tasks" from abstract — say "200 unique tasks evaluated across 9 models and 4 injection types (5,400 total traces)"
- Add column: "Error rate | eps≥1" — error rate conditioned on injection actually applied
- Clarify that P2 % is overall correctness including eps=0 traces (where injection didn't fire)
- Note that baseline and P2 use different task pools (450 vs 200 unique tasks)

---

### 2.6 Rewrite Interceptor section [Audit Issues 4, 7]
- State sweep F1=0.842 is validation-sweep estimate on n≈40 P2 rows
- Primary evidence is now the concurrent control: gpt4o_mini −23.0pp (55.8%→32.8%), gemini −1.3pp (not significant)
- Report the 25 "broken" traces analysis: all LLM run-to-run variance, no interceptor false positives, no domain clustering
- Add limitations paragraph: "Our interceptor delta is measured under heuristic judging applied identically to both arms. While the delta is robust to shared judging bias, validating the absolute hallucination rates in each arm would require additional human labeling which we leave to future work."
- Drop the Gemini claim from the abstract/highlights — only gpt4o_mini has a real effect

---

### 2.7 Add Discussion & Implications section [Editor Point 4]
**New Section 5: Discussion**

5.1 Theoretical implications:
- The injection-to-error rate shows parameter-stage filtering is critical
- Heuristic evaluation is unreliable (κ=0.143) — implications for IR evaluation
- Self-consistency is expensive and imprecise for tool-use agents

5.2 Practical implications:
- Practitioners should invest in schema validation (L1) as first defense
- Ensemble judging should replace heuristic matching in benchmarks
- Runtime monitors need proper held-out validation with concurrent controls

5.3 Comparison with existing work:
- How AgentHallu-Bench differs from TauBench, AgentBench, ToolBench, WebArena
- What our stage-level analysis reveals that end-to-end scores miss

---

### 2.8 Update Literature Review [Editor Points 6, 7]
- Add recent (2024-2026) papers from IP&M and IS/CS venues on:
  - LLM agent evaluation
  - Tool-use hallucination
  - LLM-as-judge methodology
  - Runtime monitoring / guardrails
- Search keywords: "LLM agent hallucination", "tool-use evaluation", "error propagation", "LLM judge"
- Target: 15-20 additional references from last 2 years
- Ensure APA formatting throughout

---

### 2.9 Add Limitations section [Multiple audit issues]
Explicit limitations to add:
1. `reached_output` measures final-answer wrongness, not observation-level corruption — a true observation judge would enable independent stage validation
2. Ensemble judge has no human-annotated ground truth (13 labels too small); internal consistency ≠ accuracy
3. P2 heuristic judging may differ from ensemble judging; we report both where available
4. Self-consistency comparison (n=30) is underpowered — directional only
5. Model selection: 9 API-accessible models; no open-weight models (Llama, Qwen) or Anthropic models — justified by API-level tool-call interception requirement but limits generalizability
6. Interceptor sweep optimized on small validation set; held-out performance should be treated as the primary estimate
7. Historical baseline used for held-out interceptor comparison (if not fixed by Phase 1)

---

### 2.10 Copy editing [Editor Point 5]
- Professional English editing pass
- Ensure accessibility for non-native speakers
- Consistent terminology throughout
- Fix all reference formatting to APA

---

### 2.11 Fix reference list [Editor Points 7, 8]
- Audit all references for missing fields (year, venue, pages)
- Convert to APA format
- Add IP&M-relevant references
- Remove any arXiv-only refs that have since been published

---

### 2.12 Correct all trace/task counts [Audit Issue A3]
Find and replace every instance:
- "2,000 tasks" → "2,000 task specifications (200 evaluated per model under P2)"
- "2,300 agent traces" → accurate count based on what's actually released
- Supplementary table: update to match actual CSV row counts

---

### 2.13 Add model selection justification [Audit Issue 10]
In Section 3 (Benchmark) add paragraph:
- 9 models selected for API-level tool-call interception capability
- All support function-calling or structured output
- Open-weight models (Llama, Qwen) excluded because injection requires intercepting the tool-call pipeline before execution, which is more complex with local inference
- Claude excluded to avoid potential conflict of interest in evaluation
- Note as limitation

---

### 2.14 Write Response to Editor document
Structured response addressing each of the 10 editor points with:
- What was changed
- Where in the manuscript (section, page, line)
- Brief justification

---

## PHASE 3: SUPPLEMENTARY & REPO CLEANUP

### 3.1 Update supplementary materials
- Fix "11 files, 1.8 MB total" → actual counts
- Fix "7 experiment runner scripts" → 16 exist locally, 7 on GitHub
- Add dataset statistics table with ACTUAL counts per domain/model/condition

### 3.2 Clean up GitHub repo
- Push all necessary source code
- Add proper README with reproduction instructions
- Add `requirements.txt` at repo root
- Verify end-to-end: clone → install → run smoke test

### 3.3 Remove/document unused columns
- `weps` column in p2_all.csv: either document or remove
- `injection_type` vs `error_type` naming inconsistency: standardize

---

## EXECUTION ORDER

```
Week 1: Code Fixes
  Day 1-2: Push metrics/ to GitHub, fix hop_analysis.py, fix reached_output semantics
  Day 3-4: Run ensemble judge on P2 traces, re-run hop_analysis
  Day 5:   Run concurrent no-interceptor control arm (600 tasks × 2 models)
           Re-compute interceptor metrics with consistent labels
           Re-run power_analysis.py

Week 2: Paper Rewriting  
  Day 1: Rewrite abstract, add research objectives, add dataset section
  Day 2: Reframe cascade model (3-stage), rewrite Table 2, fix Table 1
  Day 3: Rewrite interceptor section, add discussion & implications
  Day 4: Update literature review, add 15-20 new references
  Day 5: Add limitations, model selection justification, copy editing
  Day 6: Write Response to Editor, final proofread
  Day 7: Update supplementary, clean GitHub repo, final submission
```

---

## RISK ASSESSMENT

| Risk | Impact | Mitigation |
|---|---|---|
| Re-running ensemble judge changes Table 1 numbers significantly | HIGH | Compare heuristic vs ensemble on existing 960 baseline traces first to estimate drift |
| Concurrent control shows interceptor doesn't actually help | HIGH | Report honestly — negative results are publishable at IP&M |
| New r₂₃ values are incoherent (>1.0) | MEDIUM | Reframe as marginal injection-to-error rate, not conditional probability |
| Literature review finds papers that invalidate our contribution | LOW | Differentiate on stage-level decomposition, which no prior work does |
| Editor rejects despite revisions | MEDIUM | Thorough Response to Editor document; address every point explicitly |
