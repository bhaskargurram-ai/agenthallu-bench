# Changelog

## 2026-11-07 — v0.3.0: re-analysis release (paper v3)

- Added `reanalysis/`: structured validator V1.0 (specification, 24 unit tests,
  per-trace labels for 33,583 traces), paired causal analysis, judge-validity,
  runtime-monitor, fabrication and real-tool pilot code with all outputs. Paths
  are repository-relative; re-scoring from this repository reproduces
  `reanalysis/V_validator/labels_all.csv` byte for byte.
- Added `data/traces/`: the ten per-step SQLite trace databases used by the
  validator (previously excluded by `.gitignore`).
- Added open-weight interceptor runs (Qwen2.5-7B, Qwen2.5-14B,
  Hermes-3-Llama-8B; full and schema-only) and the third annotator's labels
  (`data/results/human_labels_third_annotator.csv`).
- Added `paper_v3/` (arXiv v3 source and PDF).
- Withdrawn claims of v1–v2: the propagation probability of ≈0.62, the decoupling
  of rejection and recovery, the interceptor reducing hallucination on every
  tool-calling model, and two-annotator agreement figures. README, dataset card,
  leaderboard text and CITATION.cff updated accordingly; v2 planning notes and
  promotional drafts moved to `old files/v2_planning/`.

## 2026-04-17 — Rename to AgentProp-Bench

- Renamed benchmark from AgentHallu-Bench to AgentProp-Bench to reflect
  focus on propagation measurement and avoid confusion with the
  concurrently published "AgentHallu" paper (Liu et al., 2026,
  arXiv:2601.06818), which does step-level attribution on a different
  dataset with distinct methodology.
- Repository path remains `github.com/bhaskargurram-ai/agenthallu-bench`
  for URL stability. Data filenames and session IDs are unchanged
  (historical data).

## 2026-04-17 — Phase 1 Corrections

- Fixed `analysis/hop_analysis.py`: stage indicators S1/S2/S3 now computed
  independently. S2 uses only `eps` column; S3 uses only `final_correct`.
  Prior version conflated S2 and S3 via `reached_output` override, forcing
  r_23 = 1.00 by construction.
- Fixed `error_type` filter (was checking nonexistent `injection_type` column).
- Re-judged 1,350 P2 semantic-wrong traces with 3-LLM ensemble judge.
- Collected 100 stratified human labels for ensemble validation (kappa=0.432).
- Added concurrent no-interceptor control arm (n=600 per model).
- Pushed missing source code: `metrics/`, `agent/`, `injector/`, `tracer/`,
  `config.py`, `tests/`.
