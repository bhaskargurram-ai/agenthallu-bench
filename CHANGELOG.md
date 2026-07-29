# Changelog

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
