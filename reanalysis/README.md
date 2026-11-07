# Re-analysis for the v3 paper

Code and outputs behind *Measuring What Tool-Using LLM Agents Actually Do:
Structured Validation, Causal Error Propagation, and Fabricated Tool Use*
(arXiv:2604.16706v3).

## Layout

- `validator/` — structured validator V1.0: specification (`VALIDATOR_SPEC.txt`),
  per-trace labels for all 33,583 traces (`labels_all.csv`), summary results,
  human-agreement and adjudication files, robustness analysis, paired results
  (`paired_validator_results.txt`) and LaTeX tables.
- `scripts/` — all analysis code.
  - `V_core.py` (rubric), `V_traces.py` (loaders), `V_01`–`V_05` (re-score, human
    agreement, robustness, adjudication, build); `test_V_validator.py` holds 24
    unit tests (`python3 -I -m pytest -q scripts/test_V_validator.py`).
  - `X_paired_validator.py` — paired causal estimate (McNemar, task-cluster bootstrap).
  - `A_*` propagation statistics, `B_*` judge validity, `C_*`/`C2_*` runtime monitor
    (proprietary / open-weight), `D_*` fabricated tool use and real-tool pilot,
    `W_*` figures and tables.
- `results/` — outputs of each workstream (JSON, CSV, LaTeX tables, judge caches,
  re-run traces).
- `pilot/` — real-tool pilot runner.

## Running

Inputs are the files in `../data/` (tasks, result CSVs, `data/traces/*.db`).
The scripts were run with absolute paths from the author's machine; edit the
path constants at the top of `scripts/*_common.py` and the `V_traces.py` loader
to point at your checkout. Scripts that call an LLM read `OPENAI_API_KEY` from
the environment. Trace databases are opened read-only.
