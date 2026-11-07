# Re-analysis for the v3 paper

Code and outputs behind *Measuring What Tool-Using LLM Agents Actually Do:
Structured Validation, Causal Error Propagation, and Fabricated Tool Use*
(arXiv:2604.16706v3; source in `../paper_v3/`).

## Layout

- `V_validator/` — structured validator V1.0: specification (`VALIDATOR_SPEC.txt`),
  per-trace labels for all 33,583 traces (`labels_all.csv`), summary results,
  human agreement and adjudication of the 18 disagreements, robustness analysis,
  paired results (`paired_validator_results.txt`) and LaTeX tables.
- `scripts/` — all analysis code.
  - `V_core.py` (rubric), `V_traces.py` (loaders), `V_01`–`V_05` (re-score, human
    agreement, robustness, adjudication, build); `test_V_validator.py` (24 unit tests).
  - `X_paired_validator.py` — paired causal estimate (McNemar, task-cluster bootstrap).
  - `A_*` propagation statistics, `B_*` judge validity, `C_*` / `C2_*` runtime
    monitor (proprietary / open-weight), `D_*` fabricated tool use and real-tool
    pilot, `W_*` figures and tables.
- `A_propagation/`, `B_judge/`, `C_interceptor/`, `C2_openweight/`,
  `D_fabrication/`, `W_writeup/` — outputs of each analysis (JSON, CSV, LaTeX
  tables, judge caches, re-run traces).
- `pilot/` — real-tool pilot runner and its agent code.

## Reproducing the main results

All paths are relative to the repository; inputs are in `../data/`
(tasks, result CSVs, `data/traces/*.db`, human labels). Trace databases are
opened read-only.

```bash
cd reanalysis/scripts
python3 -I -m pytest -q test_V_validator.py              # 24 unit tests
python3 -I V_01_run_all.py                               # re-score 33,583 traces -> ../V_validator/labels_all.csv
python3 -I X_paired_validator.py ../V_validator/labels_all.csv   # paired effect: +22.0 pp [18.4, 25.5]
```

`V_01_run_all.py` takes about 35 seconds and reproduces the committed
`labels_all.csv` exactly. Scripts that call an LLM (judge re-runs, trace re-runs,
the pilot) read `OPENAI_API_KEY` from the environment; their cached outputs are
included, so the analyses run without API access.
