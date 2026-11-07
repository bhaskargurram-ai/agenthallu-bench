# AgentProp-Bench

Benchmark and methodology for evaluating **tool-using language agents** under
adversarial parameter injection — measuring **judge reliability**, **error
propagation**, **tool-call fabrication**, and **runtime mitigation**.

Paper (current version, arXiv v3): *Measuring What Tool-Using LLM Agents Actually
Do: Structured Validation, Causal Error Propagation, and Fabricated Tool Use*
([arXiv:2604.16706](https://arxiv.org/abs/2604.16706)).

## Key results (re-analysis, structured validator)

Correctness is measured by a deterministic, gold-grounded validator (tools called,
executed parameters, resulting state, and grounding of every answer value in the
replayed tool observation). No LLM or human label is part of the instrument.
Code and outputs: [`reanalysis/`](reanalysis/).

- **Instrument validity.** 33,583 traces from 13 agents re-scored; the simulator
  replays 10,396/10,396 logged observations exactly. On the same 306 traces,
  Gemini-2.0-Flash scores 63.4% under the substring heuristic and 5.6% under the
  validator.
- **Causal effect of one wrong argument.** Paired runs (same task and model, with
  and without injection): error rate 52.4% → 74.5%, **+22.0 pp** (95% CI 18.4–25.5;
  1,085 pairs, 7 models); **+44.7 pp** (38.4–51.0) where the corrupted call
  executed without errors. An LLM-judge ensemble sees +2.5 pp on the same pairs.
- **Fabricated tool use.** 0–42.5% of traces depending on the model (rule-based
  lower bounds), largely caused by the harness accepting an action and a final
  answer in one completion.
- **Runtime monitor.** About 7 µs per trace, but it lowers wrong answers mainly by
  abstaining; it is not better than random abstention under the validator.

> **Withdrawn from v1–v2:** the propagation probability of ≈0.62, the decoupling of
> rejection and recovery, and the claim that the interceptor reduces hallucination
> on every tool-calling model. See Section 8 of the v3 paper. Inter-annotator
> agreement is now reported for three annotators (Fleiss κ = 0.877).

## Quick start

```bash
pip install -e .                       # or: pip install agentprop-bench (once published)
agentprop eval --model gpt-4o-mini     # score any API/HF model, get a leaderboard row
```

## Reproduce

```bash
python analysis/hop_analysis.py                 # propagation / hop rates
bash   analysis/finalize_openmodels.sh          # merge open models -> tables/figures -> recompile
python analysis/build_leaderboard.py            # rebuild leaderboard/leaderboard.csv
python analysis/citation_tracker.py track       # log the citation count over time
```

## Repository structure

- `agentprop/` — pip-installable eval harness (`agentprop eval` CLI)
- `agent/` — ReAct agent, model router (API + local vLLM), tool executor
- `injector/` — parameter / memory / propagation injection
- `interceptor/` — 3-layer runtime interceptor
- `metrics/` — EPS scorer, cascade detector, stage attribution
- `benchmark/` — task generator + 4 domain simulators
- `experiments/` — baseline / P2 / interceptor / ensemble-judge runners
- `analysis/` — statistics, merge, tool-call-rate, leaderboard + citation tracker
- `data/tasks/`, `data/results/` — 2,000 tasks; result CSVs + human labels (incl. open-weight interceptor runs)
- `data/traces/` — per-step SQLite trace databases (baseline, P2 injection, interceptor, interceptor control)
- `reanalysis/` — structured validator, unit tests, all re-analysis scripts and outputs for the v3 paper
- `hf_dataset/` — Hugging Face dataset card + upload script
- `leaderboard/` — leaderboard.csv + Gradio HF Space
- `promo/` — announcement + outreach assets
- `paper_arxiv/` — de-anonymized arXiv v2 build; `paper_tmlr/` — TMLR submission (anonymized); `paper_ieee/` — de-anonymized IEEE build

## How to cite

```bibtex
@article{gurram2026agentprop,
  title        = {Measuring What Tool-Using {LLM} Agents Actually Do: Structured
                  Validation, Causal Error Propagation, and Fabricated Tool Use},
  author       = {Gurram, Bhaskar},
  journal      = {arXiv preprint arXiv:2604.16706},
  year         = {2026},
  url          = {https://arxiv.org/abs/2604.16706}
}
```

## License

MIT (code and data).
