# AgentProp-Bench

Benchmark and methodology for evaluating **tool-using language agents** under
adversarial parameter injection — measuring **judge reliability**, **error
propagation**, **tool-call fabrication**, and **runtime mitigation**.

Paper: *Evaluating Tool-Using Language Agents: Judge Reliability, Propagation
Cascades, and Runtime Mitigation in AgentProp-Bench* — on arXiv and under review
at TMLR.

## Key results (current release: 13 models)

- **Judge reliability.** Substring/keyword judging (the default in most agent
  benchmarks) agrees with human annotation only at **chance level (Cohen's
  κ = 0.049)**; a validated 3-LLM ensemble reaches **κ = 0.432**. Validated on
  100 human labels; inter-annotator **κ = 0.835**.
- **Error propagation.** A parameter-level injection reaches a wrong final
  answer with human-calibrated probability **≈ 0.62** (range 0.46–0.73),
  **consistently across all 13 models** — 9 proprietary + 4 open-weight
  (Qwen2.5-3B/7B/14B, Hermes-3-Llama-3.1-8B).
- **Fabricated tool use.** Models differ 5–100% in how often they emit a real
  tool call; several **fabricate** tool results they never obtained
  (Gemini-2.0-Flash: **37.5%** of traces). Classifier precision 92.5% (40-trace
  hand-check).
- **Runtime interceptor.** Cuts GPT-4o-mini hallucination by **−23.0 pp**
  (55.8%→32.8%, n=600 concurrent control, p<0.001); no effect on
  Gemini-2.0-Flash, which rarely emits an injectable tool call.

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
- `data/tasks/`, `data/results/` — 2,000 tasks; result CSVs + human labels
- `hf_dataset/` — Hugging Face dataset card + upload script
- `leaderboard/` — leaderboard.csv + Gradio HF Space
- `promo/` — announcement + outreach assets
- `paper_arxiv/` — de-anonymized arXiv v2 build; `paper_tmlr/` — TMLR submission (anonymized); `paper_ieee/` — de-anonymized IEEE build

## How to cite

```bibtex
@article{gurram2026agentprop,
  title        = {Evaluating Tool-Using Language Agents: Judge Reliability,
                  Propagation Cascades, and Runtime Mitigation in AgentProp-Bench},
  author       = {Gurram, Bhaskar},
  journal      = {arXiv preprint arXiv:2604.16706},
  year         = {2026},
  url          = {https://arxiv.org/abs/2604.16706}
}
```

## License

MIT (code and data).
