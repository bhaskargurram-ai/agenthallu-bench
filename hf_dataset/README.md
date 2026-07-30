---
license: mit
pretty_name: AgentProp-Bench
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-generation
tags:
- llm-agents
- tool-use
- function-calling
- agent-evaluation
- llm-as-a-judge
- hallucination
- benchmark
configs:
- config_name: tasks
  data_files: tasks.json
- config_name: traces
  data_files: traces.csv
- config_name: human_labels
  data_files: human_labels.csv
---

# AgentProp-Bench

A benchmark for **tool-using language agents** that measures what answer-only
scoring misses: **judge reliability**, **error propagation**, and **tool-call
fabrication** — plus a runtime interceptor that mitigates them.

- 📄 Paper: [arXiv:2604.16706](https://arxiv.org/abs/2604.16706) (under review at TMLR)
- 💻 Code + eval harness: https://github.com/bhaskargurram-ai/agenthallu-bench
- 🏆 Leaderboard: HF Space `AgentProp-Bench-Leaderboard`
- ⚡ One-line eval: `pip install agentprop-bench` → `agentprop eval --model X`

## What's inside

| Config | File | Rows | Description |
|---|---|---|---|
| `tasks` | `tasks.json` | 2,000 | Agent tasks across **weather, calendar, medical, knowledge** domains — query, required tools, correct tool sequence, gold answer, difficulty, split. (A supplementary e-commerce/retail set ships in the GitHub repo.) |
| `traces` | `traces.csv` | 2,150 | Full agent traces under parameter injection for **13 models** (9 proprietary + 4 open-weight), each scored by a validated 3-LLM ensemble judge **and** a naive substring judge, with EPS / propagation / cascade fields. |
| `human_labels` | `human_labels.csv` | 100 | Human correctness labels used to validate the judges (inter-annotator Cohen's κ = 0.835). |

## Headline findings

- **The judge, not the model, often decides the leaderboard.** Substring/keyword
  judging agrees with human labels at chance (κ = 0.049); a validated 3-LLM
  ensemble reaches κ = 0.432.
- **Error propagation is near-universal.** One injected wrong tool parameter
  reaches a wrong final answer ~62% of the time, roughly constant across all 13
  models — capability does not confer robustness to upstream error.
- **Fabrication is common and invisible to answer-only scoring.** Several models
  emit tool results they never obtained (Gemini-2.0-Flash: 37.5% of traces;
  classifier precision 92.5%).
- **A 3-layer runtime interceptor** cuts GPT-4o-mini hallucination by 23 points
  (55.8%→32.8%, p<0.001), with no effect on models that rarely emit an
  injectable call.

## Usage

```python
from datasets import load_dataset

tasks  = load_dataset("<user>/AgentProp-Bench", "tasks")
traces = load_dataset("<user>/AgentProp-Bench", "traces")
labels = load_dataset("<user>/AgentProp-Bench", "human_labels")
```

## Citation

```bibtex
@article{gurram2026agentprop,
  title   = {Evaluating Tool-Using Language Agents: Judge Reliability,
             Propagation Cascades, and Runtime Mitigation in AgentProp-Bench},
  author  = {Gurram, Bhaskar},
  journal = {arXiv preprint arXiv:2604.16706},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.16706}
}
```

License: MIT (code and data).
