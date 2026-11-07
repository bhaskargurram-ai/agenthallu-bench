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
| `human_labels` | `human_labels.csv` | 100 | Human correctness labels from three annotators (Fleiss' κ = 0.877), used as a sanity check of the judges and the structured validator. |

## Headline findings (paper v3, structured validator)

- **The instrument changes conclusions.** A deterministic, gold-grounded
  validator (tools, executed parameters, state, answer grounding) replaces
  reference-based judging. On the same 306 traces, Gemini-2.0-Flash scores 63.4%
  under the substring heuristic and 5.6% under the validator.
- **One wrong argument causes errors.** In paired runs (same task and model, with
  and without injection) the error rate rises from 52.4% to 74.5% (+22.0 pp,
  95% CI 18.4–25.5; 1,085 pairs, 7 models), and by 44.7 pp where the corrupted
  call executed without errors. An LLM-judge ensemble sees +2.5 pp.
- **Fabricated tool use** occurs in 0–42.5% of traces depending on the model
  (rule-based lower bounds), largely because the harness accepts an action and a
  final answer in one completion.
- **The runtime monitor** lowers wrong answers mainly by abstaining; it is not
  better than random abstention under the validator.

Claims in arXiv v1–v2 that were withdrawn (≈0.62 propagation probability, the
interceptor working on every model) are listed in Section 8 of the v3 paper.

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
  title   = {Measuring What Tool-Using {LLM} Agents Actually Do: Structured
             Validation, Causal Error Propagation, and Fabricated Tool Use},
  author  = {Gurram, Bhaskar},
  journal = {arXiv preprint arXiv:2604.16706},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.16706}
}
```

License: MIT (code and data).
