# AgentHallu-Bench

> Numbers in this README correspond to the AgentProp-Bench journal
> submission (IPM-D-26-02511). Note that an earlier construction
> artifact (r=1.00 from coupled stage indicators) was corrected
> during revision; see paper §5.2 transparency note.

**Measuring and Mitigating Error Propagation in Tool-Using Language Agents**

Bhaskar Gurram · Zasti Inc. · [gurrambhaskar.ai@gmail.com](mailto:gurrambhaskar.ai@gmail.com)

---

## Overview

**AgentHallu-Bench** is a benchmark and methodology for studying how hallucinations propagate through multi-step tool-using LLM agents. Instead of reporting a single end-to-end correctness number, we decompose agent hallucination into a **4-stage error cascade** and measure the conditional probability of propagation at each hop.

### Key Findings

| Finding | Detail |
|---------|--------|
| **Substantial propagation** | r_{2,3} ranges 0.43-0.97 across 9 models (ensemble-judged), human-calibrated 0.46-0.73 |
| **Heuristic eval is broken** | Substring/keyword-match judging has only **κ = 0.14** agreement with a 3-LLM ensemble judge (κ = 0.77–0.82) |
| **Interceptor works** | A tuned 3-layer runtime interceptor achieves **F₁ = 0.842** and cuts GPT-4o-mini hallucination by **−23.0 pp** (concurrent n=600 control, p < 0.001). Gemini result is n.s. (−1.3 pp) |
| **Self-consistency is expensive** | k=3 voting produces 47–70% abstention at only modest hallucination reduction |

### Models Evaluated

GPT-4o, GPT-4.1-mini, GPT-4.1-nano, GPT-4o-mini, GPT-3.5-turbo, o3-mini, Gemini-2.0-Flash, Gemini-2.5-Flash, DeepSeek-V3

### Domains

Calendar (667 tasks) · Weather (674) · Medical (559) · Retail (300, held-out) · Knowledge (100, ablation)

---

## Repository Structure

```
agenthallu-bench/
├── paper/                    # LaTeX manuscript (elsarticle format)
│   ├── main.tex              # Main paper
│   ├── refs.bib              # References (peer-reviewed venues verified)
│   ├── tables/               # Auto-generated result tables
│   ├── figures/               # Figures
│   └── build_tables.py       # Regenerate tables from data/results/
├── interceptor/              # Three-layer runtime interceptor
│   └── interceptor.py        # AgentHalluInterceptor class
├── benchmark/                # Task generation and ground truth
│   ├── task_generator.py
│   ├── ground_truth.py
│   └── domains/              # Per-domain task templates
├── experiments/              # Experiment runners
│   ├── run_baseline.py
│   ├── run_p2_experiment.py
│   ├── run_interceptor.py
│   ├── run_interceptor_sweep.py
│   ├── run_ensemble_judge.py
│   ├── run_self_consistency.py
│   └── run_multi_seed.py
├── analysis/                 # Statistical analysis
│   ├── bootstrap.py          # Bootstrap CI computation
│   ├── hop_analysis.py       # Stage-reach and hop-conditional rates
│   ├── power_analysis.py     # Cohen's h and achieved power
│   ├── stats.py              # Summary statistics
│   └── figures.py            # Figure generation
├── data/results/             # All result CSVs and JSONs
│   ├── baseline_all.csv
│   ├── p2_all.csv
│   ├── hop_rates.json
│   ├── ensemble_judge_metrics.json
│   ├── power_analysis.json
│   ├── interceptor_heldout_600.csv
│   └── ...
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/bhaskargurram-ai/agenthallu-bench.git
cd agenthallu-bench
pip install -r requirements.txt
```

### Using the Interceptor

```python
from interceptor import AgentHalluInterceptor

interceptor = AgentHalluInterceptor(tau_L1=1, tau_L2=2, L3_enabled=True)

should_abstain, reason = interceptor.should_abstain(
    tool_response={"status": "error", "error": "Invalid patient ID"},
    chain_of_thought="The tool returned an error, but I'll try to answer anyway...",
    final_answer="The appointment was successfully scheduled."
)
# should_abstain=True, reason="L1_schema_error"
```

### Rebuilding Paper Tables

```bash
python paper/build_tables.py
```

### Running Experiments

```bash
# Baseline evaluation
python experiments/run_baseline.py

# P2 semantic-wrong injection
python experiments/run_p2_experiment.py

# Interceptor threshold sweep
python experiments/run_interceptor_sweep.py

# 3-LLM ensemble judge
python experiments/run_ensemble_judge.py
```

---

## The Error Propagation Score (EPS)

For a trace τ with one P2 injection, four binary stage indicators:

- **S₁**: Injector returned a modified parameter
- **S₂**: Tool executed the modified call
- **S₃**: Observation with injected effect entered reasoning context
- **S₄**: Final answer is incorrect

**EPS(τ) = S₁ + S₂ + S₃ + S₄ ∈ {0, 1, 2, 3, 4}**

Hop-conditional rate: **rₖ = P(Sₖ₊₁ | Sₖ)**

Our key finding: **r_{2,3} ranges 0.43–0.97** across 9 models (ensemble-judged), human-calibrated 0.46–0.73.

---

## Three-Layer Interceptor

| Layer | What it checks | Threshold |
|-------|---------------|-----------|
| **L1** (Schema) | Parameter validation errors, tool error responses | τ_L1 = 1 |
| **L2** (Keywords) | Uncertainty keywords in chain-of-thought | τ_L2 = 2 |
| **L3** (Consistency) | Success claims contradicting error observations | on/off |

**Best config**: τ_L1=1, τ_L2=2, L3=on → **F₁ = 0.842** (Prec 0.923, Recall 0.774)

**Held-out deployment** (n=600/model, disjoint from tuning):
- GPT-4o-mini: 55.8% -> 32.8% hallucination (-23.0 pp, concurrent n=600 control, p < 0.001)
- Gemini-2.0-Flash: 44.5% -> 43.2% (-1.3 pp, n.s. — null result; Gemini's 95% rejection rate at the schema gate eliminates the target failure mode)

---

## Citation

```bibtex
@article{gurram2026agenthallu,
  title   = {{AgentHallu-Bench}: Measuring and Mitigating Error Propagation in Tool-Using Language Agents},
  author  = {Gurram, Bhaskar},
  year    = {2026},
  url     = {https://github.com/bhaskargurram-ai/agenthallu-bench}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Bhaskar Gurram — [gurrambhaskar.ai@gmail.com](mailto:gurrambhaskar.ai@gmail.com) — Zasti Inc.
