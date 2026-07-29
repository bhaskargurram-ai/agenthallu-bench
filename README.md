# AgentProp-Bench

> Numbers in this README correspond to the AgentProp-Bench journal
> submission (IPM-D-26-02511). Note that an earlier construction
> artifact (r=1.00 from coupled stage indicators) was corrected
> during revision; see paper S5.2 transparency note.

Benchmark for evaluating tool-using language agents under adversarial
prompt injection, measuring judge reliability, error propagation
cascades, and runtime mitigation.

## Key Results

**Propagation cascade (Table 4):** r_{2,3} ranges 0.43-0.97 across
9 models (ensemble-judged), human-calibrated 0.46-0.73.

**Interceptor (held-out, n=600 concurrent control):**
- GPT-4o-mini: 55.8% -> 32.8% hallucination (-23.0 pp, concurrent
  n=600 control, p < 0.001)
- Gemini-2.0-Flash: 44.5% -> 43.2% (-1.3 pp, n.s. -- null result;
  Gemini's 95% rejection rate at the schema gate eliminates the
  target failure mode)

## Reproducing Tables

```bash
# Regenerate hop_rates.json (Table 4 cascade rates)
python analysis/hop_analysis.py

# Regenerate all LaTeX tables
python paper/build_tables.py
```

## Repository Structure

- `analysis/` -- Analysis scripts (hop rates, bootstrap, interceptor)
- `data/results/` -- CSV/JSON result files
- `data/traces/` -- Raw trace databases
- `paper/` -- LaTeX source (IEEE-style)
- `paper_elsarticle/` -- Elsevier elsarticle submission package
- `benchmark/` -- Task definitions
- `agent/` -- Agent harness
- `injector/` -- Prompt injection framework
- `interceptor/` -- Runtime mitigation layers
- `metrics/` -- Evaluation metrics
