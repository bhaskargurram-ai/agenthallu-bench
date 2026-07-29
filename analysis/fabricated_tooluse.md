# Finding: fabricated / absent tool use confounds "parameter rejection"

Discovered 2026-07-28 during the open-weight-model expansion, via a per-model
tool-call-rate audit of the P2 semantic_wrong traces (`analysis/tool_call_rates.py`,
`data/results/tool_call_rates.json`, `data/results/fabricated_tooluse.json`).

**Observation.** The injection can only apply when the agent actually emits a
parseable tool call. Models differ enormously in how often they do so:

| model | tool-call rate | fabricated tool use* |
|---|---|---|
| GPT-4o | 100% | 0% |
| GPT-4.1-nano | 99% | 0.5% |
| o3-mini | 88% | 8% |
| DeepSeek-V3 | 86% | 7% |
| GPT-4.1-mini | 85% | 12% |
| Gemini-2.5-Flash | 76% | 18% |
| GPT-3.5-turbo | 59.5% | 21.5% |
| GPT-4o-mini | 40% | 24% |
| Gemini-2.0-Flash | 5% | 42.5% |

\*fabricated = no tool call emitted, yet the final answer asserts a completed
tool-derived action/result ("I have retrieved the weather…", "I have scheduled…").

**Implication — corrects a prior-draft claim.** The earlier draft described
Gemini-2.0-Flash as the "best rejector (95%)" whose "aggressive parameter
rejection eliminates the target failure mode," and used this to explain the
interceptor null result. The data show the opposite mechanism: Gemini-2.0-Flash
almost never emits a real tool call (5%) and instead *fabricates* tool use in
42.5% of traces. Its low S1 is tool-call avoidance/fabrication, not parameter
rejection. The interceptor (schema validation on tool calls) therefore has
almost nothing to act on — the corrected, more precise explanation of the null
result. By contrast GPT-4o-mini emits real tool calls 40% of the time, so the
interceptor has genuine targets (−23pp).

**Scope of impact.**
- UNAFFECTED: judge reliability (kappa), and propagation r_{2,3} / inj->err,
  which are computed on S1=1 (tool actually called) traces only.
- REFRAMED: the "rejection" axis (1 - S1) must separate genuine parameter
  rejection (tool called, bad param handled) from absent/fabricated tool use.
  Report tool-call rate alongside rejection rate.
- STRENGTHENED: "fabricated tool use" is a distinct, measurable agent failure
  mode and a new contribution of the benchmark.
