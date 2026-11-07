# AgentProp-Bench — targeted outreach

Goal: legitimate citations and adoption. Every contact is because our work is
*directly useful* to theirs — a benchmark they can run, a judge they can reuse,
a failure mode they measure differently. **No citation trading, no mass mail.**
Personalize the first line of every email to their specific paper/tool.

Send only after arXiv v2 + HF dataset are live (so the links resolve to the
13-model version and the one-line harness).

## Priority targets (who, why us, the specific hook)

1. **τ-bench / tau-bench maintainers (Sierra)** — tool-agent benchmark authors.
   Hook: we quantify judge unreliability (κ=0.049 substring vs 0.432 ensemble) on
   exactly the kind of tool-agent traces they score. Offer our ensemble judge +
   100 labels as a drop-in reliability check for their eval.

2. **AgentBench / T-Eval / ToolBench authors** — established tool-use benchmarks.
   Hook: propagation metric (one bad param → ~62% wrong answer, model-independent)
   is orthogonal to their accuracy metric; propose adding it as a robustness axis.

3. **BFCL (Berkeley Function-Calling Leaderboard) team** — the function-calling
   leaderboard. Hook: fabricated tool calls (Gemini 37.5%) are a failure their
   AST/exec checks may not flag; offer our fabrication classifier + trace set.

4. **LLM-as-a-judge reliability researchers** (e.g. authors of recent judge-bias /
   judge-agreement papers). Hook: we have a clean, released dataset where the
   naive judge sits at chance and a validated judge is 9× better — a concrete
   case study + labels for their meta-analysis.

5. **Agent-safety / guardrails groups (NeMo Guardrails, Guardrails AI, Invariant
   Labs, tool-sandbox authors)** — Hook: our runtime interceptor is a reproducible
   −23pp result they can benchmark their guardrail against; offer a comparison slot.

6. **HELM / lm-evaluation-harness maintainers (Stanford CRFM / EleutherAI)** —
   Hook: propose AgentProp-Bench as an agent-eval task; the pip harness + HF
   dataset make integration a thin adapter. This is adoption, not just a citation.

7. **Authors of 2025–26 agent-hallucination / tool-hallucination surveys** — Hook:
   we're a primary source with released traces + a taxonomy (propagation vs
   fabrication vs judge error). Easy, natural citation in a related-work section.

8. **Open-weight agent groups (Qwen, Nous/Hermes, InternLM teams)** — Hook: we
   benchmarked their models fairly under an identical prompted-ReAct protocol;
   share their rows and invite a leaderboard PR / a comment for their model card.

## Email template (personalize the bracketed line)

> Subject: A validated judge + propagation metric for [their benchmark/tool]
>
> Hi [name],
>
> I read [their specific paper/tool] — [one concrete, honest sentence showing you
> actually read it, e.g. "your τ-bench pass^k metric is exactly the kind of
> multi-step score our findings bear on"].
>
> We just released AgentProp-Bench (arXiv:2604.16706). One result seems directly
> relevant to you: substring/keyword judging of tool-agent traces agrees with human
> labels only at chance (κ=0.049), while a validated 3-LLM ensemble reaches κ=0.432
> (100 human labels, inter-annotator κ=0.835). We also measure error *propagation*
> — one wrong tool parameter reaches a wrong final answer ~62% of the time across
> all 13 models we tested.
>
> Everything is open and runnable in one line (`pip install agentprop-bench`;
> tasks, traces, labels, judge, and a runtime interceptor are on GitHub + HF). If
> it's useful, I'd be glad to help you plug the judge or the propagation metric
> into [their tool], or add [their model] to the leaderboard.
>
> Paper: https://arxiv.org/abs/2604.16706
> Code + data: https://github.com/bhaskargurram-ai/agenthallu-bench
>
> Best,
> Bhaskar Gurram, Zasti Inc.

## Cadence
- One personalized email. **One** follow-up after ~10 days if no reply. Then stop.
- Track replies/outcomes in a simple sheet; note anyone who says they'll cite/adopt.
- Prefer offering *work* (integration help, a leaderboard row, a comparison) over
  asking for a citation. The citation follows the usefulness.
