# AgentProp-Bench — social / announcement assets

> **Post order (do this only AFTER arXiv v2 is live so links match the 13-model paper):**
> 1. Submit to **HF Daily Papers** (huggingface.co/papers) the morning the arXiv v2 goes public — this is the single highest-leverage action for reach.
> 2. X/Twitter thread (tag the models you benchmarked + a few agent-eval researchers).
> 3. LinkedIn (same content, professional framing).
> 4. Reddit r/MachineLearning as **[R]** (Tue–Thu, 9–11am ET gets the most eyes).
> 5. Blog / project page last, and link it from all of the above.
>
> Every asset links to: arXiv abstract (https://arxiv.org/abs/2604.16706), GitHub
> (https://github.com/bhaskargurram-ai/agenthallu-bench), and the HF dataset once live.

---

## X / Twitter thread

**1/**
Most agent benchmarks grade a run "correct" with substring matching. We checked that against 100 human labels.

It agrees with humans at **chance level (κ = 0.049).**

AgentProp-Bench: what actually breaks in tool-using LLM agents. 🧵

**2/**
We inject one wrong parameter into a tool call and follow it downstream across 13 models (9 proprietary + 4 open-weight).

A single bad parameter reaches a wrong final answer ~**62%** of the time — and this holds across *every* model we tested. Capability doesn't save you.

**3/**
The judge you use decides your leaderboard.

- Substring/keyword: κ = 0.049 (chance)
- Single strong LLM judge: better, still noisy
- Validated 3-LLM ensemble: κ = 0.432

Inter-annotator agreement among our humans: κ = 0.835. The gap is the problem.

**4/**
Some models don't just fail — they **fabricate**. They emit a tool *result* they never actually obtained.

Gemini-2.0-Flash did this in **37.5%** of traces. Classifier precision 92.5% (hand-checked). This is invisible to answer-only scoring.

**5/**
So we built a runtime interceptor — 3 lightweight layers that check tool calls before they execute.

GPT-4o-mini hallucination: **55.8% → 32.8%** (−23pp, n=600 concurrent control, p<0.001).

No effect on models that never emit an injectable call. Mitigation is model-specific.

**6/**
Everything is open: 2,000 tasks across 4 domains, all traces, the 100 human labels, the ensemble judge, and the interceptor.

📄 arXiv: https://arxiv.org/abs/2604.16706
💻 Code: https://github.com/bhaskargurram-ai/agenthallu-bench
🤗 Data + leaderboard: (HF link)

`pip install agentprop-bench` → `agentprop eval --model X`

**6/ (alt CTA)**
If you build or evaluate tool-using agents: please stop trusting substring judges, and report *propagation*, not just final-answer accuracy. Benchmark your model in one line and send a PR to the leaderboard.

---

## LinkedIn

**We benchmarked 13 LLM agents and found the way we grade them is broken.**

Most agent evaluations decide "correct vs. incorrect" with substring matching. We validated that against 100 human-labeled traces. It agrees with humans at chance level (Cohen's κ = 0.049). A validated 3-LLM ensemble judge reaches κ = 0.432; our human annotators agree with each other at κ = 0.835 — so the ceiling is real, the judge is the bottleneck.

Three more findings from AgentProp-Bench (9 proprietary + 4 open-weight models):

• **Error propagation is universal.** Inject one wrong tool parameter and it reaches a wrong final answer ~62% of the time — consistently across every model, regardless of capability.

• **Fabrication is common and invisible.** Several models emit tool results they never obtained (Gemini-2.0-Flash: 37.5% of traces). Answer-only scoring never sees it.

• **A runtime interceptor helps — but model-specifically.** It cut GPT-4o-mini hallucination by 23 points (55.8%→32.8%, p<0.001), with no effect on models that rarely emit an injectable call.

Everything is open — tasks, traces, human labels, judge, and interceptor. `pip install agentprop-bench` and score your own model in one line.

Paper: https://arxiv.org/abs/2604.16706 · Code: https://github.com/bhaskargurram-ai/agenthallu-bench

#LLM #AIagents #MachineLearning #Evaluation #Benchmark

---

## Reddit — r/MachineLearning

**Title:** [R] AgentProp-Bench: substring-matching judges agree with humans at chance level (κ=0.049); one bad tool parameter propagates to a wrong answer ~62% of the time across 13 models

**Body:**

We built a benchmark for tool-using LLM agents focused on three things answer-only scoring misses: judge reliability, error propagation, and tool-call fabrication.

Headline results (9 proprietary + 4 open-weight models: Qwen2.5-3B/7B/14B, Hermes-3-Llama-3.1-8B):

- **Judge reliability.** Substring/keyword judging — the default in a lot of agent benchmarks — agrees with human labels at κ=0.049 (chance). A validated 3-LLM ensemble reaches κ=0.432. Inter-annotator κ among our humans is 0.835, so this isn't a "humans disagree" artifact.
- **Propagation.** We inject a single wrong parameter into a tool call and follow it. It reaches a wrong final answer with human-calibrated probability ≈0.62 (range 0.46–0.73), and — notably — this is roughly constant across all 13 models. More capable ≠ more robust to upstream errors.
- **Fabrication.** Several models emit tool *results* they never obtained. Gemini-2.0-Flash does it in 37.5% of traces (classifier precision 92.5%, hand-checked on 40 traces).
- **Runtime interceptor.** 3 lightweight layers that validate calls before execution cut GPT-4o-mini hallucination 55.8%→32.8% (−23pp, n=600 concurrent control, p<0.001). No effect on Gemini-2.0-Flash, which rarely emits an injectable call.

Open: 2,000 tasks / 4 domains, all traces, the 100 human labels, the ensemble judge, and the interceptor. `pip install agentprop-bench` then `agentprop eval --model X` reproduces a leaderboard row.

Paper: https://arxiv.org/abs/2604.16706
Code: https://github.com/bhaskargurram-ai/agenthallu-bench

Happy to answer questions on the judge-validation protocol or the injection design.

---

## Blog / project-page draft (short)

**Title:** Your agent benchmark's judge is agreeing with you at chance level

When we grade a tool-using agent's run as "correct," we usually mean a substring appeared in the output. We measured how often that call matches a careful human reader: **κ = 0.049.** That's chance. The number your leaderboard reports is, in large part, an artifact of the judge.

AgentProp-Bench is our attempt to fix the measurement before arguing about the models. Three pieces:

1. **A validated judge.** We collected 100 human labels (inter-annotator κ = 0.835) and calibrated a 3-LLM ensemble against them (κ = 0.432 vs 0.049 for substring). Use it, or bring your own — but validate it.
2. **Propagation, not just accuracy.** We inject one wrong tool parameter and trace it. Across 13 models it reaches a wrong final answer ~62% of the time, nearly independent of model capability. Robustness to upstream error is not something scale gives you for free.
3. **A runtime mitigation you can actually run.** A 3-layer interceptor validates tool calls before they execute and cut GPT-4o-mini hallucination by 23 points. It does nothing for models that don't emit injectable calls — mitigation has to be matched to the failure mode.

Everything's open. `pip install agentprop-bench`, point it at your model, and send a leaderboard PR.

Paper · Code · Dataset+Leaderboard (links above).

---

## HF Daily Papers submission note

- Go to huggingface.co/papers, "Submit", paste the arXiv ID **2604.16706** (must be arXiv v2, the 13-model version).
- Add a 2–3 sentence summary = the LinkedIn opener.
- Link the HF **dataset** and **Space (leaderboard)** in the discussion so the paper page has the full trio.
- Best submitted early US morning on a weekday for upvote window.
