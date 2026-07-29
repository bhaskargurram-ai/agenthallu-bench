# Formal Definition of the Error Propagation Score (EPS)

_Methods-section draft for AgentHallu-Bench paper. Day 8 M3._

## 1. Agent trace formalism

We model a tool-using LLM agent execution as a finite sequence of typed events
$$\tau = (e_1, e_2, \dots, e_T),\quad e_t \in \mathcal{E} = \{\textsf{Thought}, \textsf{Action}, \textsf{Observation}, \textsf{Final}\}.$$

Each **Action** event carries a tool name $\alpha_t$ and a parameter dictionary $p_t$, each **Observation** carries the tool return value $o_t$ and an optional validator report $v_t$ (list of schema errors), and the single terminal **Final** event carries the agent's answer $\hat{y}$. The task has a ground-truth answer $y^\star$ and an *oracle correctness* predicate $C(\hat{y}, y^\star) \in \{0,1\}$.

A *parameter injector* is a stochastic map $\mathcal{I}: (\alpha, p, \kappa) \mapsto p'$ where $\kappa$ is an error type (e.g. `semantic_wrong`). In the injected condition, exactly one Action event $e_{t^\star}$ has $p_{t^\star} = \mathcal{I}(\alpha_{t^\star}, p^{\text{clean}}_{t^\star}, \kappa)$ with $p_{t^\star} \neq p^{\text{clean}}_{t^\star}$. Let $\mathbb{I}_\text{applied} \in \{0,1\}$ indicate whether injection actually perturbed the parameter.

## 2. Stage indicators

We define four binary stage-reach indicators on an injected trace $\tau$:

1. **$S_1$ — Parameter-stage reach.** $S_1(\tau) = \mathbb{I}_\text{applied}$. The injected parameter entered a tool call.

2. **$S_2$ — Observation-stage reach.** The downstream tool accepted the bad parameter without the environment's validator rejecting it *and* without the agent self-correcting on the next turn:
   $$S_2(\tau) = S_1(\tau) \wedge \big(v_{t^\star} = \emptyset\big) \wedge \neg\text{SelfCorrect}(\tau, t^\star),$$
   where $\text{SelfCorrect}(\tau, t^\star)$ is true iff any later Action $e_{t'}$, $t' > t^\star$, repeats tool $\alpha_{t^\star}$ with parameters equal to (or closer to) $p^{\text{clean}}_{t^\star}$.

3. **$S_3$ — Reasoning-stage reach.** The bad observation influenced the agent's subsequent thought:
   $$S_3(\tau) = S_2(\tau) \wedge \big(\exists\, e_{t''} \in \textsf{Thought},\; t'' > t^\star,\; \text{cites}(e_{t''}, o_{t^\star})\big),$$
   where $\text{cites}(\cdot)$ is a textual-similarity predicate (substring or embedding-cosine $>\theta$) between the thought and the bad observation.

4. **$S_4$ — Final-answer reach.** The final answer is wrong:
   $$S_4(\tau) = S_3(\tau) \wedge \neg C(\hat{y}, y^\star).$$

By construction $S_1 \geq S_2 \geq S_3 \geq S_4$.

## 3. EPS

The **Error Propagation Score** of a trace is the number of stages the injection reaches:
$$\text{EPS}(\tau) = \sum_{k=1}^{4} S_k(\tau) \in \{0, 1, 2, 3, 4\}.$$

In this paper we report the 0–3 variant with $S_3$ merged into the final-answer condition (i.e. EPS$\in\{0,1,2,3\}$); the 0–4 variant separates reasoning-stage reach from final-answer reach.

### Properties

- **Monotonicity.** For two injectors $\mathcal{I}_1, \mathcal{I}_2$ where $\mathcal{I}_2$ *strictly dominates* $\mathcal{I}_1$ (every trace reached by $\mathcal{I}_1$ is also reached by $\mathcal{I}_2$ at the same or later stage), $\mathbb{E}[\text{EPS}(\tau \mid \mathcal{I}_2)] \geq \mathbb{E}[\text{EPS}(\tau \mid \mathcal{I}_1)]$.

- **Absorbing failure.** If $S_k = 0$ for some $k$, then $S_{k'} = 0$ for all $k' > k$. Intervention at any stage stops propagation.

- **Scale-freeness.** EPS is bounded above by 3 (or 4) regardless of trace length, so averaging across traces with heterogeneous step counts is well-defined.

- **Decomposition.** Mean EPS equals a sum of stage-specific reach probabilities:
  $$\bar{\text{EPS}} = \sum_{k=1}^{4} \Pr[S_k = 1],$$
  so the *conditional-hop rate* $r_k = \Pr[S_{k+1} = 1 \mid S_k = 1]$ is directly estimable from paired bootstrap samples.

## 4. Relationship to hallucination rate

Let $H(\tau) = 1 - C(\hat{y}, y^\star)$ be the hallucination indicator. On the injected condition,
$$\Pr[H(\tau) = 1] \geq \Pr[S_4(\tau) = 1] \geq \Pr[\text{EPS}(\tau) = 3].$$

So a non-zero $\Pr[\text{EPS}=3]$ is a lower bound on injection-attributable hallucination, while the gap $\Pr[H] - \Pr[S_4]$ estimates the baseline (non-injection-caused) hallucination rate.

## 5. Estimators

For a dataset of $N$ injected traces we report:
- $\hat{\mu}_{\text{EPS}} = \frac{1}{N}\sum_i \text{EPS}(\tau_i)$, with 95% bootstrap-$t$ CI (1,000 resamples).
- Stage-specific rates $\hat{p}_k = \frac{1}{N}\sum_i S_k(\tau_i)$, likewise bootstrapped.
- Hop-conditional rates $\hat{r}_k = \hat{p}_{k+1}/\hat{p}_k$ when $\hat{p}_k > 0$.
- Paired model contrasts use the same task-id injection seeds, so each task contributes matched traces across models.

## 6. Labeling protocol

$S_1, S_2$ are derived automatically from trace metadata (parameter diff, validator log). $S_3$ is labeled by a StageAttributor that uses substring matching and, on disagreement or ambiguity, an LLM judge (see Appendix B for inter-annotator agreement: Cohen's $\kappa$ against 3-LLM ensemble ground truth). $S_4$ uses the same ensemble judge protocol with majority vote as final label.
