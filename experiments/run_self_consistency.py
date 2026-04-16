"""M2: Self-consistency voting interceptor.

For each task, the agent is run k=3 times at temperature 0.7 with the same
initial prompt and (when condition=p2_semantic) the same parameter-injection seed.
The final answer is determined by majority vote of the 3 samples' final answers.
If the 3 answers disagree (no clear majority), the agent abstains.

This is a cheap, well-established interceptor from the LLM literature
(Wang et al 2023 "Self-Consistency Improves Chain of Thought").

Conditions: baseline | p2_semantic
Metrics:
  - hallucination_rate (correct=False, not abstained)
  - abstain_rate (no majority agreement)
  - final_correct (majority vote correct)
  - FP on clean: among baseline cases that would have been correct single-shot,
    how often does SC abstain (we approximate: abstain on baseline is FP).

Output: data/results/self_consistency_results.csv
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError
from injector.parameter_injector import ParameterInjector

logger = logging.getLogger(__name__)
console = Console()


def run_one_sample(
    task: dict,
    model_key: str,
    condition: str,
    sample_seed: int,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    budget: BudgetGuard,
) -> tuple[str, float, bool]:
    """Run agent once. Returns (final_answer, cost, injection_applied)."""
    domain = task["domain"]
    query = task["query"]
    inject = condition == "p2_semantic"
    injector = ParameterInjector(seed=sample_seed) if inject else None

    tools = tool_executor.get_tools_for_domain(domain) if domain and domain != "knowledge" else tool_executor.list_tools()
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    system_msg = (
        "You are an agent that answers questions using tools.\n"
        "Follow this format exactly:\n"
        "Thought: [reasoning]\nAction: [tool_name]\nAction Input: {\"param\": \"value\"}\n\n"
        "When done, respond with:\nFinal Answer: [your answer]\n\n"
        f"Available tools:\n{tool_desc}"
    )
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": query}]

    total_cost = 0.0
    final_answer = ""
    injection_applied = False

    for _ in range(5):
        try:
            # Temperature 0.7 for sampling diversity
            result = router.complete(messages, temperature=0.7)
        except TypeError:
            result = router.complete(messages)
        except Exception as e:
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        llm_text = result["text"]

        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            break

        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except Exception:
                tool_params = {}

            if inject and not injection_applied and tool_params:
                tool_schema = None
                for t in tools:
                    if t.get("name") == tool_name:
                        tool_schema = t
                        break
                if tool_schema:
                    try:
                        bad = injector.inject(tool_name=tool_name,
                                              params=tool_params,
                                              error_type="semantic_wrong",
                                              schema=tool_schema)
                        if bad and bad != tool_params:
                            tool_params = bad
                            injection_applied = True
                    except Exception as e:
                        logger.debug("Injection skipped: %s", e)

            exec_result = tool_executor.execute(tool_name, tool_params)
            obs = json.dumps(exec_result["result"], default=str)
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    return final_answer, total_cost, injection_applied


def normalize_answer(a: str) -> str:
    """Normalize for voting — lower, strip, take first 60 chars of alphanumeric."""
    a = (a or "").lower().strip()
    a = re.sub(r"[^a-z0-9 ]", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a[:60]


def majority_vote(answers: list[str]) -> tuple[str, bool]:
    """Return (winning_answer, has_clear_majority).
    Clear majority = strictly more than half of non-empty votes."""
    normed = [normalize_answer(a) for a in answers if a and not a.startswith("error")]
    if not normed:
        return "", False
    cnt = Counter(normed)
    top_norm, top_count = cnt.most_common(1)[0]
    has_majority = top_count > len(normed) / 2
    # Return the original-case version of the first matching answer
    for orig, n in zip(answers, [normalize_answer(a) for a in answers]):
        if n == top_norm:
            return orig, has_majority
    return answers[0], has_majority


def run_self_consistency_task(
    task: dict,
    model_key: str,
    condition: str,
    k: int,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    session_id = f"sc_{condition}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type=f"sc_{condition}",
                         injection_stage="parameter_generation" if condition == "p2_semantic" else None)

    answers = []
    costs = []
    any_injection = False
    for i in range(k):
        ans, cost, inj = run_one_sample(task, model_key, condition, sample_seed=1000 + i,
                                         router=router, tool_executor=tool_executor, budget=budget)
        answers.append(ans)
        costs.append(cost)
        any_injection = any_injection or inj
        tracer.log_step(session_id, 0, i + 1, "sample", ans[:400])

    chosen, has_majority = majority_vote(answers)
    abstained = not has_majority
    if abstained:
        final_answer = "I cannot reliably answer — samples disagreed."
    else:
        final_answer = chosen

    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))
    hallucination = (not final_correct) and (not abstained)

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "condition": condition,
        "k": k,
        "injection_applied": any_injection,
        "abstained": abstained,
        "final_correct": final_correct,
        "hallucination": hallucination,
        "cost": sum(costs),
        "unique_answers": len(set(normalize_answer(a) for a in answers)),
        "answer_sample": (answers[0] or "")[:150],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--conditions", type=str, default="baseline,p2_semantic")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="data/results/self_consistency_results.csv")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    tasks_path = os.path.join(os.path.dirname(__file__), "..", args.tasks)
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    all_tasks = [t for t in all_tasks if t.get("split", "test") == "test"]
    if args.limit > 0:
        all_tasks = all_tasks[: args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    console.print(Panel.fit(
        f"[bold cyan]M2 — Self-Consistency Voting[/bold cyan]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Conditions: {', '.join(conditions)}\n"
        f"k (samples per task): {args.k}\n"
        f"Tasks: {len(all_tasks)}\n"
        f"Total calls ≈ {len(model_keys) * len(conditions) * len(all_tasks) * args.k * 3} (×~3 steps)\n"
        f"Budget: ${args.budget:.2f}",
        border_style="cyan",
    ))

    os.makedirs(TRACES_DIR, exist_ok=True)
    out_dir = os.path.dirname(args.output) or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "self_consistency_day8.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    work_items = [(task, mk, cond) for mk in model_keys for cond in conditions for task in all_tasks]
    console.print(f"Work items: {len(work_items)}")

    results = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Self-consistency runs", total=len(work_items))

        def _run(item):
            task, mk, cond = item
            router = ModelRouter(mk)
            return run_self_consistency_task(task, mk, cond, args.k, router,
                                             tool_executor, tracer, budget)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run, item): item for item in work_items}
            for future in as_completed(futures):
                progress.advance(ptask)
                try:
                    r = future.result()
                    results.append(r)
                except BudgetExceededError as e:
                    console.print(f"\n[red bold]BUDGET EXCEEDED: {e}[/red bold]")
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:
                    logger.error("Failed: %s", e)

    fields = ["session_id", "task_id", "model", "domain", "condition", "k",
              "injection_applied", "abstained", "final_correct", "hallucination",
              "cost", "unique_answers", "answer_sample"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {args.output}[/green]")

    import pandas as pd
    df = pd.DataFrame(results)
    if len(df):
        summary = df.groupby(["model", "condition"]).agg(
            n=("session_id", "count"),
            halluc=("hallucination", "mean"),
            abstain=("abstained", "mean"),
            correct=("final_correct", "mean"),
            avg_unique=("unique_answers", "mean"),
        ).round(3)
        console.print("\n[bold]Self-consistency summary:[/bold]")
        console.print(summary)

    spent = budget.spent if not callable(budget.spent) else budget.spent()
    console.print(f"\nTotal cost: [bold]${spent:.4f}[/bold]")


if __name__ == "__main__":
    main()
