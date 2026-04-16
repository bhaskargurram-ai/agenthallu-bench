"""Day 8 STEP 4: Multi-seed stability.

Runs baseline and P2 (semantic_wrong) conditions across 3 seeds × 50 tasks × 2 models
to quantify run-to-run variance. Provides CI for EPS and hallucination rate.

Output: data/results/multi_seed_results.csv with seed column.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
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


def run_seed_task(
    task: dict,
    model_key: str,
    condition: str,
    seed: int,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    """Run one task with given seed. condition: 'baseline' or 'p2_semantic'."""
    session_id = f"seed_{seed}_{condition}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    inject = condition == "p2_semantic"
    injector = ParameterInjector(seed=seed) if inject else None

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type=f"seed{seed}_{condition}",
                         injection_stage="parameter_generation" if inject else None)

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
    step_num = 0
    final_answer = ""
    injection_applied = False
    reached_obs_error = False
    reached_propagated = False
    tool_steps = 0

    for iteration in range(5):
        try:
            # Pass seed via temperature sampling (model-level randomness)
            result = router.complete(messages, seed=seed)
        except TypeError:
            # Router may not support seed param
            result = router.complete(messages)
        except Exception as e:
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        llm_text = result["text"]
        step_num += 1
        tracer.log_step(session_id, 0, step_num, "thought", llm_text[:500])

        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "final", final_answer)
            break

        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except Exception:
                tool_params = {}
            tool_steps += 1

            # Inject on first real tool call
            if inject and not injection_applied and tool_params:
                tool_schema = None
                try:
                    for t in tools:
                        if t.get("name") == tool_name:
                            tool_schema = t
                            break
                except Exception:
                    pass
                if tool_schema:
                    try:
                        bad = injector.inject(
                            tool_name=tool_name,
                            params=tool_params,
                            error_type="semantic_wrong",
                            schema=tool_schema,
                        )
                        if bad and bad != tool_params:
                            tool_params = bad
                            injection_applied = True
                    except Exception as e:
                        logger.debug("Injection skipped: %s", e)

            step_num += 1
            tracer.log_step(session_id, 0, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            exec_result = tool_executor.execute(tool_name, tool_params)
            obs = json.dumps(exec_result["result"], default=str)

            if injection_applied and exec_result.get("validation_errors"):
                reached_obs_error = True

            step_num += 1
            tracer.log_step(session_id, 0, step_num, "observation", obs[:1000],
                            tool_name=tool_name, tool_result=exec_result["result"],
                            param_errors=exec_result.get("validation_errors"))

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))
    if injection_applied and not final_correct:
        reached_propagated = True

    # Simplified EPS: 0 if no injection applied; otherwise count stages reached
    if not inject or not injection_applied:
        eps = 0.0
    else:
        eps = 1.0  # param stage
        if reached_obs_error:
            eps += 1.0  # observation stage
        if reached_propagated:
            eps += 1.0  # final stage

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "condition": condition,
        "seed": seed,
        "injection_applied": injection_applied,
        "final_correct": final_correct,
        "hallucination": (not final_correct),
        "eps": eps,
        "cost": total_cost,
        "steps": step_num,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--conditions", type=str, default="baseline,p2_semantic")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--output", type=str, default="data/results/multi_seed_results.csv")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    tasks_path = os.path.join(os.path.dirname(__file__), "..", args.tasks)
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    all_tasks = [t for t in all_tasks if t.get("split", "test") == "test"]
    if args.limit > 0:
        all_tasks = all_tasks[: args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    total = len(model_keys) * len(conditions) * len(seeds) * len(all_tasks)
    console.print(Panel.fit(
        f"[bold cyan]Day 8 STEP 4 — Multi-Seed Stability[/bold cyan]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Conditions: {', '.join(conditions)}\n"
        f"Seeds: {seeds}\n"
        f"Tasks: {len(all_tasks)}\n"
        f"Total runs: {total}\n"
        f"Budget: ${args.budget:.2f}",
        border_style="cyan",
    ))

    os.makedirs(TRACES_DIR, exist_ok=True)
    out_dir = os.path.dirname(args.output) or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "multiseed_day8.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    work_items = [(task, mk, cond, s) for mk in model_keys for cond in conditions for s in seeds for task in all_tasks]
    console.print(f"Work items: {len(work_items)}")

    results = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Multi-seed runs", total=len(work_items))

        def _run(item):
            task, mk, cond, s = item
            router = ModelRouter(mk)
            return run_seed_task(task, mk, cond, s, router, tool_executor, tracer, budget)

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

    fields = ["session_id", "task_id", "model", "domain", "condition", "seed",
              "injection_applied", "final_correct", "hallucination", "eps", "cost", "steps"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {args.output}[/green]")

    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby(["model", "condition", "seed"]).agg(
        n=("session_id", "count"),
        halluc_pct=("hallucination", "mean"),
        eps_mean=("eps", "mean"),
    ).round(3)
    console.print("\n[bold]Per-seed metrics:[/bold]")
    console.print(summary)

    # Cross-seed variance
    cross = df.groupby(["model", "condition"]).agg(
        halluc_mean=("hallucination", "mean"),
        halluc_std=("hallucination", lambda x: x.groupby(df.loc[x.index, "seed"]).mean().std()),
    ).round(4)
    console.print("\n[bold]Cross-seed variance:[/bold]")
    console.print(cross)

    spent = budget.spent if not callable(budget.spent) else budget.spent()
    console.print(f"\nTotal cost: [bold]${spent:.4f}[/bold]")


if __name__ == "__main__":
    main()
