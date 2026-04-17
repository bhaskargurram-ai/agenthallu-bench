"""Concurrent no-interceptor control arm for the held-out interceptor experiment.

Runs the SAME task_ids from interceptor_heldout_600.csv WITHOUT any interceptor,
producing a true concurrent control for paired comparison.

Output: data/results/interceptor_control_600.csv
  — same schema as interceptor_heldout_600.csv but condition='no_interceptor'

Usage:
  python experiments/run_interceptor_control.py \
    --heldout data/results/interceptor_heldout_600.csv \
    --tasks data/tasks/full_2000_tasks.json \
    --budget 2.0 \
    --output data/results/interceptor_control_600.csv

After running, compare hallucination rates between control and heldout
on matched task_ids for a valid paired comparison.
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

logger = logging.getLogger(__name__)
console = Console()


def run_no_interceptor_task(task, model_key, router, tool_executor, tracer, budget):
    """Run one task with NO interceptor — vanilla agent execution."""
    session_id = f"ctrl_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type="no_interceptor", injection_stage=None)

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

    for iteration in range(5):
        try:
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
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            exec_result = tool_executor.execute(tool_name, tool_params)
            obs = json.dumps(exec_result["result"], default=str)
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
    hallucination = not final_correct

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "condition": "no_interceptor",
        "final_answer": final_answer[:200],
        "abstained": False,
        "abstain_reason": "",
        "flag_count": 0,
        "final_correct": final_correct,
        "hallucination": hallucination,
        "cost": total_cost,
        "steps": step_num,
    }


def main():
    parser = argparse.ArgumentParser(description="No-interceptor control arm for held-out comparison")
    parser.add_argument("--heldout", default="data/results/interceptor_heldout_600.csv",
                        help="Held-out CSV to extract task_ids and models from")
    parser.add_argument("--tasks", default="data/tasks/full_2000_tasks.json")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--output", default="data/results/interceptor_control_600.csv")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    base = os.path.join(os.path.dirname(__file__), "..")

    import pandas as pd

    # Load held-out to get task_ids and models
    heldout_path = os.path.join(base, args.heldout)
    heldout = pd.read_csv(heldout_path)
    models = sorted(heldout["model"].unique())
    task_ids_per_model = {}
    for m in models:
        task_ids_per_model[m] = sorted(heldout[heldout["model"] == m]["task_id"].unique())

    console.print(f"Models: {models}")
    for m in models:
        console.print(f"  {m}: {len(task_ids_per_model[m])} task_ids")

    # Load task definitions
    tasks_path = os.path.join(base, args.tasks)
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    task_map = {t["task_id"]: t for t in all_tasks}

    budget = BudgetGuard(limit_usd=args.budget)

    total_runs = sum(len(tids) for tids in task_ids_per_model.values())
    console.print(Panel.fit(
        f"[bold cyan]No-Interceptor Control Arm[/bold cyan]\n"
        f"Models: {', '.join(models)}\n"
        f"Total runs: {total_runs}\n"
        f"Budget: ${args.budget:.2f}",
        border_style="cyan",
    ))

    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(base, args.output)) or RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "interceptor_control.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    # Build work items
    work_items = []
    for m in models:
        for tid in task_ids_per_model[m]:
            task = task_map.get(tid)
            if task:
                work_items.append((task, m))
            else:
                logger.warning("Task %s not found in task file", tid)

    console.print(f"Work items: {len(work_items)}")
    results = []

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        pt = prog.add_task("Control arm runs", total=len(work_items))

        def _run(item):
            task, mk = item
            router = ModelRouter(mk)
            return run_no_interceptor_task(task, mk, router, tool_executor, tracer, budget)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run, item): item for item in work_items}
            for future in as_completed(futures):
                prog.update(pt, advance=1)
                try:
                    r = future.result()
                    results.append(r)
                except BudgetExceededError:
                    console.print("[red]Budget exceeded[/red]")
                    break
                except Exception as e:
                    logger.error("Failed: %s", e)

    # Save
    fields = ["session_id", "task_id", "model", "domain", "condition", "final_answer",
              "abstained", "abstain_reason", "flag_count", "final_correct", "hallucination",
              "cost", "steps"]
    out_path = os.path.join(base, args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    console.print(f"[green]Saved {len(results)} rows to {args.output}[/green]")

    # Summary
    df = pd.DataFrame(results)
    for m in models:
        sub = df[df["model"] == m]
        console.print(f"  {m}: halluc={sub['hallucination'].mean()*100:.1f}%, "
                      f"correct={sub['final_correct'].mean()*100:.1f}%, n={len(sub)}")

    console.print(f"\nTotal cost: ${sum(r['cost'] for r in results):.4f}")


if __name__ == "__main__":
    main()
