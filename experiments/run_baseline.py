"""Day 6 baseline experiment — run tasks across models with real LLM calls.

No injection. Establishes baseline performance for comparison with P2/P3/P4.

Usage:
  python experiments/run_baseline.py \
    --models gpt4o_mini,gemini_20_flash,deepseek_v3 \
    --tasks data/tasks/full_500_tasks.json \
    --split test \
    --budget 5.0 \
    --output data/results/baseline.csv \
    --resume
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

from config import MODELS, BUDGET_LIMIT_USD, RANDOM_SEED, TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError

logger = logging.getLogger(__name__)
console = Console()


def run_single_baseline_task(
    task: dict,
    model_key: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    """Run one task with real LLM, no injection."""
    session_id = f"bl_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(
        session_id=session_id,
        task_id=task["task_id"],
        model=model_key,
        domain=domain,
    )

    # Build tool descriptions for domain
    tools = tool_executor.get_tools_for_domain(domain) if domain and domain != "knowledge" else tool_executor.list_tools()
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    system_msg = (
        "You are an agent that answers questions using tools.\n"
        "Follow this format exactly:\n"
        "Thought: [reasoning]\nAction: [tool_name]\nAction Input: {\"param\": \"value\"}\n\n"
        "When done, respond with:\nFinal Answer: [your answer]\n\n"
        f"Available tools:\n{tool_desc}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]

    total_cost = 0.0
    total_input = 0
    total_output = 0
    step_num = 0
    final_answer = ""
    tool_selected = ""
    tool_params_used = {}

    for iteration in range(5):
        try:
            result = router.complete(messages)
        except Exception as e:
            logger.error("LLM error %s/%s: %s", model_key, task["task_id"], e)
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])

        llm_text = result["text"]
        step_num += 1
        tracer.log_step(session_id, 0, step_num, "thought", llm_text[:500],
                        token_count=result["input_tokens"] + result["output_tokens"])

        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "final", final_answer)
            break

        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            tool_selected = tool_selected or tool_name
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except (json.JSONDecodeError, AttributeError):
                tool_params = {}
            if not tool_params_used:
                tool_params_used = tool_params

            step_num += 1
            tracer.log_step(session_id, 0, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            exec_result = tool_executor.execute(tool_name, tool_params)
            observation = json.dumps(exec_result["result"], default=str)
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "observation", observation[:1000],
                            tool_name=tool_name, tool_result=exec_result["result"],
                            param_errors=exec_result.get("validation_errors"))

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    # Check correctness
    gt = task.get("correct_final_answer", "")
    gt_tool = task.get("correct_tool_sequence", [{}])[0].get("tool", "") if task.get("correct_tool_sequence") else ""

    # Tool selection accuracy: did agent pick the right tool?
    tool_correct = (tool_selected == gt_tool) if gt_tool else True

    # Final answer: loose match
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "difficulty": task.get("difficulty", "medium"),
        "source": task.get("source", "synthetic"),
        "tool_selected": tool_selected,
        "tool_correct": tool_correct,
        "final_correct": final_correct,
        "cost": total_cost,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "steps": step_num,
    }


def main():
    parser = argparse.ArgumentParser(description="Day 6 Baseline Experiment")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "all"])
    parser.add_argument("--limit", type=int, default=0, help="Max tasks per model (0=all)")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="data/results/baseline.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    for mk in model_keys:
        if mk not in MODELS:
            console.print(f"[red]Unknown model: {mk}[/red]")
            sys.exit(1)

    # Load tasks
    tasks_path = os.path.join(os.path.dirname(__file__), "..", args.tasks)
    with open(tasks_path) as f:
        all_tasks = json.load(f)

    if args.split != "all":
        all_tasks = [t for t in all_tasks if t.get("split", "test") == args.split]

    if args.limit > 0:
        all_tasks = all_tasks[:args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    console.print(Panel.fit(
        f"[bold blue]Day 6 — Baseline Experiment[/bold blue]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Tasks: {len(all_tasks)} ({args.split} split)\n"
        f"Budget: ${args.budget:.2f}",
        border_style="blue",
    ))

    # Setup
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "baseline_day6.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    # Check for resume
    completed_pairs = set()
    if args.resume and os.path.exists(args.output):
        try:
            import pandas as pd
            existing = pd.read_csv(args.output)
            for _, row in existing.iterrows():
                completed_pairs.add((row["task_id"], row["model"]))
            console.print(f"[yellow]Resuming: {len(completed_pairs)} already done[/yellow]")
        except Exception:
            pass

    # Build work items
    work_items = []
    for mk in model_keys:
        for task in all_tasks:
            if (task["task_id"], mk) in completed_pairs:
                continue
            work_items.append((task, mk))

    console.print(f"Total runs: {len(work_items)}")

    results = []
    errors = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Running baseline...", total=len(work_items))

        def _run(item):
            task, mk = item
            router = ModelRouter(mk)
            return run_single_baseline_task(task, mk, router, tool_executor, tracer, budget)

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
                    item = futures[future]
                    errors.append(str(e))
                    logger.error("Failed: %s", e)

    # Save CSV
    if results:
        csv_path = os.path.join(os.path.dirname(__file__), "..", args.output)
        fieldnames = ["session_id", "task_id", "model", "domain", "difficulty", "source",
                       "tool_selected", "tool_correct", "final_correct", "cost",
                       "input_tokens", "output_tokens", "steps"]

        # If resuming, append
        mode = "a" if args.resume and os.path.exists(csv_path) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for r in results:
                writer.writerow(r)

    # Print summary
    console.print(f"\n[green]Completed: {len(results)} runs, Errors: {len(errors)}[/green]")

    table = Table(title="Baseline Results")
    table.add_column("Model")
    table.add_column("Tasks", justify="right")
    table.add_column("Tool Acc", justify="right")
    table.add_column("Final Acc", justify="right")
    table.add_column("Cost", justify="right")

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    total_cost = 0.0
    for mk in model_keys:
        mr = by_model.get(mk, [])
        if not mr:
            continue
        n = len(mr)
        tc = sum(1 for r in mr if r["tool_correct"])
        fc = sum(1 for r in mr if r["final_correct"])
        cost = sum(r["cost"] for r in mr)
        total_cost += cost
        table.add_row(mk, str(n), f"{100*tc/n:.0f}%", f"{100*fc/n:.0f}%", f"${cost:.4f}")

    console.print(table)
    console.print(f"\n[bold]Total cost: ${total_cost:.4f}[/bold]")
    console.print(f"Budget: {budget.status()}")
    console.print(f"Output: {args.output}")

    # Save budget log
    budget.save_log(os.path.join(RESULTS_DIR, "budget_log.json"))
    tracer.close()


if __name__ == "__main__":
    main()
