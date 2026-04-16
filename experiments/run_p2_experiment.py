"""Day 6 P2 experiment — parameter injection across 4 error types.

For each task × model × error_type:
  1. Run agent normally (use tool from ground truth)
  2. Inject parameter error BEFORE tool execution
  3. Execute with corrupted params
  4. Compute EPS, cascade detection
  5. Save results

Usage:
  python experiments/run_p2_experiment.py \
    --models gpt4o_mini,gemini_20_flash,deepseek_v3 \
    --limit 200 \
    --all-error-types \
    --budget 6.0 \
    --output data/results/p2_results.csv \
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

from config import MODELS, PARAM_ERROR_TYPES, BUDGET_LIMIT_USD, RANDOM_SEED, TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from injector.parameter_injector import ParameterInjector
from tracer.trace_logger import TraceLogger
from metrics.eps_scorer import EPSScorer
from metrics.cascade_detector import CascadeDetector
from experiments.parallel_runner import BudgetGuard, BudgetExceededError

logger = logging.getLogger(__name__)
console = Console()


def run_p2_task(
    task: dict,
    model_key: str,
    error_type: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    injector: ParameterInjector,
    tracer: TraceLogger,
    budget: BudgetGuard,
    eps_scorer: EPSScorer,
    cascade_detector: CascadeDetector,
) -> dict:
    """Run one task with P2 injection."""
    session_id = f"p2_{error_type}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(
        session_id=session_id,
        task_id=task["task_id"],
        model=model_key,
        domain=domain,
        injection_type=f"p2_{error_type}",
        injection_stage="parameter_generation",
    )

    # Build tool descriptions
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
    step_num = 0
    final_answer = ""
    tool_selected = ""
    injection_done = False

    for iteration in range(5):
        try:
            result = router.complete(messages)
        except Exception as e:
            logger.error("LLM error: %s", e)
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
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

            step_num += 1
            tracer.log_step(session_id, 0, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            # INJECT: corrupt params before execution (only first tool call)
            if not injection_done:
                exec_result = tool_executor.execute(
                    tool_name, tool_params,
                    injector=injector,
                    injection_error_type=error_type,
                    tracer=tracer,
                    session_id=session_id,
                    turn_id=0,
                )
                injection_done = True
            else:
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
    tool_correct = (tool_selected == gt_tool) if gt_tool else True
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))

    tracer.end_session(session_id, final_answer, gt, final_correct)

    # Compute EPS
    trace = tracer.get_session_trace(session_id)
    eps_result = eps_scorer.compute_eps(trace, {}) if trace else {"eps": 0, "weps": 0.0, "reached_output": False}

    # Detect cascades
    cascade_result = cascade_detector.detect_all(trace) if trace else {"cascade_count": 0}

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "difficulty": task.get("difficulty", "medium"),
        "source": task.get("source", "synthetic"),
        "error_type": error_type,
        "tool_selected": tool_selected,
        "tool_correct": tool_correct,
        "final_correct": final_correct,
        "eps": eps_result.get("eps", 0),
        "weps": eps_result.get("weps", 0.0),
        "reached_output": eps_result.get("reached_output", False),
        "cascade_count": cascade_result.get("cascade_count", 0),
        "cost": total_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Day 6 P2 Injection Experiment")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--all-error-types", action="store_true")
    parser.add_argument("--error-types", type=str, default="")
    parser.add_argument("--budget", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="data/results/p2_results.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", type=str, default="full_500_tasks.json",
                        help="Tasks filename under data/tasks/")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip N test tasks before taking --limit (for held-out slices)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    for mk in model_keys:
        if mk not in MODELS:
            console.print(f"[red]Unknown model: {mk}[/red]")
            sys.exit(1)

    if args.all_error_types:
        error_types = PARAM_ERROR_TYPES
    elif args.error_types:
        error_types = [e.strip() for e in args.error_types.split(",")]
    else:
        error_types = PARAM_ERROR_TYPES

    # Load tasks (test split only, skip knowledge domain for P2)
    tasks_filename = getattr(args, "tasks", None) or "full_500_tasks.json"
    tasks_path = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", tasks_filename)
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    all_tasks = [t for t in all_tasks if t.get("split", "test") == "test" and t.get("domain") != "knowledge"]
    offset = getattr(args, "offset", 0) or 0
    if offset > 0:
        all_tasks = all_tasks[offset:]
    if args.limit > 0:
        all_tasks = all_tasks[:args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    # Estimate cost
    cost_per_task = 0.005  # rough estimate
    est_total = len(all_tasks) * len(model_keys) * len(error_types) * cost_per_task
    console.print(Panel.fit(
        f"[bold blue]Day 6 — P2 Injection Experiment[/bold blue]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Tasks: {len(all_tasks)}\n"
        f"Error types: {', '.join(error_types)}\n"
        f"Total runs: {len(all_tasks) * len(model_keys) * len(error_types)}\n"
        f"Est. cost: ${est_total:.2f}\n"
        f"Budget: ${args.budget:.2f}",
        border_style="blue",
    ))

    # Setup
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "p2_day6.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()
    injector = ParameterInjector(seed=RANDOM_SEED)
    eps_scorer = EPSScorer()
    cascade_detector = CascadeDetector()

    # Resume check
    completed_triples = set()
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    if args.resume and os.path.exists(output_path):
        try:
            import pandas as pd
            existing = pd.read_csv(output_path)
            for _, row in existing.iterrows():
                completed_triples.add((row["task_id"], row["model"], row["error_type"]))
            console.print(f"[yellow]Resuming: {len(completed_triples)} already done[/yellow]")
        except Exception:
            pass

    # Build work items
    work_items = []
    for mk in model_keys:
        for et in error_types:
            for task in all_tasks:
                if (task["task_id"], mk, et) in completed_triples:
                    continue
                work_items.append((task, mk, et))

    console.print(f"Work items: {len(work_items)}")

    results = []
    errors_count = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Running P2...", total=len(work_items))

        def _run(item):
            task, mk, et = item
            router = ModelRouter(mk)
            return run_p2_task(task, mk, et, router, tool_executor, injector,
                               tracer, budget, eps_scorer, cascade_detector)

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
                    errors_count += 1
                    logger.error("Failed: %s", e)

    # Save CSV
    if results:
        fieldnames = ["session_id", "task_id", "model", "domain", "difficulty", "source",
                       "error_type", "tool_selected", "tool_correct", "final_correct",
                       "eps", "weps", "reached_output", "cascade_count", "cost"]
        mode = "a" if args.resume and os.path.exists(output_path) else "w"
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for r in results:
                writer.writerow(r)

    # Summary tables
    console.print(f"\n[green]Completed: {len(results)}, Errors: {errors_count}[/green]")

    # EPS summary by model × error_type
    table = Table(title="P2 Results — Mean EPS by Model × Error Type")
    table.add_column("Model")
    for et in error_types:
        table.add_column(et, justify="right")
    table.add_column("Cost", justify="right")

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    total_cost = 0.0
    for mk in model_keys:
        mr = by_model.get(mk, [])
        if not mr:
            continue
        cost = sum(r["cost"] for r in mr)
        total_cost += cost
        row = [mk]
        for et in error_types:
            et_results = [r for r in mr if r["error_type"] == et]
            if et_results:
                mean_eps = sum(r["eps"] for r in et_results) / len(et_results)
                row.append(f"{mean_eps:.2f} (n={len(et_results)})")
            else:
                row.append("—")
        row.append(f"${cost:.4f}")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold]Total cost: ${total_cost:.4f}[/bold]")
    console.print(f"Budget: {budget.status()}")
    console.print(f"Output: {args.output}")

    budget.save_log(os.path.join(RESULTS_DIR, "budget_log.json"))
    tracer.close()


if __name__ == "__main__":
    main()
