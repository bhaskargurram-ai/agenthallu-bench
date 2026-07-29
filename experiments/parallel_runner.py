"""Parallel experiment runner with budget guard.

Runs agent tasks across multiple models with asyncio + ThreadPoolExecutor.
Includes hard budget limit that stops all experiments if exceeded.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import MODELS, FULL_MODELS, DEV_MODELS, BUDGET_LIMIT_USD, RANDOM_SEED, TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from agent.memory_manager import MemoryManager
from tracer.trace_logger import TraceLogger

logger = logging.getLogger(__name__)
console = Console()


# ── Budget Guard ─────────────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    """Raised when cumulative spend exceeds the budget limit."""
    pass


class BudgetGuard:
    """Stops all experiments if cumulative spend exceeds limit.
    Checked after every API call.
    """

    def __init__(self, limit_usd: float = 75.0):
        self.limit = limit_usd
        self.spent = 0.0
        self._cost_log: list[dict] = []

    def charge(self, cost: float, model: str = "", task_id: str = ""):
        self.spent += cost
        self._cost_log.append({
            "cost": cost,
            "cumulative": self.spent,
            "model": model,
            "task_id": task_id,
            "timestamp": time.time(),
        })
        if self.spent > self.limit:
            raise BudgetExceededError(
                f"Budget limit ${self.limit:.2f} reached. "
                f"Total spent: ${self.spent:.4f}. "
                f"Use --budget flag to increase."
            )

    def remaining(self) -> float:
        return self.limit - self.spent

    def status(self) -> str:
        pct = 100 * self.spent / self.limit if self.limit > 0 else 0
        return f"${self.spent:.4f} / ${self.limit:.2f} ({pct:.1f}%)"

    def save_log(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "limit": self.limit,
                "total_spent": self.spent,
                "entries": self._cost_log,
            }, f, indent=2)


# ── Single Task Runner ───────────────────────────────────────────────────────

def run_single_task(
    task: dict,
    model_key: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
    experiment: str = "baseline",
) -> dict:
    """Run a single task through the agent with a real LLM call.

    Returns result dict with session_id, cost, correctness, etc.
    """
    session_id = f"{experiment}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(
        session_id=session_id,
        task_id=task["task_id"],
        model=model_key,
        domain=domain,
    )

    # Build system prompt with tool descriptions
    if domain:
        tools = tool_executor.get_tools_for_domain(domain)
    else:
        tools = tool_executor.list_tools()

    tool_desc = "\n".join(
        f"- {t['name']}: {t['description']}" for t in tools
    )

    system_msg = (
        "You are an agent that answers questions using tools.\n"
        "Follow this format exactly:\n"
        "Thought: [reasoning]\n"
        "Action: [tool_name]\n"
        "Action Input: {\"param\": \"value\"}\n\n"
        "When done, respond with:\n"
        "Final Answer: [your answer]\n\n"
        f"Available tools:\n{tool_desc}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    step_num = 0
    final_answer = ""

    for iteration in range(5):  # max 5 steps per task
        try:
            result = router.complete(messages)
        except Exception as e:
            logger.error("LLM call failed for %s/%s: %s", model_key, task["task_id"], e)
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        # Charge budget
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])

        llm_text = result["text"]

        step_num += 1
        tracer.log_step(
            session_id, 0, step_num, "thought",
            llm_text[:500],
            token_count=result["input_tokens"] + result["output_tokens"],
        )

        # Check for Final Answer
        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "final", final_answer)
            break

        # Parse and execute action
        import re
        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except (json.JSONDecodeError, AttributeError):
                tool_params = {}

            step_num += 1
            tracer.log_step(
                session_id, 0, step_num, "action",
                f"Action: {tool_name}",
                tool_name=tool_name,
                tool_params_raw=tool_params,
            )

            exec_result = tool_executor.execute(tool_name, tool_params)
            observation = json.dumps(exec_result["result"], default=str)

            step_num += 1
            tracer.log_step(
                session_id, 0, step_num, "observation",
                observation[:1000],
                tool_name=tool_name,
                tool_result=exec_result["result"],
            )

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No action parsed, ask for final answer
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached without final answer."

    # Check correctness
    ground_truth = task.get("correct_final_answer", "")
    correct = bool(ground_truth and ground_truth.lower()[:20] in final_answer.lower()[:100])

    tracer.end_session(session_id, final_answer, ground_truth, correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "final_answer": final_answer[:200],
        "correct": correct,
        "cost": total_cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "steps": step_num,
    }


# ── Main Runner ──────────────────────────────────────────────────────────────

def load_tasks(limit: int = 0) -> list[dict]:
    """Load tasks from JSON files."""
    tasks = []
    tasks_dir = os.path.join(os.path.dirname(__file__), "..", "data", "tasks")

    # Try full_500_tasks.json first
    full_path = os.path.join(tasks_dir, "full_500_tasks.json")
    if os.path.exists(full_path):
        with open(full_path) as f:
            tasks = json.load(f)
        logger.info("Loaded %d tasks from full_500_tasks.json", len(tasks))
    else:
        # Fall back to per-domain files
        for domain in ["weather", "calendar", "medical"]:
            path = os.path.join(tasks_dir, f"{domain}_tasks.json")
            if os.path.exists(path):
                with open(path) as f:
                    domain_tasks = json.load(f)
                tasks.extend(domain_tasks)

    if limit > 0:
        tasks = tasks[:limit]

    return tasks


def main():
    parser = argparse.ArgumentParser(description="AgentHallu-Bench Parallel Runner")
    parser.add_argument("--models", type=str, default="gemini_20_flash",
                        help="Comma-separated model keys (default: gemini_20_flash)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max tasks per model (default: 5)")
    parser.add_argument("--experiment", type=str, default="baseline",
                        help="Experiment name (default: baseline)")
    parser.add_argument("--budget", type=float, default=BUDGET_LIMIT_USD,
                        help=f"Budget limit in USD (default: {BUDGET_LIMIT_USD})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (task_id, model) pairs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    for mk in model_keys:
        if mk not in MODELS:
            console.print(f"[red]Unknown model: {mk}. Available: {list(MODELS.keys())}[/red]")
            sys.exit(1)

    budget = BudgetGuard(limit_usd=args.budget)

    console.print(Panel.fit(
        f"[bold blue]AgentHallu-Bench Parallel Runner[/bold blue]\n"
        f"Experiment: {args.experiment}\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Task limit: {args.limit}\n"
        f"Budget: ${args.budget:.2f}\n"
        f"Workers: {args.workers}",
        border_style="blue",
    ))

    # Setup
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, f"{args.experiment}_traces.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    # Load tasks
    tasks = load_tasks(limit=args.limit)
    if not tasks:
        console.print("[red]No tasks found! Run task_generator.py first.[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(tasks)} tasks")

    # Build work items: (task, model_key)
    work_items = []
    for mk in model_keys:
        for task in tasks:
            work_items.append((task, mk))

    console.print(f"Total runs: {len(work_items)} ({len(tasks)} tasks × {len(model_keys)} models)")

    # Run
    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task("Running experiments...", total=len(work_items))

        def _run_one(item):
            task, mk = item
            router = ModelRouter(mk)
            return run_single_task(
                task, mk, router, tool_executor, tracer, budget, args.experiment,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_one, item): item for item in work_items}
            for future in as_completed(futures):
                progress.advance(ptask)
                try:
                    result = future.result()
                    results.append(result)
                except BudgetExceededError as e:
                    console.print(f"\n[red bold]BUDGET EXCEEDED: {e}[/red bold]")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:
                    item = futures[future]
                    errors.append({"task_id": item[0]["task_id"], "model": item[1], "error": str(e)})
                    logger.error("Task failed: %s", e)

    # Summary
    console.print(f"\n[green]Completed: {len(results)} runs[/green]")
    if errors:
        console.print(f"[yellow]Errors: {len(errors)}[/yellow]")

    # Print results table
    table = Table(title=f"Results — {args.experiment}")
    table.add_column("Model")
    table.add_column("Tasks", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Avg Latency", justify="right")

    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    total_cost = 0.0
    for mk in model_keys:
        model_results = by_model.get(mk, [])
        if not model_results:
            continue
        n = len(model_results)
        correct = sum(1 for r in model_results if r["correct"])
        cost = sum(r["cost"] for r in model_results)
        total_cost += cost
        table.add_row(
            mk,
            str(n),
            f"{correct}/{n} ({100*correct/n:.0f}%)",
            f"${cost:.4f}",
            "—",
        )

    console.print(table)
    console.print(f"\n[bold]Total cost: ${total_cost:.4f}[/bold]")
    console.print(f"Budget status: {budget.status()}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, f"{args.experiment}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"Results saved to {results_path}")

    # Save budget log
    budget_log_path = os.path.join(RESULTS_DIR, "budget_log.json")
    budget.save_log(budget_log_path)

    tracer.close()


if __name__ == "__main__":
    main()
