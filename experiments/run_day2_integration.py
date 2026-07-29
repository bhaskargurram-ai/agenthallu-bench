"""Day 2 Integration Test — run 10 tasks end-to-end with full tracing.

Runs without LLM calls by simulating agent steps directly.
Verifies SQLite DB has rows in sessions, steps, ground_truth tables.
Saves run to ./data/traces/day2_test.db
"""

import json
import logging
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from config import RANDOM_SEED, TASKS_DIR, TRACES_DIR
from agent.tool_executor import ToolExecutor
from agent.memory_manager import MemoryManager
from tracer.trace_logger import TraceLogger
from tracer.trace_schema import SessionRecord, StepRecord, GroundTruthRecord
from benchmark.ground_truth import record_ground_truth

logger = logging.getLogger(__name__)
console = Console()


def load_tasks_sample(n: int = 10) -> list[dict]:
    """Load n tasks from saved JSON — balanced across domains."""
    tasks = []
    for domain in ["weather", "calendar", "medical"]:
        path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
        if not os.path.exists(path):
            console.print(f"[red]Task file not found: {path}[/red]")
            console.print("[yellow]Run task_generator.py first.[/yellow]")
            return []
        with open(path) as f:
            domain_tasks = json.load(f)
        # Take a balanced sample: pick from easy, medium, hard
        easy = [t for t in domain_tasks if t["difficulty"] == "easy"][:2]
        medium = [t for t in domain_tasks if t["difficulty"] == "medium"][:1]
        hard = [t for t in domain_tasks if t["difficulty"] == "hard"][:1]
        tasks.extend(easy + medium + hard)

    return tasks[:n]


def run_task_with_tracing(task: dict, tracer: TraceLogger, executor: ToolExecutor) -> dict:
    """Run a single task end-to-end with tracing (simulated, no LLM).

    Executes the ground truth tool sequence directly and traces everything.
    Returns result summary dict.
    """
    session_id = f"day2_{task['task_id']}_{uuid.uuid4().hex[:8]}"
    domain = task["domain"]
    model = "gpt-4o"

    try:
        # Start session
        tracer.start_session(session_id, task["task_id"], model, domain)

        step_num = 0
        tools_called = []

        # Trace the initial thought
        step_num += 1
        tracer.log_step(
            session_id, 0, step_num, "thought",
            f"I need to answer: {task['query'][:200]}",
        )

        # Execute each tool in the correct sequence
        for seq_item in task["correct_tool_sequence"]:
            tool_name = seq_item["tool"]
            params = seq_item["params"]

            # Trace action
            step_num += 1
            tracer.log_step(
                session_id, 0, step_num, "action",
                f"Action: {tool_name}",
                tool_name=tool_name,
                tool_params_raw=params,
            )

            # Execute tool
            exec_result = executor.execute(tool_name, params)
            tools_called.append(tool_name)

            # Trace observation
            step_num += 1
            tracer.log_step(
                session_id, 0, step_num, "observation",
                json.dumps(exec_result["result"], default=str)[:500],
                tool_name=tool_name,
                tool_params_validated=exec_result.get("validated_params"),
                tool_result=exec_result["result"],
                param_errors=exec_result.get("validation_errors"),
            )

        # Trace final answer
        step_num += 1
        final_answer = task["correct_final_answer"]
        tracer.log_step(
            session_id, 0, step_num, "final",
            final_answer,
        )

        # End session
        tracer.end_session(session_id, final_answer, final_answer, True)

        return {
            "task_id": task["task_id"],
            "domain": domain,
            "difficulty": task["difficulty"],
            "steps_taken": step_num,
            "tools_called": tools_called,
            "trace_rows": tracer.get_step_count(session_id),
            "session_id": session_id,
            "success": True,
            "error": None,
        }

    except Exception as e:
        logger.error("Task %s failed: %s", task["task_id"], e)
        try:
            tracer.end_session(session_id, f"ERROR: {e}", task["correct_final_answer"], False)
        except Exception:
            pass
        return {
            "task_id": task["task_id"],
            "domain": domain,
            "difficulty": task.get("difficulty", "unknown"),
            "steps_taken": step_num if 'step_num' in dir() else 0,
            "tools_called": tools_called if 'tools_called' in dir() else [],
            "trace_rows": 0,
            "session_id": session_id,
            "success": False,
            "error": str(e),
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    console.print("[bold blue]Day 2 Integration Test[/bold blue]\n")
    console.print(f"Random seed: {RANDOM_SEED}\n")

    # Setup
    os.makedirs(TRACES_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "day2_test.db")

    # Remove old DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    tracer = TraceLogger(db_path)
    tracer.init_db()
    executor = ToolExecutor()

    # Load tasks
    tasks = load_tasks_sample(10)
    if not tasks:
        return

    console.print(f"Loaded {len(tasks)} tasks for integration test\n")

    # Record ground truth first
    all_tasks_for_gt = {}
    for domain in ["weather", "calendar", "medical"]:
        path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
        with open(path) as f:
            all_tasks_for_gt[domain] = json.load(f)
    gt_count = record_ground_truth(tracer, all_tasks_for_gt)
    console.print(f"[green]Recorded {gt_count} ground truth entries[/green]\n")

    # Run tasks
    results = []
    for task in tasks:
        console.print(f"Running task [cyan]{task['task_id']}[/cyan] ({task['domain']}/{task['difficulty']})...")
        result = run_task_with_tracing(task, tracer, executor)
        results.append(result)

    # Print results table
    table = Table(title="Day 2 Integration Test Results")
    table.add_column("Task ID", style="cyan")
    table.add_column("Domain")
    table.add_column("Difficulty")
    table.add_column("Steps", justify="right")
    table.add_column("Tools Called")
    table.add_column("Trace Rows", justify="right")
    table.add_column("Status", style="bold")

    success_count = 0
    for r in results:
        status = "[green]✓[/green]" if r["success"] else f"[red]✗ {r['error'][:30]}[/red]"
        if r["success"]:
            success_count += 1
        table.add_row(
            r["task_id"],
            r["domain"],
            r["difficulty"],
            str(r["steps_taken"]),
            ", ".join(r["tools_called"]),
            str(r["trace_rows"]),
            status,
        )

    console.print()
    console.print(table)

    # Verify DB contents
    console.print("\n[bold]SQLite DB Verification:[/bold]")
    session_count = tracer.session.query(SessionRecord).count()
    step_count = tracer.session.query(StepRecord).count()
    gt_count_db = tracer.session.query(GroundTruthRecord).count()

    console.print(f"  Sessions table:     {session_count} rows")
    console.print(f"  Steps table:        {step_count} rows")
    console.print(f"  Ground truth table: {gt_count_db} rows")

    assert session_count >= len(tasks), f"Expected >= {len(tasks)} sessions, got {session_count}"
    assert step_count > 0, "Expected steps in DB"
    assert gt_count_db > 0, "Expected ground truth entries in DB"

    console.print(f"\n[bold green]✓ {success_count}/{len(tasks)} tasks ran successfully[/bold green]")
    console.print(f"[green]✓ Trace DB saved to {db_path}[/green]")

    if success_count < len(tasks):
        console.print(f"\n[yellow]⚠ {len(tasks) - success_count} tasks had errors:[/yellow]")
        for r in results:
            if not r["success"]:
                console.print(f"  [red]{r['task_id']}: {r['error']}[/red]")

    tracer.close()


if __name__ == "__main__":
    main()
