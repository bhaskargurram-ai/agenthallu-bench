"""Ground truth recorder — inserts correct tool calls into SQLite.

For each task: correct_tool, correct_params, correct_answer → ground_truth table.
Verifies no task_id duplicates.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

from config import TASKS_DIR, TRACES_DIR
from tracer.trace_schema import GroundTruthRecord
from tracer.trace_logger import TraceLogger

logger = logging.getLogger(__name__)
console = Console()


def load_tasks(domain: str) -> list[dict]:
    """Load tasks from JSON file."""
    path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
    with open(path) as f:
        return json.load(f)


def record_ground_truth(tracer: TraceLogger, all_tasks: dict[str, list[dict]]) -> int:
    """Insert ground truth for all tasks into SQLite.

    Returns number of records inserted.
    """
    seen_ids = set()
    count = 0

    for domain, tasks in all_tasks.items():
        for task in tasks:
            task_id = task["task_id"]

            # Verify no duplicates
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task_id: {task_id}")
            seen_ids.add(task_id)

            # Get first tool from sequence as the primary correct tool
            seq = task["correct_tool_sequence"]
            correct_tool = seq[0]["tool"] if seq else ""
            correct_params = json.dumps(seq[0]["params"] if seq else {}, default=str)

            record = GroundTruthRecord(
                task_id=task_id,
                domain=domain,
                query=task["query"],
                correct_tool=correct_tool,
                correct_params=correct_params,
                correct_answer=task["correct_final_answer"],
                difficulty=task["difficulty"],
            )
            tracer.session.add(record)
            count += 1

    tracer.session.commit()
    logger.info("Recorded %d ground truth entries", count)
    return count


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    console.print("[bold blue]Ground Truth Recorder[/bold blue]\n")

    os.makedirs(TRACES_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "ground_truth.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()

    # Load all tasks
    all_tasks = {}
    for domain in ["weather", "calendar", "medical"]:
        all_tasks[domain] = load_tasks(domain)
        console.print(f"Loaded {len(all_tasks[domain])} {domain} tasks")

    count = record_ground_truth(tracer, all_tasks)
    console.print(f"\n[green]✓ Recorded {count} ground truth entries to {db_path}[/green]")

    # Verify no duplicates
    gt_count = tracer.session.query(GroundTruthRecord).count()
    console.print(f"[green]✓ Verified: {gt_count} unique task_ids in ground_truth table[/green]")

    tracer.close()


if __name__ == "__main__":
    main()
