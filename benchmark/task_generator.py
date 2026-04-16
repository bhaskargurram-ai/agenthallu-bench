"""Benchmark task generator — produces exactly 450 tasks across 3 domains.

Distribution: 50 easy + 50 medium + 50 hard per domain.
30% of tasks flagged multi_turn=True with num_turns=6.
All random ops seeded with RANDOM_SEED=42.
"""

import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from config import RANDOM_SEED, TASKS_DIR, TASKS_PER_DOMAIN
from benchmark.domains.weather_api import generate_all_tasks as gen_weather
from benchmark.domains.calendar_api import generate_all_tasks as gen_calendar
from benchmark.domains.medical_api import generate_all_tasks as gen_medical

logger = logging.getLogger(__name__)
console = Console()

MULTI_TURN_FRACTION = 0.30
MULTI_TURN_NUM_TURNS = 6


def _apply_multi_turn_flags(tasks: list[dict], rng: random.Random) -> list[dict]:
    """Flag 30% of tasks as multi_turn=True with num_turns=6."""
    n_multi = int(len(tasks) * MULTI_TURN_FRACTION)
    indices = rng.sample(range(len(tasks)), n_multi)
    for idx in indices:
        tasks[idx]["multi_turn"] = True
        tasks[idx]["num_turns"] = MULTI_TURN_NUM_TURNS
    return tasks


def generate_all_tasks() -> dict[str, list[dict]]:
    """Generate 450 tasks (150 per domain) deterministically.

    Returns dict mapping domain name to task list.
    """
    rng = random.Random(RANDOM_SEED)

    weather_tasks = gen_weather(rng)
    calendar_tasks = gen_calendar(rng)
    medical_tasks = gen_medical(rng)

    # Apply multi-turn flags
    weather_tasks = _apply_multi_turn_flags(weather_tasks, rng)
    calendar_tasks = _apply_multi_turn_flags(calendar_tasks, rng)
    medical_tasks = _apply_multi_turn_flags(medical_tasks, rng)

    return {
        "weather": weather_tasks,
        "calendar": calendar_tasks,
        "medical": medical_tasks,
    }


def save_tasks(all_tasks: dict[str, list[dict]]) -> None:
    """Save tasks to JSON files in TASKS_DIR."""
    os.makedirs(TASKS_DIR, exist_ok=True)

    for domain, tasks in all_tasks.items():
        path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2, default=str)
        logger.info("Saved %d %s tasks to %s", len(tasks), domain, path)


def print_summary(all_tasks: dict[str, list[dict]]) -> None:
    """Print task count and difficulty distribution table using rich."""
    table = Table(title="AgentHallu-Bench Task Distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Easy", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Hard", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Multi-turn", justify="right", style="yellow")

    grand_total = 0
    grand_multi = 0

    for domain, tasks in all_tasks.items():
        easy = sum(1 for t in tasks if t["difficulty"] == "easy")
        medium = sum(1 for t in tasks if t["difficulty"] == "medium")
        hard = sum(1 for t in tasks if t["difficulty"] == "hard")
        multi = sum(1 for t in tasks if t["multi_turn"])
        total = len(tasks)
        grand_total += total
        grand_multi += multi
        table.add_row(domain, str(easy), str(medium), str(hard), str(total), str(multi))

    table.add_row("TOTAL", "", "", "", str(grand_total), str(grand_multi), style="bold green")

    console.print(table)
    console.print(f"\n[green]Total tasks generated: {grand_total}[/green]")
    console.print(f"[yellow]Multi-turn tasks: {grand_multi} ({grand_multi/grand_total*100:.0f}%)[/yellow]")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    console.print("[bold blue]AgentHallu-Bench Task Generator[/bold blue]\n")
    console.print(f"Random seed: {RANDOM_SEED}")
    console.print(f"Tasks per domain: {TASKS_PER_DOMAIN}\n")

    all_tasks = generate_all_tasks()
    save_tasks(all_tasks)
    print_summary(all_tasks)

    console.print("\n[green]✓ All tasks saved to ./data/tasks/[/green]")


if __name__ == "__main__":
    main()
