"""Check current budget spend across all experiments."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from config import BUDGET_LIMIT_USD, RESULTS_DIR

console = Console()


def check_budget():
    budget_log_path = os.path.join(RESULTS_DIR, "budget_log.json")

    if not os.path.exists(budget_log_path):
        console.print("[yellow]No budget log found. No experiments run yet.[/yellow]")
        console.print(f"Budget limit: ${BUDGET_LIMIT_USD:.2f}")
        console.print(f"Spent: $0.00")
        console.print(f"Remaining: ${BUDGET_LIMIT_USD:.2f}")
        return

    with open(budget_log_path) as f:
        data = json.load(f)

    total_spent = data.get("total_spent", 0.0)
    limit = data.get("limit", BUDGET_LIMIT_USD)
    entries = data.get("entries", [])

    # Aggregate by model
    by_model = {}
    for entry in entries:
        model = entry.get("model", "unknown")
        if model not in by_model:
            by_model[model] = {"cost": 0.0, "calls": 0}
        by_model[model]["cost"] += entry["cost"]
        by_model[model]["calls"] += 1

    console.print(Panel.fit(
        f"[bold]AgentHallu-Bench Budget Status[/bold]\n"
        f"Total spent: [{'red' if total_spent > limit * 0.8 else 'green'}]${total_spent:.4f}[/]\n"
        f"Budget limit: ${limit:.2f}\n"
        f"Remaining: ${limit - total_spent:.4f}\n"
        f"Usage: {100 * total_spent / limit:.1f}%",
        border_style="blue",
    ))

    if by_model:
        table = Table(title="Spend by Model")
        table.add_column("Model")
        table.add_column("API Calls", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("% of Budget", justify="right")

        for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            pct = 100 * stats["cost"] / limit if limit > 0 else 0
            table.add_row(
                model,
                str(stats["calls"]),
                f"${stats['cost']:.4f}",
                f"{pct:.2f}%",
            )

        console.print(table)


if __name__ == "__main__":
    check_budget()
