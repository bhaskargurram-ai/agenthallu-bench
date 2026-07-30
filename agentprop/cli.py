"""`agentprop` command-line interface.

    agentprop eval --model gpt-4o-mini            # score a model -> leaderboard row
    agentprop eval --model my-model --api local   # local vLLM (set VLLM_BASE_URL)
    agentprop leaderboard                          # print the released 13-model board
    agentprop version
"""
import csv
import os

import click

from agentprop import __version__


@click.group()
def main():
    """AgentProp-Bench: evaluate tool-using LLM agents in one command."""


@main.command("eval")
@click.option("--model", required=True, help="config key (e.g. gpt4o_mini) or a raw model id.")
@click.option("--api", default=None, type=click.Choice(["openai", "google", "openrouter", "local"]),
              help="Backend for a raw model id (inferred if omitted).")
@click.option("--tasks", "tasks_path", default=None, help="Task JSON (defaults to bundled 500-task set).")
@click.option("--split", default="test", help='Task split to run ("test"/"val"/"all"). Default test.')
@click.option("--domain", default=None, help="Restrict to one domain (weather/calendar/medical/knowledge).")
@click.option("-n", "--limit", default=50, type=int, help="Number of tasks (default 50 for a quick run).")
@click.option("--workers", default=3, type=int, help="Concurrent tasks.")
@click.option("--out", default=None, help="Write per-task results + the summary row to this CSV.")
def eval_cmd(model, api, tasks_path, split, domain, limit, workers, out):
    """Run the benchmark for one model and print a leaderboard row."""
    from agentprop.data import load_tasks
    from agentprop.report import print_summary, summarize
    from agentprop.runner import evaluate

    split = None if split == "all" else split
    tasks = load_tasks(path=tasks_path, split=split, domain=domain, n=limit)
    if not tasks:
        raise click.ClickException("No tasks matched the given filters.")
    click.echo(f"Running {len(tasks)} tasks on '{model}' ...")
    results = evaluate(model, tasks, api=api, workers=workers)
    row = summarize(model, results)
    print_summary(row)

    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        row_path = out.replace(".csv", "") + "_row.csv"
        with open(row_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerow(row)
        click.echo(f"\nWrote per-task results -> {out}\nWrote leaderboard row -> {row_path}")


@main.command("leaderboard")
def leaderboard_cmd():
    """Print the released 13-model leaderboard (leaderboard/leaderboard.csv)."""
    from agentprop.data import _repo_root
    path = os.path.join(_repo_root(), "leaderboard", "leaderboard.csv")
    if not os.path.exists(path):
        raise click.ClickException(f"{path} not found. Run: python analysis/build_leaderboard.py")
    with open(path) as f:
        rows = list(csv.reader(f))
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title="AgentProp-Bench Leaderboard (13 models)")
        for h in rows[0]:
            t.add_column(h)
        for r in rows[1:]:
            t.add_row(*r)
        Console().print(t)
    except Exception:
        for r in rows:
            click.echo(",".join(r))


@main.command("version")
def version_cmd():
    click.echo(f"agentprop-bench {__version__}")


if __name__ == "__main__":
    main()
