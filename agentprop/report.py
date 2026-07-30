"""Summarize per-task results into a leaderboard-style row + pretty output."""


def summarize(model, results):
    """Aggregate a list of run_task dicts into one leaderboard row."""
    n = len(results) or 1
    acc = sum(1 for r in results if r.get("final_correct")) / n
    tool_acc = sum(1 for r in results if r.get("tool_correct")) / n
    call_rate = sum(1 for r in results if r.get("emitted_real_call")) / n
    cost = sum(r.get("cost", 0) for r in results)
    steps = sum(r.get("steps", 0) for r in results) / n
    return {
        "model": model,
        "n": len(results),
        "acc_substring_judge": round(acc, 3),
        "tool_correct": round(tool_acc, 3),
        "tool_call_pct": round(100 * call_rate, 1),
        "avg_steps": round(steps, 2),
        "cost_usd": round(cost, 4),
    }


def print_summary(row):
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title=f"AgentProp-Bench — {row['model']}  (n={row['n']})")
        for k in ("acc_substring_judge", "tool_correct", "tool_call_pct", "avg_steps", "cost_usd"):
            t.add_column(k)
        t.add_row(*(str(row[k]) for k in
                    ("acc_substring_judge", "tool_correct", "tool_call_pct", "avg_steps", "cost_usd")))
        Console().print(t)
    except Exception:
        print(row)
    print("\nNote: this row reports the baseline (no-injection) accuracy under a "
          "substring judge, tool-call rate, and cost — directly comparable across "
          "models. Full propagation (EPS), the validated ensemble judge, and "
          "fabrication rates come from the released pipeline:\n"
          "  python experiments/run_p2_experiment.py    # injection / propagation\n"
          "  python experiments/run_ensemble_judge.py    # validated judge\n"
          "  python analysis/build_leaderboard.py        # rebuild leaderboard.csv")
