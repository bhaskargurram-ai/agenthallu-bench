"""Re-judge p2_all.csv using the 3-LLM ensemble judge.

Reads p2_all.csv + task file, sends each (query, reference, final_answer)
through the same 3-judge panel used in run_ensemble_judge.py, and writes
an updated CSV with columns:
  - ensemble_correct (majority vote of 3 judges)
  - gpt4o_verdict, gemini_25_flash_verdict, gpt4o_mini_verdict
  - heuristic_correct (original final_correct from p2_all.csv)

Usage:
  python experiments/rejudge_p2_ensemble.py \
    --input data/results/p2_all.csv \
    --tasks data/tasks/full_2000_tasks.json \
    --output data/results/p2_all_rejudged.csv \
    --budget 5.0 \
    --workers 5

After running, diff heuristic_correct vs ensemble_correct to measure drift.
If acceptable, replace final_correct in p2_all.csv with ensemble_correct
and re-run all downstream analysis.
"""

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

logger = logging.getLogger(__name__)
console = Console()

JUDGE_PROMPT = """You are an impartial judge evaluating whether an agent's final answer is correct.

QUESTION: {query}
REFERENCE ANSWER: {reference}
AGENT ANSWER: {candidate}

Respond with a single token: YES if the agent answer substantively matches the reference,
NO otherwise. Accept minor wording differences, units expressed differently, or extra
explanation as long as the core fact is correct.

YOUR VERDICT (YES or NO):"""


def llm_judge(query, reference, candidate, router):
    prompt = JUDGE_PROMPT.format(
        query=str(query)[:400],
        reference=str(reference)[:300],
        candidate=str(candidate)[:300],
    )
    try:
        resp = router.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = resp["text"].strip().upper()
        return text.startswith("YES"), resp.get("cost", 0.0)
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return False, 0.0


def main():
    parser = argparse.ArgumentParser(description="Re-judge P2 traces with 3-LLM ensemble")
    parser.add_argument("--input", default="data/results/p2_all.csv")
    parser.add_argument("--tasks", default="data/tasks/full_2000_tasks.json")
    parser.add_argument("--judges", default="gpt4o,gemini_25_flash,gpt4o_mini")
    parser.add_argument("--output", default="data/results/p2_all_rejudged.csv")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--error-type", default="semantic_wrong",
                        help="Only rejudge this error_type (or 'all')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    base = os.path.join(os.path.dirname(__file__), "..")

    import pandas as pd
    from agent.model_router import ModelRouter
    from experiments.parallel_runner import BudgetGuard

    # Load P2 data
    p2_path = os.path.join(base, args.input)
    p2 = pd.read_csv(p2_path)
    console.print(f"Loaded {len(p2)} P2 rows from {args.input}")

    # Filter by error_type if specified
    if args.error_type != "all" and "error_type" in p2.columns:
        p2 = p2[p2["error_type"] == args.error_type].copy()
        console.print(f"Filtered to error_type={args.error_type}: {len(p2)} rows")

    # Load task queries
    tasks_path = os.path.join(base, args.tasks)
    with open(tasks_path) as f:
        tasks = json.load(f)
    task_map = {t["task_id"]: t for t in tasks}

    # We need final_answer from the trace DB or session data
    # p2_all.csv doesn't have final_answer column — check
    if "final_answer" not in p2.columns:
        console.print("[yellow]p2_all.csv has no final_answer column.[/yellow]")
        console.print("[yellow]Attempting to load from trace DB...[/yellow]")

        import sqlite3
        db_path = os.path.join(base, "data/traces/p2_day6.db")
        if not os.path.exists(db_path):
            console.print(f"[red]Trace DB not found: {db_path}[/red]")
            console.print("[red]Cannot rejudge without final_answer text. "
                          "Either add final_answer column to p2_all.csv or provide trace DB.[/red]")
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, final_answer, ground_truth_answer FROM sessions")
        db_answers = {row["session_id"]: dict(row) for row in cursor.fetchall()}
        conn.close()

        p2["final_answer"] = p2["session_id"].map(
            lambda sid: db_answers.get(sid, {}).get("final_answer", ""))
        p2["ground_truth"] = p2["session_id"].map(
            lambda sid: db_answers.get(sid, {}).get("ground_truth_answer", ""))
        matched = p2["final_answer"].apply(lambda x: bool(x and x != "")).sum()
        console.print(f"Matched {matched}/{len(p2)} final_answers from trace DB")
    else:
        p2["ground_truth"] = p2["task_id"].map(
            lambda tid: task_map.get(tid, {}).get("correct_final_answer", ""))

    # Also get query from task map
    p2["query"] = p2["task_id"].map(
        lambda tid: task_map.get(tid, {}).get("query", ""))

    # Setup judges
    judge_keys = [j.strip() for j in args.judges.split(",")]
    routers = {k: ModelRouter(k) for k in judge_keys}
    budget = BudgetGuard(args.budget)

    def judge_row(idx, row):
        query = row.get("query", "")
        reference = row.get("ground_truth", "") or task_map.get(row["task_id"], {}).get("correct_final_answer", "")
        candidate = row.get("final_answer", "")

        if not candidate or not reference:
            return idx, {k: False for k in judge_keys}, 0.0

        verdicts = {}
        cost = 0.0
        for k in judge_keys:
            v, c = llm_judge(query, reference, candidate, routers[k])
            verdicts[k] = v
            cost += c
        return idx, verdicts, cost

    console.print(f"Judging {len(p2)} rows with {judge_keys}, budget=${args.budget}")

    results = {}
    total_cost = 0.0

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        pt = prog.add_task("Ensemble judging P2", total=len(p2))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for idx, row in p2.iterrows():
                fut = pool.submit(judge_row, idx, row)
                futures[fut] = idx

            for fut in as_completed(futures):
                try:
                    idx, verdicts, cost = fut.result()
                    budget.charge(cost)
                    total_cost += cost
                    results[idx] = verdicts
                except Exception as e:
                    logger.error("Judge error for idx %s: %s", futures[fut], e)
                prog.update(pt, advance=1)
                if budget.remaining() < 0.01:
                    console.print("[red]Budget exhausted[/red]")
                    break

    # Add verdict columns
    for k in judge_keys:
        p2[f"{k}_verdict"] = p2.index.map(lambda i: results.get(i, {}).get(k, False))

    p2["ensemble_correct"] = p2[[f"{k}_verdict" for k in judge_keys]].sum(axis=1) >= 2
    p2["heuristic_correct"] = p2["final_correct"]

    # Save
    out_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p2.to_csv(out_path, index=False)
    console.print(f"[green]Saved {len(results)}/{len(p2)} judged rows to {out_path}[/green]")
    console.print(f"Total cost: ${total_cost:.4f}")

    # Quick comparison
    judged = p2[p2.index.isin(results.keys())]
    if len(judged) > 0:
        agree = (judged["heuristic_correct"] == judged["ensemble_correct"]).mean()
        heur_rate = judged["heuristic_correct"].mean()
        ens_rate = judged["ensemble_correct"].mean()
        console.print(f"\nHeuristic correct rate: {100*heur_rate:.1f}%")
        console.print(f"Ensemble correct rate:  {100*ens_rate:.1f}%")
        console.print(f"Agreement:              {100*agree:.1f}%")
        console.print(f"Drift (ensemble - heuristic): {100*(ens_rate - heur_rate):.1f} pp")


if __name__ == "__main__":
    main()
