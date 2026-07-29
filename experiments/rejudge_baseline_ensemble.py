"""Re-judge baseline_all.csv using the 3-LLM ensemble judge.

Reads baseline_all.csv + trace DB for final_answer + task file for queries,
sends through the 3-judge panel, writes baseline_all_rejudged.csv.

Usage:
  python experiments/rejudge_baseline_ensemble.py \
    --input data/results/baseline_all.csv \
    --db data/traces/baseline_day6.db \
    --tasks data/tasks/full_2000_tasks.json \
    --output data/results/baseline_all_rejudged.csv \
    --budget 2.0 --workers 5
"""

import argparse
import csv
import json
import logging
import os
import sqlite3
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/baseline_all.csv")
    parser.add_argument("--db", default="data/traces/baseline_day6.db")
    parser.add_argument("--tasks", default="data/tasks/full_2000_tasks.json")
    parser.add_argument("--judges", default="gpt4o,gemini_25_flash,gpt4o_mini")
    parser.add_argument("--output", default="data/results/baseline_all_rejudged.csv")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    base = os.path.join(os.path.dirname(__file__), "..")

    import pandas as pd
    from agent.model_router import ModelRouter
    from experiments.parallel_runner import BudgetGuard

    bl = pd.read_csv(os.path.join(base, args.input))
    console.print(f"Loaded {len(bl)} baseline rows")

    # Prefer final_answer already present in the input CSV (self-contained runs,
    # e.g. locally-served open-weight models). Fall back to the trace DB for
    # legacy CSVs (baseline_all.csv) that lack a final_answer column.
    have_fa = ("final_answer" in bl.columns
               and bl["final_answer"].astype(str).str.strip().replace({"nan": ""}).str.len().gt(0).any())
    if have_fa:
        console.print("Using final_answer from input CSV (skipping trace DB)")
        bl["final_answer"] = bl["final_answer"].fillna("").astype(str)
        if "ground_truth" not in bl.columns:
            bl["ground_truth"] = ""
    else:
        # Load final_answer from trace DB
        db_path = os.path.join(base, args.db)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT session_id, final_answer, ground_truth_answer FROM sessions")
        db_map = {row["session_id"]: dict(row) for row in c.fetchall()}
        conn.close()

        bl["final_answer"] = bl["session_id"].map(
            lambda s: db_map.get(s, {}).get("final_answer", ""))
        bl["ground_truth"] = bl["session_id"].map(
            lambda s: db_map.get(s, {}).get("ground_truth_answer", ""))

        matched = bl["final_answer"].apply(lambda x: bool(x and str(x) != "")).sum()
        console.print(f"Matched {matched}/{len(bl)} final_answers from trace DB")

    # Load task queries
    with open(os.path.join(base, args.tasks)) as f:
        tasks = json.load(f)
    # Also try full_500_tasks.json for older task_ids
    task_map = {t["task_id"]: t for t in tasks}
    for fn in ["data/tasks/full_500_tasks.json"]:
        try:
            with open(os.path.join(base, fn)) as f:
                extra = json.load(f)
            for t in extra:
                if t["task_id"] not in task_map:
                    task_map[t["task_id"]] = t
        except:
            pass

    bl["query"] = bl["task_id"].map(lambda t: task_map.get(t, {}).get("query", ""))

    # Fill missing ground_truth from task_map
    for idx, row in bl.iterrows():
        if not row.get("ground_truth") or str(row["ground_truth"]) == "":
            gt = task_map.get(row["task_id"], {}).get("correct_final_answer", "")
            bl.at[idx, "ground_truth"] = gt

    judge_keys = [j.strip() for j in args.judges.split(",")]
    routers = {k: ModelRouter(k) for k in judge_keys}
    budget = BudgetGuard(args.budget)

    def judge_row(idx, row):
        query = str(row.get("query", ""))
        reference = str(row.get("ground_truth", ""))
        candidate = str(row.get("final_answer", ""))
        if not candidate or candidate == "None" or not reference:
            return idx, {k: False for k in judge_keys}, 0.0
        verdicts = {}
        cost = 0.0
        for k in judge_keys:
            v, c = llm_judge(query, reference, candidate, routers[k])
            verdicts[k] = v
            cost += c
        return idx, verdicts, cost

    console.print(f"Judging {len(bl)} rows with {judge_keys}, budget=${args.budget}")
    results = {}
    total_cost = 0.0

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        pt = prog.add_task("Ensemble judging baseline", total=len(bl))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_row, idx, row): idx for idx, row in bl.iterrows()}
            for fut in as_completed(futures):
                try:
                    idx, verdicts, cost = fut.result()
                    budget.charge(cost)
                    total_cost += cost
                    results[idx] = verdicts
                except Exception as e:
                    logger.error("Judge error: %s", e)
                prog.update(pt, advance=1)
                if budget.remaining() < 0.01:
                    console.print("[red]Budget exhausted[/red]")
                    break

    for k in judge_keys:
        bl[f"{k}_verdict"] = bl.index.map(lambda i: results.get(i, {}).get(k, False))
    bl["ensemble_correct"] = bl[[f"{k}_verdict" for k in judge_keys]].sum(axis=1) >= 2
    bl["heuristic_correct"] = bl["final_correct"]

    out_path = os.path.join(base, args.output)
    bl.to_csv(out_path, index=False)
    console.print(f"[green]Saved {len(results)}/{len(bl)} judged rows to {out_path}[/green]")
    console.print(f"Total cost: ${total_cost:.4f}")

    judged = bl[bl.index.isin(results.keys())]
    if len(judged) > 0:
        agree = (judged["heuristic_correct"] == judged["ensemble_correct"]).mean()
        h_rate = judged["heuristic_correct"].mean()
        e_rate = judged["ensemble_correct"].mean()
        console.print(f"\nHeuristic correct rate: {100*h_rate:.1f}%")
        console.print(f"Ensemble correct rate:  {100*e_rate:.1f}%")
        console.print(f"Agreement:              {100*agree:.1f}%")

        for m in sorted(judged["model"].unique()):
            sub = judged[judged["model"] == m]
            console.print(f"  {m}: heur={sub['heuristic_correct'].mean()*100:.1f}%, ens={sub['ensemble_correct'].mean()*100:.1f}%, n={len(sub)}")


if __name__ == "__main__":
    main()
