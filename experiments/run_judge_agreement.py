"""Day 8 STEP 5: LLM-judge agreement on final_correct labels.

Our main metric is final_correct which is computed by a simple substring match.
This is cheap but may be noisy. We validate it by sampling N=200 stratified
sessions across models × conditions, asking a stronger oracle (gpt4o) to judge
each answer, then computing Cohen's κ agreement with our heuristic.

If κ > 0.6, our heuristic is defensible. If not, we use oracle labels for a
sensitivity analysis in the paper.

Output:
  data/results/judge_agreement.csv — per-sample labels (heuristic, oracle, agree)
  data/results/judge_agreement_metrics.json — κ + disagreement breakdown
"""

import argparse
import csv
import json
import logging
import os
import random
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

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


def load_sessions_from_dbs(db_paths: list[str]) -> list[dict]:
    sessions = []
    for dbp in db_paths:
        if not os.path.exists(dbp):
            logger.warning("DB not found: %s", dbp)
            continue
        conn = sqlite3.connect(dbp)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT session_id, task_id, model, domain, injection_type,
                   final_answer, ground_truth_answer, final_correct
            FROM sessions
            WHERE final_answer IS NOT NULL AND final_answer NOT LIKE 'ERROR%'
        """)
        for r in c.fetchall():
            d = dict(r)
            d["source_db"] = os.path.basename(dbp)
            sessions.append(d)
        conn.close()
    return sessions


def stratified_sample(sessions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample approximately evenly across (model, injection_type) buckets."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in sessions:
        key = (s.get("model") or "?", s.get("injection_type") or "none")
        buckets[key].append(s)
    keys = list(buckets.keys())
    per_bucket = max(1, n // max(1, len(keys)))
    out = []
    for k in keys:
        pool = buckets[k]
        rng.shuffle(pool)
        out.extend(pool[:per_bucket])
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out[:n]


def load_task_queries(tasks_path: str) -> dict:
    with open(tasks_path) as f:
        tasks = json.load(f)
    return {t["task_id"]: t for t in tasks}


def oracle_judge(query: str, reference: str, candidate: str, router) -> tuple[bool, float]:
    prompt = JUDGE_PROMPT.format(
        query=query[:400],
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
        verdict = text.startswith("YES")
        return verdict, resp.get("cost", 0.0)
    except Exception as e:
        logger.warning("Oracle judge failed: %s", e)
        return False, 0.0


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Binary Cohen's κ between two label lists."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    pa_y = sum(1 for a in labels_a if a) / n
    pb_y = sum(1 for b in labels_b if b) / n
    pe = pa_y * pb_y + (1 - pa_y) * (1 - pb_y)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", type=str,
                        default="data/traces/baseline_day5.db,data/traces/p2_day6.db,data/traces/p3_day7.db,data/traces/p4_day7.db")
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--oracle", type=str, default="gpt4o")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--output", type=str, default="data/results/judge_agreement.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    base = os.path.join(os.path.dirname(__file__), "..")
    db_paths = [os.path.join(base, p.strip()) for p in args.dbs.split(",")]
    tasks_path = os.path.join(base, args.tasks)
    out_path = os.path.join(base, args.output)

    console.rule("[bold cyan]Day 8 STEP 5 — Judge Agreement[/bold cyan]")

    sessions = load_sessions_from_dbs(db_paths)
    console.print(f"Loaded [bold]{len(sessions)}[/bold] sessions from {len(db_paths)} DBs")

    if not sessions:
        console.print("[red]No sessions found. Exiting.[/red]")
        return

    sample = stratified_sample(sessions, args.n)
    console.print(f"Sampled [bold]{len(sample)}[/bold] stratified sessions")

    task_map = load_task_queries(tasks_path)

    from agent.model_router import ModelRouter
    from experiments.parallel_runner import BudgetGuard

    router = ModelRouter(args.oracle)
    budget = BudgetGuard(args.budget)

    results = []
    oracle_cost = 0.0

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        pt = prog.add_task("Oracle judging", total=len(sample))
        for s in sample:
            if budget.remaining() <= 0.01:
                console.print("[yellow]Budget exhausted[/yellow]")
                break
            tid = s.get("task_id")
            task = task_map.get(tid, {})
            query = task.get("query", "")
            reference = s.get("ground_truth_answer") or task.get("correct_final_answer", "")
            candidate = s.get("final_answer") or ""

            oracle_label, cost = oracle_judge(query, reference, candidate, router)
            budget.charge(cost)
            oracle_cost += cost

            heur_label = bool(s.get("final_correct"))

            results.append({
                "session_id": s["session_id"],
                "task_id": tid,
                "model": s.get("model"),
                "injection_type": s.get("injection_type"),
                "source_db": s.get("source_db"),
                "reference": str(reference)[:120],
                "candidate": candidate[:150],
                "heuristic_correct": heur_label,
                "oracle_correct": oracle_label,
                "agree": heur_label == oracle_label,
            })
            prog.update(pt, advance=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fields = list(results[0].keys()) if results else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {out_path}[/green]")

    if not results:
        return

    heur = [r["heuristic_correct"] for r in results]
    orac = [r["oracle_correct"] for r in results]
    agreement = sum(1 for r in results if r["agree"]) / len(results)
    kappa = cohens_kappa(heur, orac)

    # Confusion breakdown
    tab = Table(title="Heuristic × Oracle")
    tab.add_column("Heuristic")
    tab.add_column("Oracle YES", justify="right")
    tab.add_column("Oracle NO", justify="right")
    hy_oy = sum(1 for r in results if r["heuristic_correct"] and r["oracle_correct"])
    hy_on = sum(1 for r in results if r["heuristic_correct"] and not r["oracle_correct"])
    hn_oy = sum(1 for r in results if not r["heuristic_correct"] and r["oracle_correct"])
    hn_on = sum(1 for r in results if not r["heuristic_correct"] and not r["oracle_correct"])
    tab.add_row("YES", str(hy_oy), str(hy_on))
    tab.add_row("NO", str(hn_oy), str(hn_on))
    console.print(tab)

    console.print(f"\nRaw agreement: [bold]{agreement:.3f}[/bold]")
    console.print(f"Cohen's κ: [bold]{kappa:.3f}[/bold]")
    console.print(f"Oracle cost: [bold]${oracle_cost:.4f}[/bold]")

    metrics_out = out_path.replace(".csv", "_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump({
            "n": len(results),
            "raw_agreement": agreement,
            "cohens_kappa": kappa,
            "heuristic_yes_oracle_yes": hy_oy,
            "heuristic_yes_oracle_no": hy_on,
            "heuristic_no_oracle_yes": hn_oy,
            "heuristic_no_oracle_no": hn_on,
            "oracle_cost": oracle_cost,
        }, f, indent=2)
    console.print(f"[green]Metrics saved to {metrics_out}[/green]")


if __name__ == "__main__":
    main()
