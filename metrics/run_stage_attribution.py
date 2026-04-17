"""Day 8 STEP 1: Run stage attribution on all P2 traces.

Uses rule-based StageAttributor first, falls back to LLM judge for ambiguous cases
(confidence < 0.6). Compares predicted stage vs injection target_stage (ground truth).

Outputs:
  - data/results/p1_attribution.csv with per-session predictions
  - Prints per-stage precision/recall/F1 (with bootstrap CIs)
  - Prints 6×6 confusion matrix (rich table)
"""

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from config import MODELS
from metrics.stage_attribution import StageAttributor
from analysis.bootstrap import bootstrap_ci, format_ci

logger = logging.getLogger(__name__)
console = Console()


JUDGE_PROMPT = """You are evaluating which stage of an LLM agent pipeline produced a wrong answer.

STAGES (pick exactly one):
- planning: The agent chose the wrong subtask/goal decomposition
- tool_selection: The agent selected the wrong tool for the task
- parameter_generation: The agent generated invalid parameters (wrong type, missing field, wrong value)
- tool_execution: The tool itself returned an error or wrong data
- memory_write: The agent saved wrong/contradictory info to memory
- memory_retrieval: The agent retrieved wrong info from memory/RAG
- output_generation: The agent had correct intermediate info but produced wrong final answer

AGENT QUERY: {query}
GROUND TRUTH ANSWER: {ground_truth}
FINAL ANSWER: {final_answer}

KEY STEPS (trace):
{steps}

Return ONLY the single stage name (e.g. "parameter_generation"). No other text.
"""


def ambiguous(attribution: dict) -> bool:
    """Return True if we should fall back to LLM judge."""
    conf = attribution.get("confidence", 0.0)
    method = attribution.get("attribution_method", "")
    return conf < 0.6 or method in ("default", "injection_record")


def format_steps_for_judge(trace: dict, max_steps: int = 8) -> str:
    """Condensed step listing for LLM judge prompt."""
    steps = trace.get("steps", [])[:max_steps]
    lines = []
    for s in steps:
        t = s.get("step_type", "?")
        if t == "action":
            params = s.get("tool_params_validated") or s.get("tool_params_raw") or "{}"
            if len(params) > 120:
                params = params[:120] + "..."
            err = s.get("param_errors")
            err_suffix = f" [PARAM_ERR: {err[:80]}]" if err else ""
            lines.append(f"- {t}: {s.get('tool_name', '?')}({params}){err_suffix}")
        elif t == "observation":
            tr = s.get("tool_result") or s.get("content", "")[:150]
            if len(tr) > 150:
                tr = tr[:150] + "..."
            lines.append(f"- {t}: {tr}")
        elif t == "thought":
            c = s.get("content", "")[:200]
            lines.append(f"- {t}: {c}")
        else:
            c = s.get("content", "")[:120]
            lines.append(f"- {t}: {c}")
    return "\n".join(lines) if lines else "(no steps logged)"


def llm_judge_stage(trace: dict, task_query: str, router) -> tuple[str, float]:
    """Use LLM to judge which stage caused the error.

    Returns (predicted_stage, cost).
    """
    from metrics.stage_attribution import StageAttributor

    prompt = JUDGE_PROMPT.format(
        query=task_query[:500],
        ground_truth=str(trace.get("ground_truth_answer", ""))[:300],
        final_answer=str(trace.get("final_answer", ""))[:300],
        steps=format_steps_for_judge(trace),
    )
    try:
        resp = router.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        text = resp["text"].strip().lower().replace('"', '').replace("'", "")
        # Extract recognized stage name
        for stg in StageAttributor.STAGES:
            if stg in text:
                return stg, resp.get("cost", 0.0)
        # If no match, default
        return "output_generation", resp.get("cost", 0.0)
    except Exception as e:
        logger.warning("LLM judge failed: %s", e)
        return "output_generation", 0.0


def load_traces_from_db(db_path: str, filter_injected_only: bool = True) -> list[dict]:
    """Load all session traces from the DB into memory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Sessions that have injections (P2)
    if filter_injected_only:
        c.execute("""
            SELECT s.* FROM sessions s
            INNER JOIN injections i ON s.session_id = i.session_id
            WHERE s.final_answer NOT LIKE 'ERROR%' AND s.final_answer IS NOT NULL
        """)
    else:
        c.execute("SELECT * FROM sessions WHERE final_answer NOT LIKE 'ERROR%'")

    sessions = [dict(r) for r in c.fetchall()]
    logger.info("Loaded %d sessions", len(sessions))

    all_traces = []
    for sess in sessions:
        sid = sess["session_id"]
        c.execute("SELECT * FROM steps WHERE session_id=? ORDER BY step_number", (sid,))
        steps = [dict(r) for r in c.fetchall()]
        c.execute("SELECT * FROM injections WHERE session_id=?", (sid,))
        injs = [dict(r) for r in c.fetchall()]

        trace = {
            "session_id": sid,
            "task_id": sess["task_id"],
            "model": sess["model"],
            "domain": sess["domain"],
            "injection_type": sess.get("injection_type"),
            "final_answer": sess.get("final_answer"),
            "ground_truth_answer": sess.get("ground_truth_answer"),
            "final_correct": bool(sess.get("final_correct")) if sess.get("final_correct") is not None else None,
            "steps": steps,
            "injections": injs,
        }
        all_traces.append(trace)
    conn.close()
    return all_traces


def injection_to_stage(injection_type: str) -> str:
    """Map injection_type → expected target stage.

    P2 error types all target parameter_generation stage.
    P3 memory injection targets memory_write.
    P4 planner targets planning.
    """
    p2_types = {"type_mismatch", "out_of_range", "missing_required", "semantic_wrong"}
    if injection_type in p2_types:
        return "parameter_generation"
    if injection_type and "memory" in injection_type:
        return "memory_write"
    if injection_type and "plan" in injection_type:
        return "planning"
    if injection_type and "tool" in injection_type:
        return "tool_selection"
    return "parameter_generation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/traces/p2_day6.db",
                        help="Comma-separated list of SQLite trace DBs")
    parser.add_argument("--llm-judge", default="gpt4o_mini")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--output", default="data/results/p1_attribution.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM judge")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    db_paths = [os.path.join(base, p.strip()) for p in args.db.split(",")]
    out_path = os.path.join(base, args.output)

    logging.basicConfig(level=logging.WARNING)

    console.rule("[bold cyan]Day 8 STEP 1 — Stage Attribution[/bold cyan]")

    traces = []
    for dbp in db_paths:
        if not os.path.exists(dbp):
            console.print(f"[yellow]DB not found: {dbp}[/yellow]")
            continue
        loaded = load_traces_from_db(dbp, filter_injected_only=True)
        console.print(f"  Loaded {len(loaded)} sessions from {os.path.basename(dbp)}")
        traces.extend(loaded)
    # Filter to sessions where the injection actually propagated (final_correct=False)
    # Those are the ones with a *final error* that we can meaningfully attribute.
    before = len(traces)
    traces = [t for t in traces if t.get("final_correct") is False]
    console.print(f"Loaded [bold]{before}[/bold] injected sessions; "
                  f"[bold]{len(traces)}[/bold] with propagated error (final_correct=False)")
    if args.limit > 0:
        traces = traces[: args.limit]

    attributor = StageAttributor()
    router = None
    if not args.no_llm:
        from agent.model_router import ModelRouter
        from experiments.parallel_runner import BudgetGuard
        router = ModelRouter(args.llm_judge)
        budget = BudgetGuard(args.budget)

    results = []
    judge_cost_total = 0.0
    judge_calls = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as prog:
        t = prog.add_task("Attributing", total=len(traces))
        for trace in traces:
            pred = attributor.attribute(trace)

            method = pred.get("attribution_method", "heuristic")
            if router and ambiguous(pred):
                # Fallback to LLM judge
                if budget.remaining() <= 0.005:
                    logger.warning("Budget exhausted, skipping LLM judge")
                else:
                    # Task query fallback: reconstruct from first thought
                    task_query = ""
                    for s in trace.get("steps", []):
                        if s.get("step_type") == "thought":
                            task_query = s.get("content", "")[:500]
                            break
                    if not task_query:
                        task_query = f"Task in {trace.get('domain', 'unknown')} domain"

                    stage, cost = llm_judge_stage(trace, task_query, router)
                    if stage:
                        pred["attributed_stage"] = stage
                        pred["confidence"] = 0.7
                        pred["attribution_method"] = "llm_judge"
                        budget.charge(cost)
                        judge_cost_total += cost
                        judge_calls += 1

            # Ground truth stage
            gt_stage = None
            injs = trace.get("injections") or []
            if injs:
                gt_stage = injs[0].get("target_stage")
            if not gt_stage:
                gt_stage = injection_to_stage(trace.get("injection_type"))
            # Normalize alternate names to canonical stage list
            _stage_aliases = {"memory": "memory_write", "memory_retrieve": "memory_retrieval"}
            if gt_stage in _stage_aliases:
                gt_stage = _stage_aliases[gt_stage]

            results.append({
                "session_id": trace["session_id"],
                "task_id": trace["task_id"],
                "model": trace["model"],
                "domain": trace["domain"],
                "injection_type": trace.get("injection_type"),
                "predicted_stage": pred.get("attributed_stage"),
                "ground_truth_stage": gt_stage,
                "confidence": pred.get("confidence"),
                "method": pred.get("attribution_method"),
                "final_correct": trace.get("final_correct"),
                "correct_attribution": pred.get("attributed_stage") == gt_stage,
            })
            prog.update(t, advance=1)

    console.print(f"LLM judge calls: [bold]{judge_calls}[/bold] / {len(traces)} "
                  f"({100*judge_calls/max(1, len(traces)):.1f}%)")
    console.print(f"Judge cost: [bold]${judge_cost_total:.4f}[/bold]")

    # Save CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fields = list(results[0].keys()) if results else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {out_path}[/green]")

    # Per-stage metrics with bootstrap CIs
    console.rule("[bold cyan]Per-Stage Metrics (with 95% CIs)[/bold cyan]")
    stages = StageAttributor.STAGES

    metrics_table = Table(title="Stage Attribution: Precision / Recall / F1")
    metrics_table.add_column("Stage")
    metrics_table.add_column("N (GT)", justify="right")
    metrics_table.add_column("Precision [95% CI]", justify="right")
    metrics_table.add_column("Recall [95% CI]", justify="right")
    metrics_table.add_column("F1 [95% CI]", justify="right")

    # Collect per-stage arrays
    for stage in stages:
        # Per-sample indicators
        tp_i = []
        fp_i = []
        fn_i = []
        precision_samples = []
        recall_samples = []
        for r in results:
            pred = r["predicted_stage"]
            gt = r["ground_truth_stage"]
            if gt == stage:
                tp_i.append(1 if pred == stage else 0)
            if pred == stage:
                fp_i.append(0 if gt == stage else 1)  # actually fp=1 when gt!=stage
        n_gt = sum(1 for r in results if r["ground_truth_stage"] == stage)
        n_pred = sum(1 for r in results if r["predicted_stage"] == stage)

        if n_gt == 0:
            metrics_table.add_row(stage, "0", "—", "—", "—")
            continue

        # precision = TP / (TP+FP)  -- evaluate on predictions
        pred_correct = [1 if r["ground_truth_stage"] == stage else 0
                        for r in results if r["predicted_stage"] == stage]
        # recall = TP / (TP+FN) -- evaluate on ground truths
        gt_recall = [1 if r["predicted_stage"] == stage else 0
                     for r in results if r["ground_truth_stage"] == stage]

        if pred_correct:
            p_point, p_lo, p_hi = bootstrap_ci(pred_correct, n_bootstrap=500)
        else:
            p_point = p_lo = p_hi = 0.0
        r_point, r_lo, r_hi = bootstrap_ci(gt_recall, n_bootstrap=500)

        # F1 from point estimates
        f1 = 2 * p_point * r_point / (p_point + r_point) if (p_point + r_point) > 0 else 0

        metrics_table.add_row(
            stage,
            str(n_gt),
            f"{p_point:.3f} [{p_lo:.2f},{p_hi:.2f}]",
            f"{r_point:.3f} [{r_lo:.2f},{r_hi:.2f}]",
            f"{f1:.3f}",
        )

    console.print(metrics_table)

    # Overall accuracy with bootstrap
    acc_samples = [1 if r["correct_attribution"] else 0 for r in results]
    a_point, a_lo, a_hi = bootstrap_ci(acc_samples)
    console.print(f"\nOverall accuracy: [bold]{format_ci(a_point, a_lo, a_hi)}[/bold]")

    # Confusion matrix
    console.rule("[bold cyan]Confusion Matrix (rows=GT, cols=Predicted)[/bold cyan]")
    cm = defaultdict(lambda: defaultdict(int))
    for r in results:
        cm[r["ground_truth_stage"]][r["predicted_stage"]] += 1

    all_stages_set = set(stages) | set(cm.keys()) | {s for row in cm.values() for s in row.keys()}
    all_stages_set.discard(None)
    all_stages = sorted(all_stages_set)
    cm_table = Table()
    cm_table.add_column("GT \\ Pred", style="bold")
    for s in all_stages:
        cm_table.add_column(s[:8], justify="right")
    for gt in all_stages:
        row = [gt[:18]]
        for p in all_stages:
            count = cm[gt][p]
            if count == 0:
                row.append("·")
            elif gt == p:
                row.append(f"[bold green]{count}[/bold green]")
            else:
                row.append(str(count))
        cm_table.add_row(*row)
    console.print(cm_table)

    # Save metrics JSON
    metrics_out = out_path.replace(".csv", "_metrics.json")
    metrics_summary = {
        "overall_accuracy": {"point": a_point, "ci_lo": a_lo, "ci_hi": a_hi},
        "total_predictions": len(results),
        "llm_judge_calls": judge_calls,
        "llm_judge_cost": judge_cost_total,
        "method_counts": {
            m: sum(1 for r in results if r["method"] == m)
            for m in set(r["method"] for r in results)
        },
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    console.print(f"[green]Metrics saved to {metrics_out}[/green]")


if __name__ == "__main__":
    main()
