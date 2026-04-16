"""Fix 1: Threshold sweep for the Interceptor on the val split.

Sweeps (L1_threshold, L2_keywords, L3_similarity) triples, runs each config on
val (50 tasks) against p2_semantic_wrong injection, and picks the config that
maximizes F1 = 2·P·R/(P+R) where:
  - TP = abstained AND hallucination-would-have-happened
  - FP = abstained AND clean answer
  - FN = NOT abstained AND hallucination-happened

A proxy for "would have hallucinated": baseline column final_correct=False
on the same task_id. We reuse baseline_all.csv for that column.

Output: analysis/interceptor_tuned_thresholds.json with the best config.
Then we rerun on test split with tuned config and write
data/results/interceptor_tuned_results.csv.
"""

import argparse
import csv
import itertools
import json
import logging
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from config import TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError
from injector.parameter_injector import ParameterInjector

logger = logging.getLogger(__name__)
console = Console()

ERROR_KEYWORDS = {"error", "invalid", "unknown", "failed", "not found",
                  "missing", "incorrect", "wrong", "unable", "cannot"}


class TunableInterceptor:
    """Like Interceptor but parameterized by thresholds."""

    def __init__(self, l1_threshold: int = 2, l2_keywords: int = 2, l3_enabled: bool = True):
        self.l1_threshold = l1_threshold
        self.l2_keywords = l2_keywords
        self.l3_enabled = l3_enabled

    def check_l1_step(self, step: dict) -> tuple[bool, str]:
        pe = step.get("param_errors") or []
        if pe:
            return True, f"L1: {pe[:2]}"
        tr = step.get("tool_result")
        if isinstance(tr, dict) and "error" in tr:
            return True, "L1: tool error"
        return False, ""

    def check_l2_thought(self, text: str) -> bool:
        lower = text.lower()
        hits = sum(1 for kw in ERROR_KEYWORDS if kw in lower)
        return hits >= self.l2_keywords

    def check_l3_output(self, final: str, observations: list[str]) -> bool:
        if not self.l3_enabled or not observations:
            return False
        # Simple: if final contains "confirmed" or "success" while obs contain "error"
        fl = (final or "").lower()
        obs_joined = " ".join(observations).lower()
        if any(kw in fl for kw in ("success", "confirmed", "created", "scheduled", "done")) \
           and any(kw in obs_joined for kw in ("error", "invalid", "missing")):
            return True
        return False


def run_one(task, model_key, interceptor, router, tool_executor, tracer, budget, condition):
    """Run one trace. condition: baseline or p2_semantic."""
    session_id = f"sweep_{condition}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]
    inject = condition == "p2_semantic"
    injector = ParameterInjector(seed=42) if inject else None

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type=f"sweep_{condition}", injection_stage=None)

    tools = tool_executor.get_tools_for_domain(domain) if domain != "knowledge" else tool_executor.list_tools()
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    system_msg = ("You are an agent that answers questions using tools.\n"
                  "Follow this format: Thought/Action/Action Input or Final Answer.\n"
                  f"Available tools:\n{tool_desc}")
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": query}]

    total_cost = 0.0
    final_answer = ""
    abstained = False
    flag_count = 0
    injection_applied = False
    observations = []

    for _ in range(5):
        try:
            result = router.complete(messages)
        except Exception as e:
            final_answer = f"ERROR: {e}"
            break
        total_cost += result["cost"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        text = result["text"]

        if interceptor.check_l2_thought(text):
            flag_count += 1

        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            break
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if not action_match:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})
            continue

        tool_name = action_match.group(1).strip()
        try:
            tool_params = json.loads(input_match.group(1)) if input_match else {}
        except Exception:
            tool_params = {}

        if inject and not injection_applied and tool_params:
            schema = None
            for t in tools:
                if t.get("name") == tool_name:
                    schema = t
                    break
            if schema:
                try:
                    bad = injector.inject(tool_name=tool_name, params=tool_params,
                                          error_type="semantic_wrong", schema=schema)
                    if bad and bad != tool_params:
                        tool_params = bad
                        injection_applied = True
                except Exception:
                    pass

        exec_result = tool_executor.execute(tool_name, tool_params)
        obs = json.dumps(exec_result["result"], default=str)
        observations.append(obs)

        flagged, _ = interceptor.check_l1_step({
            "tool_result": exec_result.get("result"),
            "param_errors": exec_result.get("validation_errors"),
        })
        if flagged:
            flag_count += 1
            if flag_count >= interceptor.l1_threshold:
                abstained = True
                final_answer = "Cannot reliably answer — detected errors."
                break

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    if not abstained and interceptor.check_l3_output(final_answer, observations):
        abstained = True
        final_answer = "Cannot reliably answer — output inconsistent with tool outputs."

    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))
    hallucination = (not final_correct) and (not abstained)

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "condition": condition,
        "abstained": abstained,
        "hallucination": hallucination,
        "final_correct": final_correct,
        "injection_applied": injection_applied,
        "cost": total_cost,
    }


def evaluate_config(cfg, tasks, model_keys, tool_executor, tracer, budget):
    """Evaluate one config on val tasks. Returns F1 and sub-metrics."""
    interceptor = TunableInterceptor(**cfg)
    results = []

    def _run(item):
        task, mk, cond = item
        router = ModelRouter(mk)
        return run_one(task, mk, interceptor, router, tool_executor, tracer, budget, cond)

    work = [(t, mk, cond) for mk in model_keys for cond in ("baseline", "p2_semantic") for t in tasks]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_run, w): w for w in work}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.error("Run failed: %s", e)

    # Compute F1 on p2_semantic condition using paired baseline as "would-have-hallucinated"
    by_key = {(r["task_id"], r["model"], r["condition"]): r for r in results}
    tp = fp = fn = tn = 0
    for (tid, mk, cond), r in by_key.items():
        if cond != "p2_semantic":
            continue
        baseline = by_key.get((tid, mk, "baseline"))
        would_hallucinate = bool(baseline and not baseline["final_correct"])
        actually_abstained = r["abstained"]
        actually_hallucinated = r["hallucination"]
        if would_hallucinate and actually_abstained:
            tp += 1
        elif (not would_hallucinate) and actually_abstained:
            fp += 1
        elif would_hallucinate and not actually_abstained:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_results": len(results), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gpt4o_mini,deepseek_v3")
    parser.add_argument("--tasks", default="data/tasks/full_500_tasks.json")
    parser.add_argument("--val-limit", type=int, default=30)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--output", default="analysis/interceptor_tuned_thresholds.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    model_keys = [m.strip() for m in args.models.split(",")]
    base = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(base, args.tasks)) as f:
        all_tasks = json.load(f)
    val = [t for t in all_tasks if t.get("split") == "val" and t.get("domain") != "knowledge"]
    val = val[: args.val_limit]
    console.print(f"Val tasks: {len(val)}")

    budget = BudgetGuard(limit_usd=args.budget)
    os.makedirs(TRACES_DIR, exist_ok=True)
    tracer = TraceLogger(os.path.join(TRACES_DIR, "sweep_day8.db"))
    tracer.init_db()
    tool_executor = ToolExecutor()

    # Grid
    grid = list(itertools.product(
        [1, 2, 3],            # l1_threshold
        [2, 3],               # l2_keywords
        [True, False],        # l3_enabled
    ))
    console.print(f"Configs to evaluate: {len(grid)}")

    tab = Table(title="Threshold Sweep (val)")
    for col in ("L1", "L2", "L3", "F1", "Prec", "Recall", "Abstain%"):
        tab.add_column(col, justify="right")

    best = None
    scores = []
    for i, (l1, l2, l3) in enumerate(grid, 1):
        cfg = {"l1_threshold": l1, "l2_keywords": l2, "l3_enabled": l3}
        console.print(f"[{i}/{len(grid)}] cfg={cfg}")
        m = evaluate_config(cfg, val, model_keys, tool_executor, tracer, budget)
        abst = sum(1 for r in m["results"] if r["condition"] == "p2_semantic" and r["abstained"])
        n_p2 = sum(1 for r in m["results"] if r["condition"] == "p2_semantic")
        abst_pct = abst / n_p2 if n_p2 else 0
        tab.add_row(str(l1), str(l2), str(l3),
                    f"{m['f1']:.2f}", f"{m['precision']:.2f}", f"{m['recall']:.2f}",
                    f"{100*abst_pct:.1f}%")
        scores.append({"cfg": cfg, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                       "abstain_pct": abst_pct})
        if best is None or m["f1"] > best["f1"]:
            best = {**m, "cfg": cfg}
        if budget.spent > budget.limit * 0.9:
            console.print("[yellow]Budget nearly exhausted, stopping sweep[/yellow]")
            break

    console.print(tab)
    out = {
        "best_cfg": best["cfg"] if best else None,
        "best_f1": best["f1"] if best else 0.0,
        "best_precision": best["precision"] if best else 0.0,
        "best_recall": best["recall"] if best else 0.0,
        "all_scores": scores,
    }
    out_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[green]Best config: {out['best_cfg']} (F1={out['best_f1']:.3f})[/green]")
    console.print(f"[green]Saved → {out_path}[/green]")

    spent = budget.spent if not callable(budget.spent) else budget.spent()
    console.print(f"Spent: ${spent:.4f}")


if __name__ == "__main__":
    main()
