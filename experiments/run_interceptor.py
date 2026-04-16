"""Day 8 STEP 2: Interceptor evaluation.

Conditions:
  - output_only: After agent finishes, verify final_answer against tool observations.
                 If inconsistent → return "Cannot reliably answer" (abstain).
  - early_interceptor: At EACH step, check for red flags (param_errors, tool errors,
                 suspicious LLM reasoning). If detected → halt with abstain.

Compares against baseline.csv (no_interceptor condition, reused as-is).

Metrics:
  - hallucination_rate: final_correct=False rate
  - abstain_rate: how often the interceptor kicks in
  - recall: of true hallucinations, how many the interceptor caught (abstained on)
  - FP%: of correct answers, how many the interceptor wrongly flagged
  - EPS: same metric as P2
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from config import MODELS, TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError

logger = logging.getLogger(__name__)
console = Console()

ERROR_KEYWORDS = {"error", "invalid", "unknown", "failed", "not found",
                  "missing", "incorrect", "wrong"}


class Interceptor:
    """Heuristic interceptor with 3 levels of checks.

    L1: Schema/validator errors (param_errors, tool error results) — cheap
    L2: Reasoning-level anomalies (error keywords in thoughts) — cheap
    L3: Consistency check (final_answer supported by observations) — cheap
    """

    def __init__(self, mode: str = "early"):
        """mode: 'output_only' or 'early'"""
        assert mode in ("output_only", "early"), mode
        self.mode = mode

    # ── L1 check: schema/tool-level ──────────────────────────────────────
    def check_l1_step(self, step: dict) -> tuple[bool, str]:
        """Return (flagged, reason). Called after each tool execution step."""
        pe = step.get("param_errors") or []
        if pe:
            return True, f"L1: param_errors={pe[:2]}"
        tr = step.get("tool_result")
        if isinstance(tr, dict) and "error" in tr:
            return True, f"L1: tool returned error: {str(tr.get('error'))[:80]}"
        return False, ""

    # ── L2 check: LLM-reasoning red flags ────────────────────────────────
    def check_l2_thought(self, thought_text: str) -> tuple[bool, str]:
        lower = thought_text.lower()
        hits = [kw for kw in ERROR_KEYWORDS if kw in lower]
        if len(hits) >= 2:  # Require 2+ keywords to reduce false positives
            return True, f"L2: reasoning contains {hits[:3]}"
        return False, ""

    # ── L3 check: final answer consistency ───────────────────────────────
    def check_l3_output(self, final_answer: str, observations: list[str]) -> tuple[bool, str]:
        """Flag if answer contains numbers not in any observation."""
        if not final_answer or not observations:
            return False, ""
        ans_nums = set(re.findall(r"-?\d+\.?\d*", final_answer))
        obs_nums = set()
        for obs in observations:
            obs_nums.update(re.findall(r"-?\d+\.?\d*", str(obs)))
        unsupported = {n for n in ans_nums - obs_nums
                       if len(n) > 1 or int(float(n)) > 9}
        if unsupported:
            return True, f"L3: answer has unsupported numbers: {list(unsupported)[:3]}"
        return False, ""


def run_interceptor_task(
    task: dict,
    model_key: str,
    condition: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    """Run one task with the specified interceptor condition."""
    interceptor = Interceptor(mode="early" if condition == "early_interceptor" else "output_only")
    session_id = f"int_{condition}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type=f"interceptor_{condition}",
                         injection_stage=None)

    tools = tool_executor.get_tools_for_domain(domain) if domain and domain != "knowledge" else tool_executor.list_tools()
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    system_msg = (
        "You are an agent that answers questions using tools.\n"
        "Follow this format exactly:\n"
        "Thought: [reasoning]\nAction: [tool_name]\nAction Input: {\"param\": \"value\"}\n\n"
        "When done, respond with:\nFinal Answer: [your answer]\n\n"
        f"Available tools:\n{tool_desc}"
    )
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": query}]

    total_cost = 0.0
    total_input = 0
    total_output = 0
    step_num = 0
    final_answer = ""
    abstained = False
    abstain_reason = ""
    flag_history = []
    observations_collected = []

    for iteration in range(5):
        try:
            result = router.complete(messages)
        except Exception as e:
            logger.error("LLM error: %s", e)
            final_answer = f"ERROR: {e}"
            break

        total_cost += result["cost"]
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        llm_text = result["text"]
        step_num += 1
        tracer.log_step(session_id, 0, step_num, "thought", llm_text[:500])

        # L2 check on reasoning (early mode only)
        if interceptor.mode == "early":
            flagged, reason = interceptor.check_l2_thought(llm_text)
            if flagged:
                flag_history.append(reason)
                # Don't abstain on L2 alone — require another signal

        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "final", final_answer)
            break

        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except Exception:
                tool_params = {}
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            exec_result = tool_executor.execute(tool_name, tool_params)
            obs = json.dumps(exec_result["result"], default=str)
            observations_collected.append(obs)
            step_data = {
                "tool_name": tool_name,
                "tool_result": exec_result.get("result"),
                "param_errors": exec_result.get("validation_errors"),
            }
            step_num += 1
            tracer.log_step(session_id, 0, step_num, "observation", obs[:1000],
                            tool_name=tool_name, tool_result=exec_result["result"],
                            param_errors=exec_result.get("validation_errors"))

            # L1 check on tool step (early mode only)
            if interceptor.mode == "early":
                flagged, reason = interceptor.check_l1_step(step_data)
                if flagged:
                    flag_history.append(reason)
                    # L1 alone is a strong signal (schema violation/tool error)
                    # Only abstain if we've seen 2+ L1 hits (avoids normal retries)
                    l1_count = sum(1 for f in flag_history if f.startswith("L1"))
                    if l1_count >= 2:
                        abstained = True
                        abstain_reason = f"Early interceptor: {reason}"
                        final_answer = "I cannot reliably answer this question due to detected errors in tool execution."
                        break

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    # L3 + output-only check: verify final_answer against observations
    if not abstained:
        flagged, reason = interceptor.check_l3_output(final_answer, observations_collected)
        # For output_only mode also flag on any L1 issue seen
        if interceptor.mode == "output_only":
            if any(f.startswith("L1") for f in flag_history) or flagged:
                # Count how many actually had issues
                l1_seen = sum(1 for f in flag_history if f.startswith("L1"))
                if l1_seen >= 2 or flagged:
                    abstained = True
                    abstain_reason = f"Output-only interceptor: {reason or 'L1 signals'}"
                    # Override answer
                    final_answer = "I cannot reliably answer this question."
        else:
            # Early mode: only L3 at the end
            if flagged:
                abstained = True
                abstain_reason = f"Early interceptor L3: {reason}"
                final_answer = "I cannot reliably answer this question."

    # Ground truth check
    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))
    # If abstained, we consider it "not correct" (but also not a hallucination)
    # — hallucination_rate counts non-abstain wrong answers
    hallucination = (not final_correct) and (not abstained)

    tracer.end_session(session_id, final_answer, gt, final_correct)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "condition": condition,
        "final_answer": final_answer[:200],
        "abstained": abstained,
        "abstain_reason": abstain_reason[:200],
        "flag_count": len(flag_history),
        "final_correct": final_correct,
        "hallucination": hallucination,
        "cost": total_cost,
        "steps": step_num,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--conditions", type=str, default="output_only,early_interceptor")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="data/results/interceptor_results.csv")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    tasks_path = os.path.join(os.path.dirname(__file__), "..", args.tasks)
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    all_tasks = [t for t in all_tasks if t.get("split", "test") == "test"]
    if args.limit > 0:
        all_tasks = all_tasks[: args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    console.print(Panel.fit(
        f"[bold cyan]Day 8 STEP 2 — Interceptor Evaluation[/bold cyan]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Conditions: {', '.join(conditions)}\n"
        f"Tasks: {len(all_tasks)}\n"
        f"Total runs: {len(model_keys) * len(conditions) * len(all_tasks)}\n"
        f"Budget: ${args.budget:.2f}",
        border_style="cyan",
    ))

    os.makedirs(TRACES_DIR, exist_ok=True)
    out_dir = os.path.dirname(args.output) or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "interceptor_day8.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()

    work_items = [(task, mk, cond) for mk in model_keys for cond in conditions for task in all_tasks]
    console.print(f"Work items: {len(work_items)}")

    results = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Interceptor runs", total=len(work_items))

        def _run(item):
            task, mk, cond = item
            router = ModelRouter(mk)
            return run_interceptor_task(task, mk, cond, router, tool_executor, tracer, budget)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run, item): item for item in work_items}
            for future in as_completed(futures):
                progress.advance(ptask)
                try:
                    r = future.result()
                    results.append(r)
                except BudgetExceededError as e:
                    console.print(f"\n[red bold]BUDGET EXCEEDED: {e}[/red bold]")
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:
                    logger.error("Failed: %s", e)

    # Save
    fields = ["session_id", "task_id", "model", "domain", "condition", "final_answer",
              "abstained", "abstain_reason", "flag_count", "final_correct", "hallucination",
              "cost", "steps"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {args.output}[/green]")

    # Summary
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby(["model", "condition"]).agg(
        n=("session_id", "count"),
        halluc_pct=("hallucination", "mean"),
        abstain_pct=("abstained", "mean"),
        correct_pct=("final_correct", "mean"),
    ).round(3)
    console.print("\n[bold]Summary by model × condition:[/bold]")
    console.print(summary)

    spent = budget.spent if not callable(budget.spent) else budget.spent()
    console.print(f"\nTotal cost: [bold]${spent:.4f}[/bold]")


if __name__ == "__main__":
    main()
