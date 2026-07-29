"""Day 8 STEP 3: Interceptor ablation — which level matters most?

Conditions (all run in 'early' mode):
  - l1_only: Only L1 (schema/tool errors) triggers abstention
  - l1_l2:  L1 + L2 (reasoning keywords). L2 alone still doesn't trigger;
            L2 alone counted, but counts as supporting signal with L1.
  - l1_l2_l3: Full stack with L3 output consistency at the end.

Uses same task set & models as the main interceptor run but with 60 tasks × 2 models
= 360 runs total per ablation (stays within budget).

Output: data/results/ablation_results.csv with per-run records.
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

from config import TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError
from experiments.run_interceptor import Interceptor

logger = logging.getLogger(__name__)
console = Console()


def run_ablation_task(
    task: dict,
    model_key: str,
    condition: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    """Run one task with the specified ablation condition.

    Conditions change which checks can trigger abstention:
      - l1_only: only L1 (schema/tool) can trigger
      - l1_l2:   L1 triggers (with threshold); L2 lowers threshold
      - l1_l2_l3: L1 + L3 can trigger; L2 supports
    """
    interceptor = Interceptor(mode="early")
    enable_l2 = condition in ("l1_l2", "l1_l2_l3")
    enable_l3 = condition == "l1_l2_l3"

    session_id = f"abl_{condition}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(session_id=session_id, task_id=task["task_id"],
                         model=model_key, domain=domain,
                         injection_type=f"ablation_{condition}",
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
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        llm_text = result["text"]
        step_num += 1
        tracer.log_step(session_id, 0, step_num, "thought", llm_text[:500])

        l2_hit = False
        if enable_l2:
            flagged, reason = interceptor.check_l2_thought(llm_text)
            if flagged:
                flag_history.append(reason)
                l2_hit = True

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

            # L1 check
            flagged, reason = interceptor.check_l1_step(step_data)
            if flagged:
                flag_history.append(reason)
                l1_count = sum(1 for f in flag_history if f.startswith("L1"))
                # L2 hits lower the threshold to trigger L1 abstention
                threshold = 1 if (enable_l2 and l2_hit) else 2
                if l1_count >= threshold:
                    abstained = True
                    abstain_reason = f"{condition}: {reason}"
                    final_answer = "I cannot reliably answer this question due to detected errors."
                    break

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Please provide a Final Answer."})

    if not final_answer:
        final_answer = "Max steps reached."

    # L3 check only in full stack
    if not abstained and enable_l3:
        flagged, reason = interceptor.check_l3_output(final_answer, observations_collected)
        if flagged:
            abstained = True
            abstain_reason = f"{condition} L3: {reason}"
            final_answer = "I cannot reliably answer this question."

    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))
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
        "l1_hits": sum(1 for f in flag_history if f.startswith("L1")),
        "l2_hits": sum(1 for f in flag_history if f.startswith("L2")),
        "final_correct": final_correct,
        "hallucination": hallucination,
        "cost": total_cost,
        "steps": step_num,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="data/tasks/full_500_tasks.json")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--conditions", type=str, default="l1_only,l1_l2,l1_l2_l3")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="data/results/ablation_results.csv")
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
        f"[bold cyan]Day 8 STEP 3 — Interceptor Ablation[/bold cyan]\n"
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
    db_path = os.path.join(TRACES_DIR, "ablation_day8.db")
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
        ptask = progress.add_task("Ablation runs", total=len(work_items))

        def _run(item):
            task, mk, cond = item
            router = ModelRouter(mk)
            return run_ablation_task(task, mk, cond, router, tool_executor, tracer, budget)

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

    fields = ["session_id", "task_id", "model", "domain", "condition", "final_answer",
              "abstained", "abstain_reason", "flag_count", "l1_hits", "l2_hits",
              "final_correct", "hallucination", "cost", "steps"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    console.print(f"[green]Saved {len(results)} rows to {args.output}[/green]")

    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby(["model", "condition"]).agg(
        n=("session_id", "count"),
        halluc_pct=("hallucination", "mean"),
        abstain_pct=("abstained", "mean"),
        correct_pct=("final_correct", "mean"),
    ).round(3)
    console.print("\n[bold]Ablation summary:[/bold]")
    console.print(summary)

    spent = budget.spent if not callable(budget.spent) else budget.spent()
    console.print(f"\nTotal cost: [bold]${spent:.4f}[/bold]")


if __name__ == "__main__":
    main()
