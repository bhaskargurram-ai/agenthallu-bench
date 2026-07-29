"""Day 7 P4 experiment — cross-agent error propagation.

3-agent chain: Planner (LLM) → Executor (LLM+tools) → Synthesizer (LLM)
Inject hallucination into Agent 1's plan, measure propagation to Agents 2 & 3.

Usage:
  python experiments/run_p4_experiment.py \
    --models gpt4o_mini,gemini_20_flash,deepseek_v3 \
    --limit 60 \
    --injection-types all \
    --budget 3.0 \
    --output data/results/p4_results.csv \
    --resume
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import MODELS, RANDOM_SEED, TRACES_DIR, RESULTS_DIR
from agent.model_router import ModelRouter
from agent.tool_executor import ToolExecutor
from injector.propagation_injector import MultiAgentChain, PLANNER_PROMPT, EXECUTOR_PROMPT, SYNTHESIZER_PROMPT
from tracer.trace_logger import TraceLogger
from experiments.parallel_runner import BudgetGuard, BudgetExceededError

logger = logging.getLogger(__name__)
console = Console()

P4_INJECTION_TYPES = ["wrong_subtask", "wrong_tool_spec", "wrong_parameter_spec"]


def run_p4_task(
    task: dict,
    model_key: str,
    injection_type: str,
    router: ModelRouter,
    tool_executor: ToolExecutor,
    chain: MultiAgentChain,
    tracer: TraceLogger,
    budget: BudgetGuard,
) -> dict:
    """Run one P4 3-agent chain task with LLM calls."""
    session_id = f"p4_{injection_type}_{model_key}_{task['task_id']}_{uuid.uuid4().hex[:6]}"
    domain = task["domain"]
    query = task["query"]

    tracer.start_session(
        session_id=session_id,
        task_id=task["task_id"],
        model=model_key,
        domain=domain,
        injection_type=f"p4_{injection_type}",
        injection_stage="planning",
    )

    total_cost = 0.0
    step_num = 0

    # Build tool descriptions
    tools = tool_executor.get_tools_for_domain(domain) if domain and domain != "knowledge" else tool_executor.list_tools()
    tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    # ── Agent 1: Planner (LLM) ──────────────────────────────────────────
    planner_messages = [
        {"role": "system", "content": PLANNER_PROMPT + f"\n\nAvailable tools:\n{tool_desc}"},
        {"role": "user", "content": query},
    ]

    try:
        result = router.complete(planner_messages)
    except Exception as e:
        logger.error("Planner LLM error: %s", e)
        tracer.end_session(session_id, f"ERROR: {e}", "", False)
        return _error_result(session_id, task, model_key, injection_type, 0.0)

    total_cost += result["cost"]
    budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
    agent1_output = result["text"]

    step_num += 1
    tracer.log_step(session_id, 0, step_num, "thought",
                    f"Agent1 (Planner): {agent1_output[:500]}",
                    token_count=result["input_tokens"] + result["output_tokens"])

    # ── Inject into Agent 1's plan ──────────────────────────────────────
    original_plan = agent1_output
    injected_plan = chain.inject_agent1_hallucination(
        agent1_output, injection_type, domain, task,
    )

    tracer.log_injection(
        session_id=session_id,
        injection_type=f"p4_{injection_type}",
        target_stage="planning",
        original=original_plan[:500],
        injected=injected_plan[:500],
        turn_id=0,
    )

    # ── Agent 2: Executor (LLM + tools) ────────────────────────────────
    executor_messages = [
        {"role": "system", "content": EXECUTOR_PROMPT + f"\n\nAvailable tools:\n{tool_desc}"},
        {"role": "user", "content": f"Here is the plan from the Planner:\n{injected_plan}\n\nExecute each step."},
    ]

    agent2_tool_results = []
    agent2_text = ""

    for iteration in range(5):
        try:
            result = router.complete(executor_messages)
        except Exception as e:
            logger.error("Executor LLM error: %s", e)
            break

        total_cost += result["cost"]
        budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
        llm_text = result["text"]
        agent2_text += llm_text + "\n"

        step_num += 1
        tracer.log_step(session_id, 1, step_num, "thought",
                        f"Agent2 (Executor): {llm_text[:500]}",
                        token_count=result["input_tokens"] + result["output_tokens"])

        if "Result:" in llm_text or "Final Answer:" in llm_text:
            break

        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_text, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_params = json.loads(input_match.group(1)) if input_match else {}
            except (json.JSONDecodeError, AttributeError):
                tool_params = {}

            step_num += 1
            tracer.log_step(session_id, 1, step_num, "action", f"Action: {tool_name}",
                            tool_name=tool_name, tool_params_raw=tool_params)

            exec_result = tool_executor.execute(tool_name, tool_params)
            observation = json.dumps(exec_result["result"], default=str)
            agent2_tool_results.append({
                "tool": tool_name,
                "params": tool_params,
                "result": exec_result["result"],
                "validation_errors": exec_result.get("validation_errors", []),
            })

            step_num += 1
            tracer.log_step(session_id, 1, step_num, "observation", observation[:1000],
                            tool_name=tool_name, tool_result=exec_result["result"],
                            param_errors=exec_result.get("validation_errors"))

            executor_messages.append({"role": "assistant", "content": llm_text})
            executor_messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            break

    # ── Agent 3: Synthesizer (LLM) ─────────────────────────────────────
    synth_messages = [
        {"role": "system", "content": SYNTHESIZER_PROMPT},
        {"role": "user", "content": (
            f"Original question: {query}\n\n"
            f"Results from the Executor:\n{agent2_text[:2000]}\n\n"
            "Synthesize a final answer."
        )},
    ]

    try:
        result = router.complete(synth_messages)
    except Exception as e:
        logger.error("Synthesizer LLM error: %s", e)
        tracer.end_session(session_id, f"ERROR: {e}", "", False)
        return _error_result(session_id, task, model_key, injection_type, total_cost)

    total_cost += result["cost"]
    budget.charge(result["cost"], model=model_key, task_id=task["task_id"])
    agent3_output = result["text"]

    step_num += 1
    tracer.log_step(session_id, 2, step_num, "final",
                    f"Agent3 (Synthesizer): {agent3_output[:500]}",
                    token_count=result["input_tokens"] + result["output_tokens"])

    # Extract final answer
    if "Final Answer:" in agent3_output:
        final_answer = agent3_output.split("Final Answer:")[-1].strip()
    else:
        final_answer = agent3_output.strip()

    # Check correctness
    gt = task.get("correct_final_answer", "")
    final_correct = bool(gt and (gt.lower()[:20] in final_answer.lower()[:200] or
                                  any(w in final_answer.lower() for w in gt.lower().split()[:3] if len(w) > 3)))

    tracer.end_session(session_id, final_answer, gt, final_correct)

    # ── Measure propagation ──────────────────────────────────────────
    # Find diff tokens between original and injected plan
    diff_tokens = chain._find_diff_tokens(original_plan, injected_plan)

    # Check if error reached Agent 2
    agent2_str = agent2_text + json.dumps(agent2_tool_results, default=str)
    reached_agent2 = any(tok in agent2_str for tok in diff_tokens) if diff_tokens else False

    # Check if error reached Agent 3 / final output
    reached_agent3 = any(tok in agent3_output for tok in diff_tokens) if diff_tokens else False

    # Also: if agent2 had tool errors from injected params, error propagated
    has_tool_errors = any(
        r.get("validation_errors") or "error" in str(r.get("result", {}))
        for r in agent2_tool_results
    )

    # Impact-based: if final answer is wrong due to injection
    if not final_correct:
        reached_agent3 = True  # Error impacted output

    if reached_agent2 or has_tool_errors:
        reached_agent2 = True

    propagation_path = [1]
    if reached_agent2:
        propagation_path.append(2)
    if reached_agent3:
        propagation_path.append(3)

    amplification = len(propagation_path)

    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": domain,
        "difficulty": task.get("difficulty", "medium"),
        "injection_type": injection_type,
        "reached_agent2": reached_agent2,
        "reached_agent3": reached_agent3,
        "has_tool_errors": has_tool_errors,
        "propagation_path": json.dumps(propagation_path),
        "amplification_factor": amplification,
        "final_correct": final_correct,
        "cost": total_cost,
    }


def _error_result(session_id, task, model_key, injection_type, cost):
    return {
        "session_id": session_id,
        "task_id": task["task_id"],
        "model": model_key,
        "domain": task.get("domain", ""),
        "difficulty": task.get("difficulty", "medium"),
        "injection_type": injection_type,
        "reached_agent2": False,
        "reached_agent3": False,
        "has_tool_errors": False,
        "propagation_path": json.dumps([1]),
        "amplification_factor": 1,
        "final_correct": False,
        "cost": cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Day 7 P4 Cross-Agent Experiment")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--injection-types", type=str, default="all")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--output", type=str, default="data/results/p4_results.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    model_keys = [m.strip() for m in args.models.split(",")]
    for mk in model_keys:
        if mk not in MODELS:
            console.print(f"[red]Unknown model: {mk}[/red]")
            sys.exit(1)

    if args.injection_types == "all":
        injection_types = P4_INJECTION_TYPES
    else:
        injection_types = [t.strip() for t in args.injection_types.split(",")]

    # Load tasks (test split, exclude knowledge)
    tasks_path = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", "full_500_tasks.json")
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    all_tasks = [t for t in all_tasks if t.get("split", "test") == "test" and t.get("domain") != "knowledge"]
    if args.limit > 0:
        all_tasks = all_tasks[:args.limit]

    budget = BudgetGuard(limit_usd=args.budget)

    console.print(Panel.fit(
        f"[bold blue]Day 7 — P4 Cross-Agent Experiment[/bold blue]\n"
        f"Models: {', '.join(model_keys)}\n"
        f"Tasks: {len(all_tasks)}\n"
        f"Injection types: {', '.join(injection_types)}\n"
        f"Total runs: {len(all_tasks) * len(model_keys) * len(injection_types)}\n"
        f"Budget: ${args.budget:.2f}",
        border_style="blue",
    ))

    # Setup
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(TRACES_DIR, "p4_day7.db")
    tracer = TraceLogger(db_path)
    tracer.init_db()
    tool_executor = ToolExecutor()
    chain = MultiAgentChain(tool_executor, seed=RANDOM_SEED)

    # Resume check
    completed_triples = set()
    output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    if args.resume and os.path.exists(output_path):
        try:
            import pandas as pd
            existing = pd.read_csv(output_path)
            for _, row in existing.iterrows():
                completed_triples.add((row["task_id"], row["model"], row["injection_type"]))
            console.print(f"[yellow]Resuming: {len(completed_triples)} already done[/yellow]")
        except Exception:
            pass

    # Build work items
    work_items = []
    for mk in model_keys:
        for it in injection_types:
            for task in all_tasks:
                if (task["task_id"], mk, it) in completed_triples:
                    continue
                work_items.append((task, mk, it))

    console.print(f"Work items: {len(work_items)}")

    results = []
    errors_count = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        ptask = progress.add_task("Running P4...", total=len(work_items))

        def _run(item):
            task, mk, it = item
            router = ModelRouter(mk)
            return run_p4_task(task, mk, it, router, tool_executor, chain,
                               tracer, budget)

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
                    errors_count += 1
                    logger.error("Failed: %s", e)

    # Save CSV
    if results:
        fieldnames = ["session_id", "task_id", "model", "domain", "difficulty",
                       "injection_type", "reached_agent2", "reached_agent3",
                       "has_tool_errors", "propagation_path", "amplification_factor",
                       "final_correct", "cost"]
        mode = "a" if args.resume and os.path.exists(output_path) else "w"
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for r in results:
                writer.writerow(r)

    # Summary
    console.print(f"\n[green]Completed: {len(results)}, Errors: {errors_count}[/green]")

    table = Table(title="P4 Results — Cross-Agent Propagation")
    table.add_column("Model")
    table.add_column("Inj Type")
    table.add_column("→Agent2 %", justify="right")
    table.add_column("→Agent3 %", justify="right")
    table.add_column("Amplification", justify="right")
    table.add_column("Cost", justify="right")

    total_cost = 0.0
    for mk in model_keys:
        for it in injection_types:
            subset = [r for r in results if r["model"] == mk and r["injection_type"] == it]
            if not subset:
                continue
            n = len(subset)
            a2_pct = 100 * sum(1 for r in subset if r["reached_agent2"]) / n
            a3_pct = 100 * sum(1 for r in subset if r["reached_agent3"]) / n
            mean_amp = sum(r["amplification_factor"] for r in subset) / n
            cost = sum(r["cost"] for r in subset)
            total_cost += cost
            table.add_row(mk, it, f"{a2_pct:.0f}%", f"{a3_pct:.0f}%",
                           f"{mean_amp:.2f}", f"${cost:.4f}")

    console.print(table)
    console.print(f"\n[bold]Total cost: ${total_cost:.4f}[/bold]")
    console.print(f"Budget: {budget.status()}")
    console.print(f"Output: {args.output}")

    budget.save_log(os.path.join(RESULTS_DIR, "budget_log.json"))
    tracer.close()


if __name__ == "__main__":
    main()
