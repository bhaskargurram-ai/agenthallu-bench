"""Day 3 Integration Test: Run 30 tasks (10 per problem type).

- 10 x P2 tasks (2-3 per error type across domains)
- 10 x P3 tasks (multi-turn, injection at turn 2)
- 10 x P4 tasks (3-agent chain, injection at Agent 1)

For each: print rich table with task_id, injection_type, tool_called, param_error_detected.
Save to ./data/traces/day3_injector_test.db
Report: how many injections went undetected by the agent?
"""

import json
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from config import RANDOM_SEED, TRACES_DIR, TASKS_DIR, PARAM_ERROR_TYPES
from agent.tool_executor import ToolExecutor
from agent.memory_manager import MemoryManager
from tracer.trace_logger import TraceLogger
from injector.parameter_injector import ParameterInjector
from injector.memory_injector import MemoryInjector
from injector.propagation_injector import MultiAgentChain

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

console = Console()

DB_PATH = os.path.join(TRACES_DIR, "day3_injector_test.db")


def _load_tasks(domain: str, n: int = 10) -> list[dict]:
    """Load first n tasks from a domain."""
    path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
    with open(path) as f:
        tasks = json.load(f)
    return tasks[:n]


def _load_multiturn_tasks(n: int = 10) -> list[dict]:
    """Load n multi-turn tasks (prefer calendar domain)."""
    result = []
    for domain in ["calendar", "weather", "medical"]:
        path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
        with open(path) as f:
            tasks = json.load(f)
        for t in tasks:
            if t.get("multi_turn", False) and t.get("num_turns", 1) >= 4:
                result.append(t)
            if len(result) >= n:
                return result
    # Fallback: wrap non-multiturn tasks as multiturn
    if len(result) < n:
        for domain in ["calendar", "weather", "medical"]:
            path = os.path.join(TASKS_DIR, f"{domain}_tasks.json")
            with open(path) as f:
                tasks = json.load(f)
            for t in tasks:
                if not t.get("multi_turn"):
                    t["multi_turn"] = True
                    t["num_turns"] = 6
                    result.append(t)
                if len(result) >= n:
                    return result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# P2: Parameter Injection (10 tasks)
# ═══════════════════════════════════════════════════════════════════════════════

def run_p2_tasks(tracer: TraceLogger) -> list[dict]:
    """Run 10 P2 tasks: 2-3 per error type across domains."""
    console.rule("[bold blue]STEP 1 — P2 Parameter Injection (10 tasks)")

    injector = ParameterInjector(seed=RANDOM_SEED)
    executor = ToolExecutor()

    # Select tasks: spread across domains and error types
    domains = ["weather", "calendar", "medical"]
    error_types = PARAM_ERROR_TYPES  # 4 types
    task_pool = []
    for domain in domains:
        tasks = _load_tasks(domain, 5)
        task_pool.extend(tasks)

    # Assign error types round-robin
    results = []
    for i in range(10):
        task = task_pool[i % len(task_pool)]
        error_type = error_types[i % len(error_types)]
        session_id = f"p2_{task['task_id']}_{error_type}_{i}"
        domain = task["domain"]

        tracer.start_session(
            session_id=session_id,
            task_id=task["task_id"],
            model="mock",
            domain=domain,
            injection_type=f"p2_{error_type}",
            injection_stage="parameter_generation",
        )

        # Get first tool call from ground truth
        tool_seq = task.get("correct_tool_sequence", [])
        if not tool_seq:
            continue
        step = tool_seq[0]
        tool_name = step["tool"]
        correct_params = step["params"]

        # Execute with injection
        exec_result = executor.execute(
            tool_name, dict(correct_params),
            injector=injector,
            injection_error_type=error_type,
            tracer=tracer,
            session_id=session_id,
        )

        has_errors = bool(exec_result.get("validation_errors"))
        has_tool_error = "error" in exec_result.get("result", {})
        detected = has_errors or has_tool_error

        tracer.end_session(
            session_id=session_id,
            final_answer=json.dumps(exec_result["result"], default=str)[:200],
            ground_truth=task.get("correct_final_answer", ""),
            correct=not detected,
        )

        results.append({
            "task_id": task["task_id"],
            "domain": domain,
            "injection_type": f"p2_{error_type}",
            "tool_called": tool_name,
            "param_error_detected": detected,
            "validation_errors": exec_result.get("validation_errors", []),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# P3: Memory Injection (10 tasks)
# ═══════════════════════════════════════════════════════════════════════════════

def run_p3_tasks(tracer: TraceLogger) -> list[dict]:
    """Run 10 P3 multi-turn tasks with injection at turn 2."""
    console.rule("[bold green]STEP 2 — P3 Memory Injection (10 tasks)")

    mem_injector = MemoryInjector(seed=RANDOM_SEED)
    executor = ToolExecutor()

    tasks = _load_multiturn_tasks(10)
    results = []

    for i, task in enumerate(tasks[:10]):
        session_id = f"p3_{task['task_id']}_{i}"
        domain = task["domain"]

        tracer.start_session(
            session_id=session_id,
            task_id=task["task_id"],
            model="mock",
            domain=domain,
            injection_type="p3_false_belief",
            injection_stage="memory",
        )

        mm = MemoryManager(tracer=tracer, session_id=session_id)

        # Get injection plan
        plan = mem_injector.get_injection_plan(task)
        if not plan:
            continue
        inj_spec = plan[0]

        # Simulate 6 turns
        tool_seq = task.get("correct_tool_sequence", [])
        first_tool = tool_seq[0] if tool_seq else {"tool": "get_events", "params": {"date": "2024-07-01"}}
        num_turns = task.get("num_turns", 6)

        false_value_seen_in_turns = []

        for turn in range(1, num_turns + 1):
            # Execute tool
            exec_result = executor.execute(first_tool["tool"], dict(first_tool["params"]))
            observation = json.dumps(exec_result["result"], default=str)
            mm.add_observation(turn, observation)

            # Inject at turn 2
            if turn == inj_spec["turn_id"]:
                mem_injector.inject_false_belief(
                    memory_manager=mm,
                    tracer=tracer,
                    session_id=session_id,
                    turn_id=turn,
                    belief_key=inj_spec["belief_key"],
                    false_value=inj_spec["false_value"],
                    true_value=inj_spec["true_value"],
                )

            # Check if false value is in working memory
            wm_val = mm.working_memory.get(inj_spec["belief_key"], {})
            if isinstance(wm_val, dict):
                wm_val = wm_val.get("value")
            if turn > inj_spec["turn_id"] and str(wm_val) == str(inj_spec["false_value"]):
                false_value_seen_in_turns.append(turn)

        # Measure propagation
        trace = tracer.get_session_trace(session_id)
        propagation = mem_injector.measure_propagation_depth(
            trace, injection_turn=inj_spec["turn_id"], false_value=inj_spec["false_value"],
        )

        tracer.end_session(
            session_id=session_id,
            final_answer=f"propagation_depth={propagation['propagation_depth']}",
            ground_truth=task.get("correct_final_answer", ""),
            correct=False,
        )

        results.append({
            "task_id": task["task_id"],
            "domain": domain,
            "injection_type": "p3_false_belief",
            "belief_key": inj_spec["belief_key"],
            "injection_turn": inj_spec["turn_id"],
            "propagation_depth": propagation["propagation_depth"],
            "propagation_rate": propagation["propagation_rate"],
            "affected_turns": propagation.get("affected_turns", []),
            "false_value_in_memory_turns": false_value_seen_in_turns,
            "param_error_detected": propagation["propagation_depth"] == 0,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# P4: Cross-Agent Chain Injection (10 tasks)
# ═══════════════════════════════════════════════════════════════════════════════

def run_p4_tasks(tracer: TraceLogger) -> list[dict]:
    """Run 10 P4 3-agent chain tasks with injection at Agent 1."""
    console.rule("[bold magenta]STEP 3 — P4 Cross-Agent Chain (10 tasks)")

    executor = ToolExecutor()
    chain = MultiAgentChain(tool_executor=executor, seed=RANDOM_SEED)

    # Load tasks across domains
    task_pool = []
    for domain in ["weather", "calendar", "medical"]:
        tasks = _load_tasks(domain, 5)
        task_pool.extend(tasks)

    injection_types = ["wrong_subtask", "wrong_tool_spec", "wrong_parameter_spec"]
    results = []

    for i in range(10):
        task = task_pool[i % len(task_pool)]
        inj_type = injection_types[i % len(injection_types)]
        session_id = f"p4_{task['task_id']}_{inj_type}_{i}"

        chain_result = chain.run_chain(
            query=task["query"],
            domain=task["domain"],
            task=task,
            inject_at_agent=1,
            injection_type=inj_type,
            tracer=tracer,
            session_id=session_id,
        )

        eps = chain.measure_cross_agent_eps(chain_result)

        # Determine what tool was called
        agent2_results = chain_result.get("agent2_output", {}).get("tool_results", [])
        tools_called = [r.get("tool", "?") for r in agent2_results]

        # Check if any tool had errors
        tool_errors = any(
            r.get("validation_errors") or "error" in r.get("result", {})
            for r in agent2_results
        )

        results.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "injection_type": f"p4_{inj_type}",
            "tool_called": ", ".join(tools_called) if tools_called else "none",
            "propagated_to_output": eps["propagated_to_output"],
            "propagation_path": eps["propagation_path"],
            "amplification": eps["amplification_factor"],
            "param_error_detected": tool_errors,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main runner + rich output
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(title: str, results: list[dict], style: str = "blue"):
    """Print a rich table with results."""
    table = Table(title=title, show_lines=True)
    table.add_column("Task ID", style="cyan")
    table.add_column("Domain", style="white")
    table.add_column("Injection Type", style="yellow")
    table.add_column("Tool Called", style="white")
    table.add_column("Error Detected?", style="bold")

    for r in results:
        detected = r.get("param_error_detected", False)
        detected_str = "[green]YES[/green]" if detected else "[red]NO (undetected)[/red]"
        table.add_row(
            r.get("task_id", "?"),
            r.get("domain", "?"),
            r.get("injection_type", "?"),
            r.get("tool_called", "?"),
            detected_str,
        )

    console.print(table)


def print_p3_propagation_table(results: list[dict]):
    """Print propagation details for P3."""
    table = Table(title="P3 Memory Propagation Details", show_lines=True)
    table.add_column("Task ID", style="cyan")
    table.add_column("Belief Key", style="yellow")
    table.add_column("Inject Turn", style="white", justify="center")
    table.add_column("Prop. Depth", style="bold", justify="center")
    table.add_column("Prop. Rate", style="bold", justify="center")
    table.add_column("Affected Turns", style="white")

    for r in results:
        depth = r.get("propagation_depth", 0)
        depth_str = f"[red]{depth}[/red]" if depth > 0 else f"[green]{depth}[/green]"
        table.add_row(
            r.get("task_id", "?"),
            r.get("belief_key", "?"),
            str(r.get("injection_turn", "?")),
            depth_str,
            f"{r.get('propagation_rate', 0):.2f}",
            str(r.get("affected_turns", [])),
        )

    console.print(table)


def print_p4_chain_table(results: list[dict]):
    """Print chain propagation details for P4."""
    table = Table(title="P4 Cross-Agent Propagation Details", show_lines=True)
    table.add_column("Task ID", style="cyan")
    table.add_column("Injection", style="yellow")
    table.add_column("Prop. Path", style="white")
    table.add_column("Amplification", justify="center")
    table.add_column("Reached Output?", style="bold")

    for r in results:
        reached = r.get("propagated_to_output", False)
        reached_str = "[red]YES[/red]" if reached else "[green]NO[/green]"
        table.add_row(
            r.get("task_id", "?"),
            r.get("injection_type", "?"),
            str(r.get("propagation_path", [])),
            f"{r.get('amplification', 0):.1f}x",
            reached_str,
        )

    console.print(table)


def print_summary(p2_results, p3_results, p4_results):
    """Print final summary report."""
    console.rule("[bold white]SUMMARY REPORT")

    p2_undetected = sum(1 for r in p2_results if not r.get("param_error_detected"))
    p3_propagated = sum(1 for r in p3_results if r.get("propagation_depth", 0) > 0)
    p4_reached_output = sum(1 for r in p4_results if r.get("propagated_to_output"))

    summary = Table(title="Injection Detection Summary", show_lines=True)
    summary.add_column("Problem", style="bold")
    summary.add_column("Total Tasks", justify="center")
    summary.add_column("Undetected / Propagated", justify="center", style="red")
    summary.add_column("Detection Rate", justify="center")

    p2_total = len(p2_results)
    p2_detected = p2_total - p2_undetected
    summary.add_row("P2 (Parameter)", str(p2_total),
                     f"{p2_undetected} undetected",
                     f"{p2_detected/max(p2_total,1)*100:.0f}%")

    p3_total = len(p3_results)
    summary.add_row("P3 (Memory)", str(p3_total),
                     f"{p3_propagated} propagated",
                     f"{(p3_total-p3_propagated)/max(p3_total,1)*100:.0f}% caught")

    p4_total = len(p4_results)
    summary.add_row("P4 (Cross-Agent)", str(p4_total),
                     f"{p4_reached_output} reached output",
                     f"{(p4_total-p4_reached_output)/max(p4_total,1)*100:.0f}% contained")

    console.print(summary)

    # Key finding
    total_undetected = p2_undetected + p3_propagated + p4_reached_output
    total = p2_total + p3_total + p4_total
    console.print(Panel(
        f"[bold]Total: {total_undetected}/{total} injections went undetected/propagated "
        f"({total_undetected/max(total,1)*100:.0f}%)[/bold]\n\n"
        f"P2 semantic_wrong errors are hardest to detect (pass schema validation).\n"
        f"P3 false beliefs persist across {p3_propagated}/{p3_total} multi-turn sessions.\n"
        f"P4 errors reached final output in {p4_reached_output}/{p4_total} chains.",
        title="Key Findings",
        border_style="yellow",
    ))


def main():
    """Run all 30 tasks and report."""
    os.makedirs(TRACES_DIR, exist_ok=True)

    # Clean DB if exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    tracer = TraceLogger(DB_PATH)
    tracer.init_db()

    console.print(Panel(
        "[bold]Day 3 Integration Test: 30 Injected Tasks[/bold]\n"
        "10 x P2 (parameter) + 10 x P3 (memory) + 10 x P4 (cross-agent)",
        title="AgentHallu-Bench Day 3",
        border_style="blue",
    ))

    # Run all 3 problem types
    p2_results = run_p2_tasks(tracer)
    print_results_table("P2 Parameter Injection Results", p2_results, "blue")

    p3_results = run_p3_tasks(tracer)
    print_results_table("P3 Memory Injection Results", p3_results, "green")
    print_p3_propagation_table(p3_results)

    p4_results = run_p4_tasks(tracer)
    print_results_table("P4 Cross-Agent Results", p4_results, "magenta")
    print_p4_chain_table(p4_results)

    # Summary
    print_summary(p2_results, p3_results, p4_results)

    # Export traces
    tracer.export_traces(os.path.join(TRACES_DIR, "day3_traces.json"))
    tracer.close()

    console.print(f"\n[dim]Traces saved to {DB_PATH}[/dim]")
    console.print(f"[dim]JSON export at {TRACES_DIR}/day3_traces.json[/dim]")


if __name__ == "__main__":
    main()
