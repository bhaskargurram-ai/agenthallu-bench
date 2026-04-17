"""P4: Cross-agent error propagation injector.

Runs a 3-agent chain where:
- Agent 1 (Planner): decomposes goal, creates sub-task plan
- Agent 2 (Executor): executes sub-task using tools + RAG
- Agent 3 (Synthesizer): combines results into final answer

Injects hallucination into Agent 1's output and measures propagation.
"""

import json
import logging
import random
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Optional

from config import RANDOM_SEED, MODELS

logger = logging.getLogger(__name__)


# ── Agent role prompts ───────────────────────────────────────────────────────

PLANNER_PROMPT = """You are Agent 1 (Planner). Given a user query, decompose it into a step-by-step plan.
Output format:
Plan:
1. Use tool [tool_name] with params: {{param_dict}}
2. Use tool [tool_name] with params: {{param_dict}}
...
Goal: [restate the goal clearly]
"""

EXECUTOR_PROMPT = """You are Agent 2 (Executor). You receive a plan from Agent 1.
Execute each step using the available tools.
Follow this format:
Thought: [reasoning]
Action: [tool_name]
Action Input: [JSON params]
After executing all steps, output:
Result: [summary of all tool results]
"""

SYNTHESIZER_PROMPT = """You are Agent 3 (Synthesizer). You receive results from Agent 2.
Combine all results into a clear, concise final answer.
Output format:
Final Answer: [your synthesized answer]
"""


# ── Injection templates ──────────────────────────────────────────────────────

_GOAL_SWAPS = {
    "weather": {"from": "New York", "to": "Los Angeles"},
    "calendar": {"from": "Team Meeting", "to": "Birthday Party"},
    "medical": {"from": "P001", "to": "P003"},
}

_WRONG_TOOLS = {
    "get_weather": "get_historical",
    "get_forecast": "get_weather",
    "get_historical": "get_forecast",
    "create_event": "get_events",
    "get_events": "delete_event",
    "delete_event": "create_event",
    "get_patient_record": "schedule_appointment",
    "check_drug_interaction": "get_patient_record",
    "schedule_appointment": "check_drug_interaction",
}


class MultiAgentChain:
    """Runs a 3-agent chain for P4 cross-agent propagation experiments."""

    def __init__(self, tool_executor: Any, seed: int = RANDOM_SEED):
        self.tool_executor = tool_executor
        self.rng = random.Random(seed)
        logger.info("MultiAgentChain initialized with seed=%d", seed)

    def run_chain(
        self,
        query: str,
        domain: str,
        task: dict,
        inject_at_agent: Optional[int] = None,
        injection_type: Optional[str] = None,
        tracer: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Run 3-agent chain. Optionally inject at agent 1.

        Returns dict with outputs from all 3 agents and propagation analysis.
        """
        session_id = session_id or f"p4_{uuid.uuid4().hex[:8]}"

        if tracer:
            tracer.start_session(
                session_id=session_id,
                task_id=task.get("task_id", "unknown"),
                model="mock_chain",
                domain=domain,
                injection_type=f"p4_{injection_type}" if injection_type else None,
                injection_stage="planning" if inject_at_agent == 1 else None,
            )

        # ── Agent 1: Planner ─────────────────────────────────────────────
        agent1_output = self._run_planner(query, domain, task)

        if tracer:
            step_num = tracer.get_step_count(session_id) + 1
            tracer.log_step(
                session_id, turn_id=0, step_number=step_num,
                step_type="thought",
                content=f"Agent1 (Planner) output: {agent1_output}",
            )

        # ── Inject at Agent 1 if requested ───────────────────────────────
        original_agent1 = agent1_output
        if inject_at_agent == 1 and injection_type:
            agent1_output = self.inject_agent1_hallucination(
                agent1_output, injection_type, domain, task,
            )
            if tracer:
                tracer.log_injection(
                    session_id=session_id,
                    injection_type=f"p4_{injection_type}",
                    target_stage="planning",
                    original=original_agent1,
                    injected=agent1_output,
                    turn_id=0,
                )

        # ── Agent 2: Executor ────────────────────────────────────────────
        agent2_output = self._run_executor(agent1_output, domain, task, tracer, session_id)

        if tracer:
            step_num = tracer.get_step_count(session_id) + 1
            tracer.log_step(
                session_id, turn_id=1, step_number=step_num,
                step_type="observation",
                content=f"Agent2 (Executor) output: {json.dumps(agent2_output, default=str)[:500]}",
            )

        # ── Agent 3: Synthesizer ─────────────────────────────────────────
        agent3_output = self._run_synthesizer(query, agent2_output, domain)

        if tracer:
            step_num = tracer.get_step_count(session_id) + 1
            tracer.log_step(
                session_id, turn_id=2, step_number=step_num,
                step_type="final",
                content=f"Agent3 (Synthesizer) output: {agent3_output}",
            )
            tracer.end_session(
                session_id=session_id,
                final_answer=agent3_output,
                ground_truth=task.get("correct_final_answer", ""),
                correct=self._check_correctness(
                    agent3_output, task, injection_type,
                ),
            )

        # ── Measure propagation ──────────────────────────────────────────
        propagation_path = self._trace_propagation(
            original_agent1, agent1_output, agent2_output, agent3_output,
            injection_type,
        )

        return {
            "session_id": session_id,
            "agent1_output": agent1_output,
            "agent2_output": agent2_output,
            "agent3_output": agent3_output,
            "final_answer": agent3_output,
            "original_agent1": original_agent1,
            "injection_type": injection_type,
            "error_detected_at_agent": propagation_path.get("first_detectable_at"),
            "propagation_path": propagation_path.get("path", []),
        }

    def _run_planner(self, query: str, domain: str, task: dict) -> str:
        """Simulate Agent 1 (Planner) — produces a deterministic plan from task."""
        tool_seq = task.get("correct_tool_sequence", [])
        plan_lines = [f"Goal: {query}"]
        for i, step in enumerate(tool_seq, 1):
            plan_lines.append(
                f"{i}. Use tool [{step['tool']}] with params: {json.dumps(step['params'])}"
            )
        return "\n".join(plan_lines)

    def _run_executor(
        self, plan: str, domain: str, task: dict,
        tracer: Optional[Any], session_id: Optional[str],
    ) -> dict:
        """Simulate Agent 2 (Executor) — parse plan and execute tools."""
        results = []

        # Parse tool calls from plan text
        tool_calls = re.findall(
            r"Use tool \[(\w+)\] with params: ({.*?})(?:\n|$)", plan, re.DOTALL,
        )

        for tool_name, params_str in tool_calls:
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                params = {}

            exec_result = self.tool_executor.execute(tool_name, params)
            results.append({
                "tool": tool_name,
                "params": params,
                "result": exec_result["result"],
                "validation_errors": exec_result.get("validation_errors", []),
            })

            # Log executor step
            if tracer and session_id:
                step_num = tracer.get_step_count(session_id) + 1
                tracer.log_step(
                    session_id, turn_id=1, step_number=step_num,
                    step_type="action",
                    content=f"Executor calls {tool_name}",
                    tool_name=tool_name,
                    tool_params_raw=params,
                    tool_params_validated=exec_result.get("validated_params"),
                    tool_result=exec_result["result"],
                    param_errors=exec_result.get("validation_errors"),
                )

        return {"plan_received": plan, "tool_results": results}

    def _run_synthesizer(self, query: str, executor_output: dict, domain: str) -> str:
        """Simulate Agent 3 (Synthesizer) — combine results into answer."""
        results = executor_output.get("tool_results", [])
        parts = [f"Answer to: {query}"]
        for r in results:
            tool = r.get("tool", "unknown")
            result = r.get("result", {})
            if "error" in result:
                parts.append(f"Tool {tool} returned error: {result['error']}")
            else:
                parts.append(f"Tool {tool} returned: {json.dumps(result, default=str)[:200]}")
        return " | ".join(parts)

    def inject_agent1_hallucination(
        self, agent1_output: str, injection_type: str,
        domain: str, task: dict,
    ) -> str:
        """Modify Agent 1's plan before passing to Agent 2.

        injection_type options:
        - wrong_subtask: change the goal noun
        - wrong_tool_spec: specify wrong tool in plan
        - wrong_parameter_spec: specify wrong param value
        """
        if injection_type == "wrong_subtask":
            return self._inject_wrong_subtask(agent1_output, domain)
        elif injection_type == "wrong_tool_spec":
            return self._inject_wrong_tool(agent1_output)
        elif injection_type == "wrong_parameter_spec":
            return self._inject_wrong_param(agent1_output, domain, task)
        else:
            logger.warning("Unknown P4 injection type: %s", injection_type)
            return agent1_output

    def _inject_wrong_subtask(self, plan: str, domain: str) -> str:
        """Change the goal noun in the plan."""
        swap = _GOAL_SWAPS.get(domain, {"from": "target", "to": "wrong_target"})
        modified = plan.replace(swap["from"], swap["to"])
        if modified == plan:
            # No exact match — do a generic substitution in the Goal line
            modified = re.sub(
                r"(Goal:.*)",
                lambda m: m.group(1) + " [MODIFIED: wrong subtask]",
                plan,
            )
        logger.info("P4 wrong_subtask: swapped '%s' → '%s'", swap["from"], swap["to"])
        return modified

    def _inject_wrong_tool(self, plan: str) -> str:
        """Replace tool names in the plan with wrong tools."""
        modified = plan
        for correct, wrong in _WRONG_TOOLS.items():
            if correct in modified:
                modified = modified.replace(correct, wrong, 1)
                logger.info("P4 wrong_tool: %s → %s", correct, wrong)
                break
        return modified

    def _inject_wrong_param(self, plan: str, domain: str, task: dict) -> str:
        """Inject wrong parameter value into the plan."""
        tool_seq = task.get("correct_tool_sequence", [])
        if not tool_seq:
            return plan

        step = tool_seq[0]
        params = deepcopy(step["params"])

        # Corrupt the first parameter
        for key, value in params.items():
            if isinstance(value, str):
                if "date" in key:
                    params[key] = "2099-01-01"
                elif "city" in key:
                    params[key] = "Atlantis"
                elif "patient_id" in key:
                    params[key] = "P999"
                elif "event_id" in key:
                    params[key] = "EVT_WRONG"
                else:
                    params[key] = value + "_WRONG"
                break
            elif isinstance(value, int):
                params[key] = value * 100
                break

        # Replace the params in the plan
        original_params_str = json.dumps(step["params"])
        new_params_str = json.dumps(params)
        modified = plan.replace(original_params_str, new_params_str)
        logger.info("P4 wrong_param: %s → %s", original_params_str[:80], new_params_str[:80])
        return modified

    def _trace_propagation(
        self, original_plan: str, injected_plan: str,
        agent2_output: dict, agent3_output: str,
        injection_type: Optional[str],
    ) -> dict:
        """Track where the injected error appears in the chain."""
        if not injection_type:
            return {"path": [], "first_detectable_at": None, "propagated_to_output": False}

        # Detect what changed
        diff_tokens = self._find_diff_tokens(original_plan, injected_plan)
        path = [1]  # Always starts at agent 1
        agent2_str = json.dumps(agent2_output, default=str)

        agent2_affected = any(token in agent2_str for token in diff_tokens)
        if agent2_affected:
            path.append(2)

        agent3_affected = any(token in agent3_output for token in diff_tokens)
        if agent3_affected:
            path.append(3)

        # Also check for errors in agent2 tool results
        tool_results = agent2_output.get("tool_results", [])
        has_tool_errors = any(
            r.get("validation_errors") or "error" in r.get("result", {})
            for r in tool_results
        )

        first_detectable = None
        if has_tool_errors:
            first_detectable = 2
        elif agent3_affected and not agent2_affected:
            first_detectable = 3
        elif agent2_affected:
            first_detectable = 2

        return {
            "path": path,
            "first_detectable_at": first_detectable,
            "propagated_to_output": agent3_affected,
            "diff_tokens": diff_tokens[:5],
            "agent2_affected": agent2_affected,
            "agent3_affected": agent3_affected,
            "has_tool_errors": has_tool_errors,
        }

    def _find_diff_tokens(self, original: str, modified: str) -> list[str]:
        """Find tokens that differ between original and modified text."""
        orig_words = set(original.split())
        mod_words = set(modified.split())
        new_tokens = mod_words - orig_words
        return list(new_tokens)[:10]

    def _check_correctness(
        self, final_answer: str, task: dict, injection_type: Optional[str],
    ) -> bool:
        """Check if the final answer matches ground truth."""
        if injection_type:
            return False  # Injected tasks are assumed incorrect
        gt = task.get("correct_final_answer", "")
        return gt.lower() in final_answer.lower() if gt else False

    def measure_cross_agent_eps(self, chain_result: dict) -> dict:
        """Measure error propagation across agents.

        Returns dict with injection point, detection point, propagation info.
        """
        path = chain_result.get("propagation_path", [])
        injection_type = chain_result.get("injection_type")

        propagated_to_output = 3 in path
        amplification = len(path) / 1.0 if path else 0.0

        first_detectable = chain_result.get("error_detected_at_agent")

        return {
            "injected_at": 1,
            "first_detectable_at": first_detectable,
            "propagated_to_output": propagated_to_output,
            "amplification_factor": amplification,
            "propagation_path": path,
            "injection_type": injection_type,
        }
