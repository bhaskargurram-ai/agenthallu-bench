"""P3: Multi-turn memory error injector.

Plants false beliefs into agent working memory during multi-turn sessions.
Operates at turn boundary — after turn N observation, before turn N+1 reasoning.
Simulates tool execution feedback misinterpretation (Cascade Pattern 2).
"""

import json
import logging
import random
from typing import Any, Optional

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


# ── False belief templates per domain ────────────────────────────────────────

_FALSE_BELIEFS = {
    "calendar": [
        {"key": "event_id", "true_pattern": "EVT_{id}", "false_value": "EVT_9999",
         "description": "Wrong event ID stored after create_event"},
        {"key": "last_created_event", "true_pattern": "EVT_{id}", "false_value": "EVT_9999",
         "description": "Wrong event ID in working memory"},
        {"key": "date", "true_pattern": "{date}", "false_value": "2025-12-25",
         "description": "Wrong date stored from observation"},
    ],
    "weather": [
        {"key": "temperature", "true_pattern": "{temp}", "false_value": -99,
         "description": "Drastically wrong temperature stored"},
        {"key": "city", "true_pattern": "{city}", "false_value": "Atlantis",
         "description": "Non-existent city stored in memory"},
        {"key": "condition", "true_pattern": "{cond}", "false_value": "tornado",
         "description": "Wrong weather condition stored"},
    ],
    "medical": [
        {"key": "patient_id", "true_pattern": "{pid}", "false_value": "P999",
         "description": "Wrong patient ID stored"},
        {"key": "interaction", "true_pattern": "{interaction}", "false_value": True,
         "description": "False drug interaction stored"},
        {"key": "appointment_id", "true_pattern": "{appt}", "false_value": "APPT_00000",
         "description": "Wrong appointment ID stored"},
    ],
}


class MemoryInjector:
    """Plants false beliefs into agent working memory during multi-turn sessions."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        logger.info("MemoryInjector initialized with seed=%d", seed)

    def inject_false_belief(
        self,
        memory_manager: Any,
        tracer: Optional[Any],
        session_id: str,
        turn_id: int,
        belief_key: str,
        false_value: Any,
        true_value: Any,
    ) -> dict:
        """Write false_value into memory_manager.working_memory[belief_key].

        Logs: true_value, false_value, turn_id, downstream turns affected.
        Returns injection log entry.
        """
        # Use memory_manager's built-in injection method
        log_entry = memory_manager.inject_false_belief(belief_key, false_value, turn_id)

        # Log to tracer
        if tracer and session_id:
            tracer.log_injection(
                session_id=session_id,
                injection_type="p3_false_belief",
                target_stage="memory",
                original=true_value,
                injected=false_value,
                turn_id=turn_id,
            )

        logger.info(
            "P3 injection at turn %d: %s = %r (true: %r)",
            turn_id, belief_key, false_value, true_value,
        )
        return {
            "belief_key": belief_key,
            "true_value": true_value,
            "false_value": false_value,
            "turn_id": turn_id,
            "log_entry": log_entry,
        }

    def get_injection_plan(self, task: dict) -> list[dict]:
        """For a multi-turn task, decide what/when to inject.

        Always inject at turn 2 so the error propagates for remaining turns.
        Returns list of injection specs.
        """
        domain = task.get("domain", "weather")
        num_turns = task.get("num_turns", 6)
        injection_turn = 2  # Always turn 2

        # Pick a false belief template for the domain
        templates = _FALSE_BELIEFS.get(domain, _FALSE_BELIEFS["weather"])
        template = self.rng.choice(templates)

        # Determine true value from task ground truth
        true_value = self._extract_true_value(task, template["key"])

        plan = [{
            "turn_id": injection_turn,
            "belief_key": template["key"],
            "false_value": template["false_value"],
            "true_value": true_value,
            "description": template["description"],
            "expected_propagation_turns": list(range(injection_turn + 1, num_turns + 1)),
        }]

        logger.info(
            "P3 injection plan for %s: inject '%s' at turn %d, expect propagation to turns %s",
            task.get("task_id", "?"), template["key"], injection_turn,
            plan[0]["expected_propagation_turns"],
        )
        return plan

    def _extract_true_value(self, task: dict, key: str) -> Any:
        """Extract the true value from task ground truth for the injection key."""
        tool_seq = task.get("correct_tool_sequence", [])
        for step in tool_seq:
            params = step.get("params", {})
            if key in params:
                return params[key]

        # Domain-specific defaults
        defaults = {
            "event_id": "EVT_0001",
            "last_created_event": "EVT_0001",
            "temperature": 22,
            "city": "Paris",
            "condition": "sunny",
            "patient_id": "P001",
            "interaction": False,
            "appointment_id": "APPT_10001",
            "date": "2024-07-01",
        }
        return defaults.get(key, "UNKNOWN")

    def measure_propagation_depth(
        self,
        trace: dict,
        injection_turn: int,
        false_value: Any,
    ) -> dict:
        """Count how many subsequent turns were affected by the false belief.

        Checks tool_params in each step after injection for presence of false value.
        Returns propagation depth score (0 = caught immediately, N = all remaining turns).
        """
        steps = trace.get("steps", [])
        false_str = str(false_value)

        affected_turns: set[int] = set()
        total_turns_after = set()

        for step in steps:
            turn = step.get("turn_id", 0)
            if turn <= injection_turn:
                continue

            total_turns_after.add(turn)

            # Check tool_params_raw for false value
            params_raw = step.get("tool_params_raw", "")
            if params_raw and false_str in str(params_raw):
                affected_turns.add(turn)

            # Check content for false value
            content = step.get("content", "")
            if false_str in str(content):
                affected_turns.add(turn)

            # Check memory_state for false value
            mem_state = step.get("memory_state", "")
            if mem_state and false_str in str(mem_state):
                affected_turns.add(turn)

        propagation_depth = len(affected_turns)
        total_after = len(total_turns_after) if total_turns_after else 1

        result = {
            "injection_turn": injection_turn,
            "false_value": false_value,
            "affected_turns": sorted(affected_turns),
            "propagation_depth": propagation_depth,
            "propagation_rate": propagation_depth / max(total_after, 1),
            "total_turns_after_injection": total_after,
        }

        logger.info(
            "P3 propagation: depth=%d, rate=%.2f, affected=%s",
            result["propagation_depth"], result["propagation_rate"],
            result["affected_turns"],
        )
        return result
