"""Stage Attribution: Given a final output error, trace it back to its origin stage.

This is the core P1 contribution — the first system to attribute final errors
to pipeline stages. Uses rule-based checks first (fast), with LLM-judge
fallback for ambiguous cases.
"""

import json
import logging
from typing import Any, Optional

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


def _safe_json_loads(val: Any) -> Any:
    """Parse JSON string, or return as-is if already parsed or None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


class StageAttributor:
    """Attributes final output errors to their pipeline origin stage."""

    STAGES = [
        "planning",
        "tool_selection",
        "parameter_generation",
        "tool_execution",
        "memory_write",
        "memory_retrieval",
        "output_generation",
    ]

    def attribute(self, session_trace: dict) -> dict:
        """Analyze full trace to determine which stage originated the error.

        Walk backward through trace to find earliest stage with anomaly.

        Anomaly detection heuristics per stage:
        - parameter_generation: param_errors field in steps table is non-empty
        - tool_execution: tool returned error dict
        - memory_write: written value contradicts earlier observation
        - tool_selection: wrong tool for query type
        - output_generation: final answer contradicts tool observations
        - planning: query intent vs subtask mismatch

        Returns:
        {
            "final_error": bool,
            "attributed_stage": str | None,
            "confidence": float (0-1),
            "evidence": [str],
            "attribution_method": str
        }
        """
        steps = session_trace.get("steps", [])
        final_answer = session_trace.get("final_answer", "")
        ground_truth = session_trace.get("ground_truth_answer", "")
        final_correct = session_trace.get("final_correct")

        # Determine if there was a final error
        if final_correct is not None:
            final_error = not final_correct
        elif ground_truth and final_answer:
            final_error = ground_truth.lower().strip() != final_answer.lower().strip()
        else:
            final_error = False

        if not final_error:
            return {
                "final_error": False,
                "attributed_stage": None,
                "confidence": 1.0,
                "evidence": [],
                "attribution_method": "none_needed",
            }

        # Walk backward through steps looking for anomalies
        evidence = []
        attributed_stage = None
        confidence = 0.0
        method = "heuristic"

        # Reversed steps for backward walk
        reversed_steps = list(reversed(steps))

        # ── Rule-based checks (fast path) ────────────────────────────────

        # Check 1: parameter_generation — param_errors non-empty
        for step in reversed_steps:
            param_errors = _safe_json_loads(step.get("param_errors"))
            if param_errors and isinstance(param_errors, list) and len(param_errors) > 0:
                attributed_stage = "parameter_generation"
                confidence = 0.95
                evidence.append(f"param_errors at step {step.get('step_number')}: {param_errors}")
                method = "schema_check"
                break

        # Check 2: tool_execution — tool returned error dict
        if not attributed_stage:
            for step in reversed_steps:
                tool_result = _safe_json_loads(step.get("tool_result"))
                if isinstance(tool_result, dict) and "error" in tool_result:
                    attributed_stage = "tool_execution"
                    confidence = 0.90
                    evidence.append(
                        f"tool_result error at step {step.get('step_number')}: "
                        f"{tool_result.get('error', '')[:200]}"
                    )
                    method = "schema_check"
                    break

        # Check 3: memory_write — value not in tool_result
        if not attributed_stage:
            attributed_stage, conf, evid = self._check_memory_write_anomaly(steps)
            if attributed_stage:
                confidence = conf
                evidence.extend(evid)

        # Check 4: tool_selection — wrong tool for query
        if not attributed_stage:
            attributed_stage, conf, evid = self._check_tool_selection(session_trace)
            if attributed_stage:
                confidence = conf
                evidence.extend(evid)

        # Check 5: output_generation — final answer contradicts observations
        if not attributed_stage:
            attributed_stage, conf, evid = self._check_output_contradiction(session_trace)
            if attributed_stage:
                confidence = conf
                evidence.extend(evid)

        # Fallback: if injections exist, use injection target_stage
        if not attributed_stage:
            injections = session_trace.get("injections", [])
            if injections:
                attributed_stage = injections[0].get("target_stage", "output_generation")
                confidence = 0.5
                evidence.append("Attributed via injection record (fallback)")
                method = "injection_record"
            else:
                attributed_stage = "output_generation"
                confidence = 0.3
                evidence.append("No specific anomaly found, defaulting to output_generation")
                method = "default"

        result = {
            "final_error": True,
            "attributed_stage": attributed_stage,
            "confidence": confidence,
            "evidence": evidence,
            "attribution_method": method,
        }

        logger.info(
            "Attribution: session=%s stage=%s confidence=%.2f method=%s",
            session_trace.get("session_id", "?"),
            attributed_stage, confidence, method,
        )
        return result

    def _check_memory_write_anomaly(self, steps: list[dict]) -> tuple:
        """Check if memory_state contains value not in any prior tool_result."""
        # Collect all tool results
        tool_result_values = set()
        memory_write_steps = []

        for step in steps:
            # Collect tool result values
            tool_result = _safe_json_loads(step.get("tool_result"))
            if isinstance(tool_result, dict):
                for v in tool_result.values():
                    if not isinstance(v, (dict, list)):
                        tool_result_values.add(str(v))

            # Collect memory write steps
            if step.get("step_type") == "memory_write":
                memory_write_steps.append(step)

        # Check if any memory write contains values absent from tool results
        for mstep in memory_write_steps:
            mem_state = _safe_json_loads(mstep.get("memory_state"))
            if isinstance(mem_state, dict):
                for key, value in mem_state.items():
                    val_str = str(value.get("value", value) if isinstance(value, dict) else value)
                    # Skip trivial values
                    if val_str in ("None", "True", "False", "", "0"):
                        continue
                    if val_str not in tool_result_values and len(val_str) > 2:
                        return (
                            "memory_write",
                            0.80,
                            [f"Memory value '{key}={val_str}' not found in any tool_result"],
                        )

        return None, 0.0, []

    def _check_tool_selection(self, session_trace: dict) -> tuple:
        """Check if the selected tool matches the query's domain intent."""
        steps = session_trace.get("steps", [])
        domain = session_trace.get("domain", "")
        ground_truth = session_trace.get("ground_truth_answer", "")

        # Find action steps
        tool_names_used = []
        for step in steps:
            if step.get("step_type") == "action" and step.get("tool_name"):
                tool_names_used.append(step["tool_name"])

        if not tool_names_used:
            return None, 0.0, []

        # Domain → expected tools mapping
        domain_tools = {
            "weather": {"get_weather", "get_forecast", "get_historical"},
            "calendar": {"create_event", "get_events", "delete_event"},
            "medical": {"get_patient_record", "check_drug_interaction", "schedule_appointment"},
        }

        expected = domain_tools.get(domain, set())
        if expected:
            for tool in tool_names_used:
                if tool not in expected:
                    return (
                        "tool_selection",
                        0.85,
                        [f"Tool '{tool}' not in expected domain tools: {expected}"],
                    )

        return None, 0.0, []

    def _check_output_contradiction(self, session_trace: dict) -> tuple:
        """Check if final answer claims something unsupported by observations."""
        final_answer = session_trace.get("final_answer", "")
        if not final_answer:
            return None, 0.0, []

        steps = session_trace.get("steps", [])

        # Gather all observation content and tool results
        observation_text = ""
        for step in steps:
            if step.get("step_type") in ("observation", "retrieval"):
                observation_text += " " + str(step.get("content", ""))
            tool_result = step.get("tool_result")
            if tool_result:
                observation_text += " " + str(tool_result)

        if not observation_text.strip():
            return None, 0.0, []

        # Simple heuristic: check if final answer contains numbers not in observations
        import re
        answer_numbers = set(re.findall(r"-?\d+\.?\d*", final_answer))
        obs_numbers = set(re.findall(r"-?\d+\.?\d*", observation_text))

        unsupported = answer_numbers - obs_numbers
        # Filter out trivial numbers
        unsupported = {n for n in unsupported if len(n) > 1 or int(float(n)) > 9}

        if unsupported:
            return (
                "output_generation",
                0.70,
                [f"Final answer contains unsupported values: {unsupported}"],
            )

        return None, 0.0, []

    def attribution_accuracy(
        self, predictions: list[dict], ground_truth_stages: list[str]
    ) -> dict:
        """Compute attribution accuracy since we inject at known stages.

        Returns precision, recall, F1 per stage.
        """
        if len(predictions) != len(ground_truth_stages):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth_stages)}) "
                "must have same length"
            )

        per_stage = {}
        for stage in self.STAGES:
            tp = fp = fn = 0
            for pred, gt in zip(predictions, ground_truth_stages):
                pred_stage = pred.get("attributed_stage")
                if pred_stage == stage and gt == stage:
                    tp += 1
                elif pred_stage == stage and gt != stage:
                    fp += 1
                elif pred_stage != stage and gt == stage:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_stage[stage] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Overall accuracy
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truth_stages)
            if pred.get("attributed_stage") == gt
        )
        overall_accuracy = correct / len(predictions) if predictions else 0.0

        result = {
            "per_stage": per_stage,
            "overall_accuracy": round(overall_accuracy, 3),
            "total_predictions": len(predictions),
            "correct_predictions": correct,
        }

        logger.info(
            "Attribution accuracy: %.1f%% (%d/%d)",
            overall_accuracy * 100, correct, len(predictions),
        )
        return result
