"""Cascade pattern detector — detects the 3 cascade patterns from survey Section 4.

Pattern 1: Planning → Tool Use
Pattern 2: Tool Use → Memory
Pattern 3: Memory → Output
"""

import json
import logging
import re
from typing import Any

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


# Domain → expected tool mapping for pattern 1
_DOMAIN_TOOLS = {
    "weather": {"get_weather", "get_forecast", "get_historical"},
    "calendar": {"create_event", "get_events", "delete_event"},
    "medical": {"get_patient_record", "check_drug_interaction", "schedule_appointment"},
}

# Query keyword → likely correct tool mapping
_KEYWORD_TOOL_MAP = {
    "weather": "get_weather",
    "forecast": "get_forecast",
    "historical": "get_historical",
    "create": "create_event",
    "schedule": "schedule_appointment",
    "delete": "delete_event",
    "events": "get_events",
    "patient": "get_patient_record",
    "record": "get_patient_record",
    "drug": "check_drug_interaction",
    "interaction": "check_drug_interaction",
    "appointment": "schedule_appointment",
}


class CascadeDetector:
    """Detects 3 cascade hallucination patterns in agent traces."""

    def detect_pattern_1(self, trace: dict) -> dict:
        """Planning → Tool Use cascade.

        Signal: goal_misinterpretation in planning step leads to
                wrong tool selection or wrong parameters in next action.
        Detection: compare planned tool vs selected tool vs correct tool.
        """
        steps = trace.get("steps", [])
        domain = trace.get("domain", "")
        task_id = trace.get("task_id", "")

        detected = False
        confidence = 0.0
        evidence_parts = []

        # Find the first thought (planning) and first action (tool use)
        planning_step = None
        action_step = None
        for step in steps:
            if step.get("step_type") == "thought" and planning_step is None:
                planning_step = step
            if step.get("step_type") == "action" and action_step is None:
                action_step = step
            if planning_step and action_step:
                break

        if not planning_step or not action_step:
            return {"detected": False, "confidence": 0.0, "evidence": "No planning+action steps found"}

        planning_content = (planning_step.get("content") or "").lower()
        selected_tool = action_step.get("tool_name", "")

        # Infer intended tool from planning content keywords
        intended_tool = None
        for keyword, tool in _KEYWORD_TOOL_MAP.items():
            if keyword in planning_content:
                intended_tool = tool
                break

        # Check: does the selected tool match the inferred intent?
        if intended_tool and selected_tool and intended_tool != selected_tool:
            detected = True
            confidence = 0.80
            evidence_parts.append(
                f"Planning implies '{intended_tool}' but selected '{selected_tool}'"
            )

        # Check: is the selected tool from the wrong domain?
        expected_tools = _DOMAIN_TOOLS.get(domain, set())
        if expected_tools and selected_tool and selected_tool not in expected_tools:
            detected = True
            confidence = max(confidence, 0.90)
            evidence_parts.append(
                f"Tool '{selected_tool}' not in domain '{domain}' tools: {expected_tools}"
            )

        result = {
            "detected": detected,
            "confidence": round(confidence, 2),
            "evidence": "; ".join(evidence_parts) if evidence_parts else "No cascade detected",
        }
        logger.info("Pattern 1 (Planning→Tool): detected=%s conf=%.2f", detected, confidence)
        return result

    def detect_pattern_2(self, trace: dict) -> dict:
        """Tool Use → Memory cascade.

        Signal: tool execution feedback misinterpreted → false belief stored.
        Detection: compare tool_result in step N with memory_state in step N+1.
                   If memory contains value not present in tool_result → cascade.
        """
        steps = trace.get("steps", [])
        detected = False
        confidence = 0.0
        evidence_parts = []

        # Collect all tool result values
        tool_result_values = set()
        for step in steps:
            tool_result = _safe_json_loads(step.get("tool_result"))
            if isinstance(tool_result, dict):
                for v in tool_result.values():
                    if not isinstance(v, (dict, list)):
                        tool_result_values.add(str(v))

        # Check memory_write steps for values absent from tool results
        for step in steps:
            if step.get("step_type") != "memory_write":
                continue

            mem_state = _safe_json_loads(step.get("memory_state"))
            if not isinstance(mem_state, dict):
                continue

            for key, value in mem_state.items():
                # Extract actual value
                if isinstance(value, dict) and "value" in value:
                    val = value["value"]
                    # Check for injected flag
                    if value.get("injected"):
                        detected = True
                        confidence = max(confidence, 0.95)
                        evidence_parts.append(
                            f"Injected memory value: {key}={val}"
                        )
                        continue
                else:
                    val = value

                val_str = str(val)
                # Skip trivial values
                if val_str in ("None", "True", "False", "", "0", "unknown"):
                    continue

                if len(val_str) > 2 and val_str not in tool_result_values:
                    detected = True
                    confidence = max(confidence, 0.75)
                    evidence_parts.append(
                        f"Memory value '{key}={val_str}' not found in any tool_result"
                    )

        result = {
            "detected": detected,
            "confidence": round(confidence, 2),
            "evidence": "; ".join(evidence_parts) if evidence_parts else "No cascade detected",
        }
        logger.info("Pattern 2 (Tool→Memory): detected=%s conf=%.2f", detected, confidence)
        return result

    def detect_pattern_3(self, trace: dict) -> dict:
        """Memory → Output cascade.

        Signal: final answer uses parametric prior instead of retrieved/observed facts.
        Detection: compare final_answer tokens with retrieval_results and tool_results.
                   If final_answer contains claims absent from both → cascade.
        """
        final_answer = trace.get("final_answer", "")
        if not final_answer:
            return {"detected": False, "confidence": 0.0, "evidence": "No final answer"}

        steps = trace.get("steps", [])

        # Collect all evidence from retrieval and tool results
        evidence_text = ""
        for step in steps:
            if step.get("step_type") in ("observation", "retrieval"):
                evidence_text += " " + str(step.get("content", ""))
            tool_result = step.get("tool_result")
            if tool_result:
                evidence_text += " " + str(tool_result)
            retrieval = step.get("retrieval_results")
            if retrieval:
                evidence_text += " " + str(retrieval)

        if not evidence_text.strip():
            return {
                "detected": False,
                "confidence": 0.0,
                "evidence": "No observation/retrieval evidence to compare",
            }

        detected = False
        confidence = 0.0
        evidence_parts = []

        # Extract significant numbers from final answer
        answer_numbers = set(re.findall(r"-?\d+\.?\d*", final_answer))
        evidence_numbers = set(re.findall(r"-?\d+\.?\d*", evidence_text))

        # Filter trivial numbers (single digits, common values)
        significant_answer_nums = {n for n in answer_numbers if len(n) > 1}
        unsupported_nums = significant_answer_nums - evidence_numbers

        if unsupported_nums:
            detected = True
            confidence = 0.70
            evidence_parts.append(
                f"Final answer contains unsupported numbers: {unsupported_nums}"
            )

        # Extract significant named entities (capitalized words of length > 4)
        answer_entities = set(re.findall(r"\b[A-Z][a-z]{4,}\b", final_answer))
        evidence_entities = set(re.findall(r"\b[A-Z][a-z]{4,}\b", evidence_text))

        unsupported_entities = answer_entities - evidence_entities
        # Filter common words and template terms
        common_words = {
            "The", "This", "That", "There", "Then", "What", "When",
            "Where", "None", "True", "False", "Weather", "Patient",
            "Event", "Created", "Retrieved", "Completed", "Deleted",
            "Meeting", "Review", "Standup", "Planning", "Training",
            "Workshop", "Interview", "Offsite", "Appointment",
            "Forecast", "Historical", "Calendar", "Medical",
        }
        unsupported_entities -= common_words

        # Only flag if there are multiple unsupported entities (reduce false positives)
        if len(unsupported_entities) >= 2:
            detected = True
            confidence = max(confidence, 0.65)
            evidence_parts.append(
                f"Final answer contains unsupported entities: {unsupported_entities}"
            )

        # Check for known false values from injections
        injections = trace.get("injections", [])
        for inj in injections:
            injected_val = _safe_json_loads(inj.get("injected_value"))
            if injected_val and str(injected_val) in final_answer:
                detected = True
                confidence = max(confidence, 0.95)
                evidence_parts.append(
                    f"Injected value '{injected_val}' appears in final answer"
                )

        result = {
            "detected": detected,
            "confidence": round(confidence, 2),
            "evidence": "; ".join(evidence_parts) if evidence_parts else "No cascade detected",
        }
        logger.info("Pattern 3 (Memory→Output): detected=%s conf=%.2f", detected, confidence)
        return result

    def detect_all(self, trace: dict) -> dict:
        """Run all 3 detectors. Return combined result."""
        p1 = self.detect_pattern_1(trace)
        p2 = self.detect_pattern_2(trace)
        p3 = self.detect_pattern_3(trace)

        cascade_chain = []
        if p1["detected"]:
            cascade_chain.append("planning_to_tool_use")
        if p2["detected"]:
            cascade_chain.append("tool_use_to_memory")
        if p3["detected"]:
            cascade_chain.append("memory_to_output")

        result = {
            "pattern_1": p1,
            "pattern_2": p2,
            "pattern_3": p3,
            "cascade_count": len(cascade_chain),
            "cascade_chain": cascade_chain,
        }

        logger.info(
            "Cascade detection: %d patterns fired: %s",
            result["cascade_count"], cascade_chain,
        )
        return result
