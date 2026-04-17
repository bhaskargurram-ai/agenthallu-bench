"""Multi-turn memory state manager.

Tracks short_term (last N messages), working_memory (extracted facts),
and episodic (turn summaries) for the ReAct agent.
"""

import logging
from typing import Any, Optional

from config import RANDOM_SEED

logger = logging.getLogger(__name__)

SHORT_TERM_LIMIT = 6  # last N (role, content) pairs


class MemoryManager:
    """Manages agent memory across turns."""

    def __init__(self, tracer=None, session_id: Optional[str] = None):
        self.short_term: list[tuple[str, str]] = []          # (role, content) pairs
        self.working_memory: dict[str, Any] = {}             # key-value facts
        self.episodic: list[tuple[int, str]] = []            # (turn_id, summary)
        self._turn_log: list[dict] = []                      # internal log for tracing
        self.tracer = tracer  # Optional TraceLogger instance
        self.session_id = session_id
        logger.info("MemoryManager initialized")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to short-term memory, evicting oldest if over limit."""
        self.short_term.append((role, content))
        if len(self.short_term) > SHORT_TERM_LIMIT:
            self.short_term = self.short_term[-SHORT_TERM_LIMIT:]
        logger.info("Short-term memory: %d messages", len(self.short_term))

    def add_observation(self, turn_id: int, observation: str) -> dict:
        """Extract key facts from an observation and store in working memory.

        Returns dict of extracted facts for tracing.
        """
        # Rule-based fact extraction (no LLM needed for deterministic testing)
        facts = self._extract_facts(observation)

        for key, value in facts.items():
            self.working_memory[key] = {"value": value, "turn_id": turn_id}

        # Add episodic summary
        summary = observation[:200] if len(observation) > 200 else observation
        self.episodic.append((turn_id, summary))

        log_entry = {
            "turn_id": turn_id,
            "extracted_facts": facts,
            "working_memory_state": self.get_working_memory_snapshot(),
        }
        self._turn_log.append(log_entry)

        logger.info(
            "Turn %d: extracted %d facts, working_memory has %d entries",
            turn_id, len(facts), len(self.working_memory),
        )

        # Trace memory write
        if self.tracer and self.session_id:
            for key, value in facts.items():
                self.tracer.log_memory_write(self.session_id, turn_id, key, value, turn_id)

        return log_entry

    def _extract_facts(self, observation: str) -> dict[str, Any]:
        """Rule-based fact extraction from observation text.

        Extracts key-value pairs from structured tool results.
        """
        facts = {}
        # Try to parse as a structured result
        import json
        try:
            data = json.loads(observation) if isinstance(observation, str) else observation
            if isinstance(data, dict):
                # Store important fields as facts
                for key in ["temperature", "humidity", "condition", "city", "date",
                            "event_id", "title", "name", "patient_id",
                            "interaction", "severity", "appointment_id",
                            "success", "status"]:
                    if key in data:
                        facts[key] = data[key]
                # Store nested results
                if "result" in data and isinstance(data["result"], dict):
                    for key, value in data["result"].items():
                        if not isinstance(value, (list, dict)):
                            facts[key] = value
        except (json.JSONDecodeError, TypeError):
            # Plain text — store as last_observation
            facts["last_observation"] = observation[:500]

        return facts

    def get_context(self) -> str:
        """Return formatted string of short_term + working_memory for the prompt."""
        parts = []

        # Short-term conversation history
        if self.short_term:
            parts.append("=== Recent Conversation ===")
            for role, content in self.short_term:
                parts.append(f"{role}: {content[:300]}")

        # Working memory facts
        if self.working_memory:
            parts.append("\n=== Known Facts ===")
            for key, entry in self.working_memory.items():
                val = entry["value"] if isinstance(entry, dict) and "value" in entry else entry
                parts.append(f"- {key}: {val}")

        return "\n".join(parts)

    def get_working_memory_snapshot(self) -> dict:
        """Return a copy of current working memory state."""
        snapshot = {}
        for key, entry in self.working_memory.items():
            if isinstance(entry, dict) and "value" in entry:
                snapshot[key] = {"value": entry["value"], "turn_id": entry["turn_id"]}
            else:
                snapshot[key] = entry
        return snapshot

    def inject_false_belief(self, key: str, value: Any, turn_id: int) -> dict:
        """Directly write a false fact into working_memory. Used by P3 injector.

        Returns injection log entry.
        """
        original = self.working_memory.get(key)
        self.working_memory[key] = {"value": value, "turn_id": turn_id, "injected": True}

        log_entry = {
            "action": "inject_false_belief",
            "key": key,
            "original_value": original,
            "injected_value": value,
            "turn_id": turn_id,
        }
        self._turn_log.append(log_entry)

        logger.warning(
            "FALSE BELIEF INJECTED: key=%s value=%s at turn=%d", key, value, turn_id
        )
        return log_entry

    def clear(self) -> None:
        """Reset all memory."""
        self.short_term.clear()
        self.working_memory.clear()
        self.episodic.clear()
        self._turn_log.clear()
        logger.info("MemoryManager cleared")
