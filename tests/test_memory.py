"""Tests for agent/memory_manager.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from agent.memory_manager import MemoryManager


@pytest.fixture
def mm():
    return MemoryManager()


class TestMemoryManager:
    def test_add_message(self, mm):
        mm.add_message("user", "Hello")
        mm.add_message("assistant", "Hi there")
        assert len(mm.short_term) == 2

    def test_short_term_eviction(self, mm):
        for i in range(10):
            mm.add_message("user", f"message {i}")
        assert len(mm.short_term) == 6  # SHORT_TERM_LIMIT

    def test_add_observation_json(self, mm):
        obs = json.dumps({"temperature": 25, "humidity": 60, "city": "London"})
        result = mm.add_observation(1, obs)
        assert "extracted_facts" in result
        assert result["extracted_facts"]["temperature"] == 25
        assert result["extracted_facts"]["city"] == "London"

    def test_add_observation_plain_text(self, mm):
        result = mm.add_observation(1, "The weather is sunny today")
        assert "last_observation" in result["extracted_facts"]

    def test_working_memory_round_trip(self, mm):
        obs = json.dumps({"event_id": "EVT_001", "status": "created"})
        mm.add_observation(1, obs)
        snapshot = mm.get_working_memory_snapshot()
        assert snapshot["event_id"]["value"] == "EVT_001"
        assert snapshot["status"]["value"] == "created"

    def test_get_context_includes_facts(self, mm):
        obs = json.dumps({"temperature": 22, "city": "Paris"})
        mm.add_observation(1, obs)
        mm.add_message("user", "What is the weather?")
        context = mm.get_context()
        assert "temperature" in context
        assert "Paris" in context
        assert "What is the weather?" in context

    def test_inject_false_belief(self, mm):
        # First set a real fact
        obs = json.dumps({"event_id": "EVT_001"})
        mm.add_observation(1, obs)
        assert mm.working_memory["event_id"]["value"] == "EVT_001"

        # Inject false belief
        log = mm.inject_false_belief("event_id", "EVT_999", turn_id=2)
        assert mm.working_memory["event_id"]["value"] == "EVT_999"
        assert mm.working_memory["event_id"]["injected"] is True
        assert log["original_value"]["value"] == "EVT_001"
        assert log["injected_value"] == "EVT_999"

    def test_episodic_memory(self, mm):
        mm.add_observation(1, "First observation")
        mm.add_observation(2, "Second observation")
        assert len(mm.episodic) == 2
        assert mm.episodic[0][0] == 1
        assert mm.episodic[1][0] == 2

    def test_clear(self, mm):
        mm.add_message("user", "test")
        mm.add_observation(1, json.dumps({"key": "value"}))
        mm.clear()
        assert len(mm.short_term) == 0
        assert len(mm.working_memory) == 0
        assert len(mm.episodic) == 0
