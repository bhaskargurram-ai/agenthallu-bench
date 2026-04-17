"""Tests for all 3 injectors: P2 parameter, P3 memory, P4 cross-agent.

Every error type is tested with a known correct param dict.
Asserts the output is detectably wrong AND the injection is logged to DB.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.tool_executor import ToolExecutor, TOOL_SCHEMAS
from agent.memory_manager import MemoryManager
from tracer.trace_logger import TraceLogger
from injector.parameter_injector import ParameterInjector
from injector.memory_injector import MemoryInjector
from injector.propagation_injector import MultiAgentChain


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def injector():
    return ParameterInjector(seed=42)


@pytest.fixture
def executor():
    return ToolExecutor()


@pytest.fixture
def tracer(tmp_path):
    db_path = str(tmp_path / "test_injector.db")
    t = TraceLogger(db_path)
    t.init_db()
    return t


@pytest.fixture
def session_id():
    return "test_session_001"


def _start_test_session(tracer, session_id, injection_type=None):
    tracer.start_session(
        session_id=session_id,
        task_id="test_task",
        model="test",
        domain="weather",
        injection_type=injection_type,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — P2 Parameter Injector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterInjector:
    """Test all 4 error types with known correct params."""

    # ── type_mismatch ────────────────────────────────────────────────────

    def test_type_mismatch_int_to_string(self, injector):
        """Integer param → string version."""
        params = {"city": "Tokyo", "days": 5}
        schema = TOOL_SCHEMAS["get_forecast"]
        result = injector.inject("get_forecast", params, "type_mismatch", schema)
        # days should now be a string
        assert isinstance(result["days"], str), f"Expected str, got {type(result['days'])}"
        assert result["days"] == "five"

    def test_type_mismatch_string_to_int(self, injector):
        """String param → int 0."""
        params = {"city": "London", "date": "2024-06-15"}
        schema = TOOL_SCHEMAS["get_weather"]
        result = injector.inject("get_weather", params, "type_mismatch", schema)
        # First string param (city) should become int 0
        assert result["city"] == 0

    def test_type_mismatch_bool_to_string(self, injector):
        """Boolean param → string "true"."""
        params = {"city": "Tokyo", "days": 3, "include_hourly": True}
        # include_hourly is bool but days (int) comes first in iteration
        schema = TOOL_SCHEMAS["get_forecast"]
        result = injector.inject("get_forecast", params, "type_mismatch", schema)
        # days (int) gets corrupted first
        assert isinstance(result["days"], str)

    def test_type_mismatch_causes_validation_error(self, injector, executor):
        """Injected type mismatch should cause validation failure."""
        params = {"city": "Tokyo", "days": 5}
        schema = TOOL_SCHEMAS["get_forecast"]
        corrupted = injector.inject("get_forecast", params, "type_mismatch", schema)
        result = executor.execute("get_forecast", corrupted)
        assert result["validation_errors"], "Expected validation errors from type mismatch"

    # ── out_of_range ─────────────────────────────────────────────────────

    def test_out_of_range_numeric(self, injector):
        """Numeric param should exceed max bounds."""
        params = {"city": "Tokyo", "days": 5}
        schema = TOOL_SCHEMAS["get_forecast"]
        result = injector.inject("get_forecast", params, "out_of_range", schema)
        # days max=14, so should be 140
        assert result["days"] == 140

    def test_out_of_range_enum(self, injector):
        """Enum param should have invalid value."""
        params = {"city": "London", "date": "2024-06-15", "unit": "celsius"}
        schema = TOOL_SCHEMAS["get_weather"]
        result = injector.inject("get_weather", params, "out_of_range", schema)
        assert result["unit"] not in ["celsius", "fahrenheit"]

    def test_out_of_range_causes_validation_error(self, injector, executor):
        """Out-of-range should cause validation failure."""
        params = {"city": "Tokyo", "days": 5}
        schema = TOOL_SCHEMAS["get_forecast"]
        corrupted = injector.inject("get_forecast", params, "out_of_range", schema)
        result = executor.execute("get_forecast", corrupted)
        assert result["validation_errors"], "Expected validation errors from out_of_range"

    # ── missing_required ─────────────────────────────────────────────────

    def test_missing_required_removes_param(self, injector):
        """Should remove the first required param."""
        params = {"city": "London", "date": "2024-06-15", "unit": "celsius"}
        schema = TOOL_SCHEMAS["get_weather"]
        result = injector.inject("get_weather", params, "missing_required", schema)
        # "city" is first required param
        assert "city" not in result
        assert "date" in result

    def test_missing_required_causes_validation_error(self, injector, executor):
        """Missing required param should cause validation failure."""
        params = {"city": "London", "date": "2024-06-15"}
        schema = TOOL_SCHEMAS["get_weather"]
        corrupted = injector.inject("get_weather", params, "missing_required", schema)
        result = executor.execute("get_weather", corrupted)
        assert result["validation_errors"], "Expected validation errors from missing_required"

    # ── semantic_wrong ───────────────────────────────────────────────────

    def test_semantic_wrong_city(self, injector):
        """City should be swapped to a different real city."""
        params = {"city": "Paris", "date": "2024-01-15"}
        schema = TOOL_SCHEMAS["get_weather"]
        result = injector.inject("get_weather", params, "semantic_wrong", schema)
        assert result["city"] != "Paris"
        assert result["city"] == "Lyon"  # Paris → Lyon in alternatives

    def test_semantic_wrong_date(self, injector):
        """City gets swapped first (checked before date); Lagos→Abuja."""
        params = {"city": "Lagos", "start_date": "2024-01-15", "end_date": "2024-01-20"}
        schema = TOOL_SCHEMAS["get_historical"]
        result = injector.inject("get_historical", params, "semantic_wrong", schema)
        # City swap happens before date shift in the semantic_wrong priority order
        assert result["city"] == "Abuja"

    def test_semantic_wrong_patient_id(self, injector):
        """Patient ID should be swapped to different patient."""
        params = {"patient_id": "P001", "fields": ["all"]}
        schema = TOOL_SCHEMAS["get_patient_record"]
        result = injector.inject("get_patient_record", params, "semantic_wrong", schema)
        assert result["patient_id"] != "P001"
        assert result["patient_id"] == "P002"

    def test_semantic_wrong_still_valid_schema(self, injector, executor):
        """Semantic errors should pass schema validation (that's the danger)."""
        params = {"city": "Paris", "date": "2024-01-15"}
        schema = TOOL_SCHEMAS["get_weather"]
        corrupted = injector.inject("get_weather", params, "semantic_wrong", schema)
        result = executor.execute("get_weather", corrupted)
        assert result["validation_errors"] == [], "Semantic errors should pass validation!"

    # ── Logging to DB ────────────────────────────────────────────────────

    def test_injection_logged_to_db(self, injector, tracer, session_id):
        """Verify injection is recorded in the injections table."""
        _start_test_session(tracer, session_id, "p2_type_mismatch")

        params = {"city": "Tokyo", "days": 5}
        schema = TOOL_SCHEMAS["get_forecast"]
        injector.inject(
            "get_forecast", params, "type_mismatch", schema,
            tracer=tracer, session_id=session_id,
        )

        trace = tracer.get_session_trace(session_id)
        assert len(trace["injections"]) == 1
        inj = trace["injections"][0]
        assert inj["injection_type"] == "p2_type_mismatch"
        assert inj["target_stage"] == "parameter_generation"

    def test_all_error_types_logged(self, injector, tracer):
        """Each error type should produce a logged injection."""
        for i, error_type in enumerate(["type_mismatch", "out_of_range", "missing_required", "semantic_wrong"]):
            sid = f"test_log_{i}"
            _start_test_session(tracer, sid, f"p2_{error_type}")
            params = {"city": "Tokyo", "days": 5}
            schema = TOOL_SCHEMAS["get_forecast"]
            injector.inject(
                "get_forecast", params, error_type, schema,
                tracer=tracer, session_id=sid,
            )
            trace = tracer.get_session_trace(sid)
            assert len(trace["injections"]) == 1, f"Expected 1 injection for {error_type}"

    # ── tool_executor integration ────────────────────────────────────────

    def test_executor_with_injector_param(self, injector, tracer, session_id):
        """ToolExecutor.execute() with injector parameter corrupts params."""
        _start_test_session(tracer, session_id, "p2_type_mismatch")
        executor = ToolExecutor()
        result = executor.execute(
            "get_forecast",
            {"city": "Tokyo", "days": 5},
            injector=injector,
            injection_error_type="type_mismatch",
            tracer=tracer,
            session_id=session_id,
        )
        # Should have validation errors because days was corrupted to string
        assert result["validation_errors"]

    def test_executor_with_instance_injector(self, injector, tracer, session_id):
        """ToolExecutor initialized with injector corrupts when error_type given."""
        _start_test_session(tracer, session_id, "p2_out_of_range")
        executor = ToolExecutor(injector=injector)
        result = executor.execute(
            "get_forecast",
            {"city": "Tokyo", "days": 5},
            injection_error_type="out_of_range",
            tracer=tracer,
            session_id=session_id,
        )
        assert result["validation_errors"]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — P3 Memory Injector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryInjector:
    """Test false belief injection and propagation measurement."""

    def test_inject_false_belief(self, tracer):
        """False belief should appear in working memory."""
        sid = "test_mem_001"
        _start_test_session(tracer, sid, "p3_false_belief")
        mm = MemoryManager(tracer=tracer, session_id=sid)
        mi = MemoryInjector(seed=42)

        # Simulate turn 1: normal observation
        mm.add_observation(1, json.dumps({"event_id": "EVT_0001", "status": "created"}))
        assert mm.working_memory.get("event_id", {}).get("value") == "EVT_0001"

        # Inject at turn 2
        result = mi.inject_false_belief(
            memory_manager=mm,
            tracer=tracer,
            session_id=sid,
            turn_id=2,
            belief_key="event_id",
            false_value="EVT_9999",
            true_value="EVT_0001",
        )

        # Verify false belief is in memory
        assert mm.working_memory["event_id"]["value"] == "EVT_9999"
        assert mm.working_memory["event_id"]["injected"] is True

    def test_inject_false_belief_logged_to_db(self, tracer):
        """Injection should appear in trace DB."""
        sid = "test_mem_002"
        _start_test_session(tracer, sid, "p3_false_belief")
        mm = MemoryManager(tracer=tracer, session_id=sid)
        mi = MemoryInjector(seed=42)

        mi.inject_false_belief(
            mm, tracer, sid, turn_id=2,
            belief_key="event_id", false_value="EVT_9999", true_value="EVT_0001",
        )

        trace = tracer.get_session_trace(sid)
        assert len(trace["injections"]) == 1
        assert trace["injections"][0]["injection_type"] == "p3_false_belief"

    def test_get_injection_plan(self):
        """Plan should always inject at turn 2."""
        mi = MemoryInjector(seed=42)
        task = {
            "task_id": "calendar_001",
            "domain": "calendar",
            "num_turns": 6,
            "correct_tool_sequence": [
                {"tool": "create_event", "params": {"title": "Meeting", "date": "2024-07-01", "time": "10:00", "duration_minutes": 60}},
            ],
        }
        plan = mi.get_injection_plan(task)
        assert len(plan) == 1
        assert plan[0]["turn_id"] == 2
        assert plan[0]["belief_key"] in ["event_id", "last_created_event", "date"]

    def test_six_turn_calendar_propagation(self, tracer):
        """Run a 6-turn calendar session, inject at turn 2, check turns 3-6."""
        sid = "test_mem_prop"
        _start_test_session(tracer, sid, "p3_false_belief")
        mm = MemoryManager(tracer=tracer, session_id=sid)
        mi = MemoryInjector(seed=42)

        # Turn 1: create event
        mm.add_observation(1, json.dumps({"event_id": "EVT_0001", "status": "created"}))

        # Turn 2: inject false event_id
        mi.inject_false_belief(
            mm, tracer, sid, turn_id=2,
            belief_key="event_id", false_value="EVT_9999", true_value="EVT_0001",
        )

        # Turns 3-6: simulate agent using the (now false) event_id from memory
        # Log explicit action steps with the false event_id in tool_params
        for turn in range(3, 7):
            event_id = mm.working_memory.get("event_id", {}).get("value", "unknown")
            # Log a tool-call step (simulating what the agent would do)
            step_num = tracer.get_step_count(sid) + 1
            tracer.log_step(
                session_id=sid, turn_id=turn, step_number=step_num,
                step_type="action",
                content=f"Agent calls delete_event with event_id={event_id}",
                tool_name="delete_event",
                tool_params_raw={"event_id": event_id},
            )
            # Also feed memory
            mm.add_observation(turn, json.dumps({"event_id": event_id, "status": "processed"}))

        # Verify false value propagated through all turns
        assert mm.working_memory["event_id"]["value"] == "EVT_9999"

        # Measure propagation from trace
        trace = tracer.get_session_trace(sid)
        propagation = mi.measure_propagation_depth(trace, injection_turn=2, false_value="EVT_9999")
        assert propagation["propagation_depth"] >= 1, "False belief should propagate to at least 1 subsequent turn"
        assert propagation["injection_turn"] == 2

    def test_measure_propagation_depth(self):
        """measure_propagation_depth with synthetic trace."""
        mi = MemoryInjector(seed=42)
        trace = {
            "steps": [
                {"turn_id": 1, "tool_params_raw": '{"event_id": "EVT_0001"}', "content": "turn 1", "memory_state": None},
                {"turn_id": 2, "tool_params_raw": None, "content": "injection turn", "memory_state": '{"event_id": "EVT_9999"}'},
                {"turn_id": 3, "tool_params_raw": '{"event_id": "EVT_9999"}', "content": "uses false", "memory_state": None},
                {"turn_id": 4, "tool_params_raw": '{"event_id": "EVT_9999"}', "content": "still false", "memory_state": None},
                {"turn_id": 5, "tool_params_raw": '{"event_id": "EVT_0001"}', "content": "corrected", "memory_state": None},
            ]
        }
        result = mi.measure_propagation_depth(trace, injection_turn=2, false_value="EVT_9999")
        assert result["propagation_depth"] == 2  # turns 3 and 4
        assert 3 in result["affected_turns"]
        assert 4 in result["affected_turns"]
        assert 5 not in result["affected_turns"]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — P4 Multi-Agent Chain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiAgentChain:
    """Test 3-agent chain with injection at Agent 1."""

    @pytest.fixture
    def chain(self):
        executor = ToolExecutor()
        return MultiAgentChain(tool_executor=executor, seed=42)

    @pytest.fixture
    def sample_task(self):
        return {
            "task_id": "weather_001",
            "domain": "weather",
            "query": "What is the weather in New York on 2024-06-15?",
            "correct_tool_sequence": [
                {"tool": "get_weather", "params": {"city": "New York", "date": "2024-06-15"}}
            ],
            "correct_final_answer": "The weather in New York on 2024-06-15.",
            "multi_turn": False,
            "num_turns": 1,
        }

    def test_chain_no_injection(self, chain, sample_task, tracer):
        """Clean run with no injection should produce valid output."""
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            tracer=tracer,
            session_id="p4_clean",
        )
        assert result["agent1_output"]
        assert result["agent2_output"]
        assert result["agent3_output"]
        assert result["final_answer"]
        assert result["propagation_path"] == []

    def test_chain_wrong_subtask(self, chain, sample_task, tracer):
        """wrong_subtask should change the city in the plan."""
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            inject_at_agent=1,
            injection_type="wrong_subtask",
            tracer=tracer,
            session_id="p4_subtask",
        )
        # Agent 1 plan should have "Los Angeles" instead of "New York"
        assert "Los Angeles" in result["agent1_output"]
        assert 1 in result["propagation_path"]

    def test_chain_wrong_tool(self, chain, sample_task, tracer):
        """wrong_tool_spec should replace tool name in plan."""
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            inject_at_agent=1,
            injection_type="wrong_tool_spec",
            tracer=tracer,
            session_id="p4_tool",
        )
        # Plan should have wrong tool
        assert result["agent1_output"] != result["original_agent1"]

    def test_chain_wrong_param(self, chain, sample_task, tracer):
        """wrong_parameter_spec should inject wrong param value."""
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            inject_at_agent=1,
            injection_type="wrong_parameter_spec",
            tracer=tracer,
            session_id="p4_param",
        )
        assert result["agent1_output"] != result["original_agent1"]

    def test_chain_injection_logged(self, chain, sample_task, tracer):
        """Injection should be logged in trace DB."""
        sid = "p4_logged"
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            inject_at_agent=1,
            injection_type="wrong_subtask",
            tracer=tracer,
            session_id=sid,
        )
        trace = tracer.get_session_trace(sid)
        assert len(trace["injections"]) == 1
        assert trace["injections"][0]["injection_type"] == "p4_wrong_subtask"

    def test_measure_cross_agent_eps(self, chain, sample_task, tracer):
        """EPS measurement should return valid structure."""
        result = chain.run_chain(
            query=sample_task["query"],
            domain="weather",
            task=sample_task,
            inject_at_agent=1,
            injection_type="wrong_subtask",
            tracer=tracer,
            session_id="p4_eps",
        )
        eps = chain.measure_cross_agent_eps(result)
        assert eps["injected_at"] == 1
        assert eps["injection_type"] == "wrong_subtask"
        assert isinstance(eps["propagated_to_output"], bool)
        assert eps["amplification_factor"] >= 1.0

    def test_five_chain_tasks_with_injection(self, chain, tracer):
        """Run 5 chain tasks with different injection types."""
        tasks = [
            {"task_id": f"p4_test_{i}", "domain": "weather",
             "query": f"Weather query {i}",
             "correct_tool_sequence": [
                 {"tool": "get_weather", "params": {"city": "New York", "date": "2024-06-15"}}
             ],
             "correct_final_answer": "Weather answer"}
            for i in range(5)
        ]
        injection_types = ["wrong_subtask", "wrong_tool_spec", "wrong_parameter_spec",
                          "wrong_subtask", "wrong_tool_spec"]

        results = []
        for task, inj_type in zip(tasks, injection_types):
            result = chain.run_chain(
                query=task["query"],
                domain="weather",
                task=task,
                inject_at_agent=1,
                injection_type=inj_type,
                tracer=tracer,
                session_id=f"p4_batch_{task['task_id']}",
            )
            eps = chain.measure_cross_agent_eps(result)
            results.append(eps)

        # All 5 should have been injected at agent 1
        assert all(r["injected_at"] == 1 for r in results)
        assert len(results) == 5
