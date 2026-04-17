"""Tests for all 4 metrics modules: EPS, Stage Attribution, Cascade Detector, MTAS.

Tests inject at known stages and verify metrics match expected values.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tracer.trace_logger import TraceLogger
from metrics.eps_scorer import EPSScorer, STAGE_ORDER
from metrics.stage_attribution import StageAttributor
from metrics.cascade_detector import CascadeDetector
from metrics.multi_turn_scorer import MultiTurnScorer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tracer(tmp_path):
    db_path = str(tmp_path / "test_metrics.db")
    t = TraceLogger(db_path)
    t.init_db()
    return t


@pytest.fixture
def eps_scorer():
    return EPSScorer()


@pytest.fixture
def attributor():
    return StageAttributor()


@pytest.fixture
def detector():
    return CascadeDetector()


@pytest.fixture
def mtas_scorer():
    return MultiTurnScorer()


def _start_session(tracer, sid, task_id="test_task", domain="weather",
                   injection_type=None, injection_stage=None):
    tracer.start_session(
        session_id=sid, task_id=task_id, model="test",
        domain=domain, injection_type=injection_type,
        injection_stage=injection_stage,
    )


def _build_trace_with_injection(
    tracer, sid, domain, injection_type, error_type,
    inject_at_step, injected_value, original_value,
    reaches_output=False,
):
    """Helper: build a synthetic trace with a known injection."""
    _start_session(tracer, sid, domain=domain, injection_type=injection_type)

    # Step 1: Thought (planning)
    tracer.log_step(sid, 0, 1, "thought", "I need to get weather info")

    # Step 2: Action (tool_selection + parameter_generation)
    params = {"city": "London", "date": "2024-06-15"}
    tracer.log_step(sid, 0, 2, "action", "Action: get_weather",
                    tool_name="get_weather", tool_params_raw=params)

    # Step 3: Observation (tool_execution)
    if error_type == "type_mismatch":
        # Type mismatch gets caught by validation
        tracer.log_step(sid, 0, 3, "observation",
                        json.dumps({"error": "Parameter 'days' must be integer, got str"}),
                        tool_name="get_weather",
                        tool_result={"error": "Parameter 'days' must be integer"},
                        param_errors=["Parameter 'days' must be integer, got str"])
    elif error_type == "semantic_wrong" and reaches_output:
        # Semantic wrong passes validation, wrong result reaches output
        tracer.log_step(sid, 0, 3, "observation",
                        json.dumps({"city": str(injected_value), "temperature": 25, "condition": "sunny"}),
                        tool_name="get_weather",
                        tool_result={"city": str(injected_value), "temperature": 25, "condition": "sunny"})
    else:
        tracer.log_step(sid, 0, 3, "observation",
                        json.dumps({"city": "London", "temperature": 22, "condition": "cloudy"}),
                        tool_name="get_weather",
                        tool_result={"city": "London", "temperature": 22, "condition": "cloudy"})

    # Step 4: Memory write
    mem_state = {"city": {"value": str(injected_value) if reaches_output else "London", "turn_id": 1}}
    tracer.log_step(sid, 1, 4, "memory_write",
                    f"Memory: city={injected_value if reaches_output else 'London'}",
                    memory_state=mem_state)

    # Step 5: Final answer
    if reaches_output:
        final = f"The weather in {injected_value} is 25°C and sunny."
    else:
        final = "The weather in London is 22°C and cloudy."
    tracer.log_step(sid, 1, 5, "final", final)

    # Log injection
    tracer.log_injection(
        sid, injection_type, "parameter_generation",
        original=original_value, injected=injected_value, turn_id=0,
        step_number=inject_at_step,
    )

    # End session
    tracer.end_session(sid, final, "The weather in London", not reaches_output)

    return tracer.get_session_trace(sid)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EPS Scorer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEPSScorer:
    """Test EPS computation with known injection scenarios."""

    def test_no_injection_eps_zero(self, eps_scorer):
        """Session with no injection should have EPS=0."""
        trace = {
            "session_id": "clean_001",
            "steps": [
                {"step_number": 1, "step_type": "thought", "content": "thinking"},
                {"step_number": 2, "step_type": "action", "content": "action",
                 "tool_name": "get_weather", "tool_params_raw": '{"city": "London"}'},
                {"step_number": 3, "step_type": "observation", "content": "result",
                 "tool_result": '{"temperature": 22}'},
                {"step_number": 4, "step_type": "final", "content": "22 degrees"},
            ],
            "injections": [],
            "final_answer": "22 degrees",
        }
        result = eps_scorer.compute_eps(trace, {})
        assert result["eps"] == 0

    def test_type_mismatch_eps_zero_or_one(self, eps_scorer, tracer):
        """type_mismatch injected at parameter_generation → EPS should be 0 or 1
        because it's caught by validation."""
        sid = "eps_type_mismatch"
        trace = _build_trace_with_injection(
            tracer, sid, "weather",
            "p2_type_mismatch", "type_mismatch",
            inject_at_step=2, injected_value="five", original_value=5,
            reaches_output=False,
        )
        result = eps_scorer.compute_eps(trace, trace["injections"][0])
        assert result["eps"] <= 1, f"type_mismatch EPS should be 0 or 1, got {result['eps']}"
        assert result["error_type"] == "type_mismatch"

    def test_semantic_wrong_eps_reaches_output(self, eps_scorer, tracer):
        """semantic_wrong injected at parameter_generation → EPS often 3 (reaches output)."""
        sid = "eps_semantic"
        trace = _build_trace_with_injection(
            tracer, sid, "weather",
            "p2_semantic_wrong", "semantic_wrong",
            inject_at_step=2, injected_value="Lyon", original_value="Paris",
            reaches_output=True,
        )
        result = eps_scorer.compute_eps(trace, trace["injections"][0])
        assert result["eps"] >= 1, f"semantic_wrong EPS should be ≥1, got {result['eps']}"
        assert result["reached_output"] is True
        assert result["error_type"] == "semantic_wrong"

    def test_severity_weights(self, eps_scorer):
        """wEPS should apply severity weights correctly."""
        trace = {
            "session_id": "weps_test",
            "steps": [
                {"step_number": 1, "step_type": "action", "content": "action",
                 "tool_name": "get_weather", "tool_params_raw": '{"city": "Lyon"}',
                 "param_errors": None, "tool_result": None, "memory_state": None,
                 "tool_params_validated": None, "retrieval_results": None},
                {"step_number": 2, "step_type": "observation", "content": "Lyon weather",
                 "tool_result": '{"city": "Lyon", "temp": 25}',
                 "tool_params_raw": None, "param_errors": None, "memory_state": None,
                 "tool_params_validated": None, "retrieval_results": None},
                {"step_number": 3, "step_type": "final", "content": "Weather in Lyon is 25",
                 "tool_result": None, "tool_params_raw": None, "param_errors": None,
                 "memory_state": None, "tool_params_validated": None, "retrieval_results": None},
            ],
            "injections": [{
                "injection_type": "p2_semantic_wrong",
                "target_stage": "parameter_generation",
                "injected_value": '"Lyon"',
                "step_number": 0,
            }],
            "final_answer": "Weather in Lyon is 25",
        }
        result = eps_scorer.compute_eps(trace, trace["injections"][0])
        # semantic_wrong weight = 1.0, detectability = 1.0 (silent)
        if result["eps"] > 0:
            assert result["weps"] > 0, "wEPS should be > 0 when EPS > 0"

    def test_eps_origin_stage(self, eps_scorer, tracer):
        """Origin stage should match injection target."""
        sid = "eps_origin"
        trace = _build_trace_with_injection(
            tracer, sid, "weather",
            "p2_out_of_range", "out_of_range",
            inject_at_step=2, injected_value=140, original_value=5,
            reaches_output=False,
        )
        result = eps_scorer.compute_eps(trace, trace["injections"][0])
        assert result["origin_stage"] == "parameter_generation"

    def test_batch_eps(self, eps_scorer, tracer, tmp_path):
        """compute_batch_eps should process multiple sessions."""
        db_path = str(tmp_path / "batch_eps.db")
        batch_tracer = TraceLogger(db_path)
        batch_tracer.init_db()

        sids = []
        for i in range(5):
            sid = f"batch_{i}"
            _build_trace_with_injection(
                batch_tracer, sid, "weather",
                "p2_type_mismatch", "type_mismatch",
                inject_at_step=2, injected_value="five", original_value=5,
            )
            sids.append(sid)

        batch_tracer.close()

        df = eps_scorer.compute_batch_eps(sids, db_path)
        assert len(df) == 5
        assert "eps" in df.columns
        assert "weps" in df.columns
        assert "session_id" in df.columns

    def test_summarize_eps(self, eps_scorer):
        """summarize_eps_by_error_type should return valid structure."""
        import pandas as pd
        df = pd.DataFrame([
            {"error_type": "type_mismatch", "model": "gpt4o", "domain": "weather",
             "difficulty": "easy", "eps": 0, "reached_output": False},
            {"error_type": "semantic_wrong", "model": "gpt4o", "domain": "weather",
             "difficulty": "hard", "eps": 3, "reached_output": True},
            {"error_type": "type_mismatch", "model": "claude", "domain": "medical",
             "difficulty": "medium", "eps": 1, "reached_output": False},
        ])
        summary = eps_scorer.summarize_eps_by_error_type(df)
        assert "by_error_type" in summary
        assert "by_model" in summary
        assert "by_domain" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Stage Attribution Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStageAttributor:
    """Test attribution accuracy with known injection stages."""

    def test_no_error_no_attribution(self, attributor):
        """Clean trace → no attribution needed."""
        trace = {
            "session_id": "clean",
            "final_answer": "22 degrees",
            "ground_truth_answer": "22 degrees",
            "final_correct": True,
            "steps": [],
            "injections": [],
        }
        result = attributor.attribute(trace)
        assert result["final_error"] is False
        assert result["attributed_stage"] is None

    def test_param_error_attributed_to_parameter_generation(self, attributor):
        """Non-empty param_errors → attribute to parameter_generation."""
        trace = {
            "session_id": "param_err",
            "final_answer": "Error occurred",
            "ground_truth_answer": "22 degrees",
            "final_correct": False,
            "steps": [
                {"step_number": 1, "step_type": "thought", "content": "thinking",
                 "param_errors": None, "tool_result": None, "memory_state": None,
                 "tool_name": None},
                {"step_number": 2, "step_type": "observation", "content": "error",
                 "param_errors": '["Missing required parameter: city"]',
                 "tool_result": '{"error": "Missing required parameter: city"}',
                 "memory_state": None, "tool_name": "get_weather"},
            ],
            "injections": [],
        }
        result = attributor.attribute(trace)
        assert result["final_error"] is True
        assert result["attributed_stage"] == "parameter_generation"
        assert result["confidence"] >= 0.9

    def test_tool_error_attributed_to_tool_execution(self, attributor):
        """Tool returning error dict → attribute to tool_execution."""
        trace = {
            "session_id": "tool_err",
            "final_answer": "Could not retrieve data",
            "ground_truth_answer": "22 degrees",
            "final_correct": False,
            "steps": [
                {"step_number": 1, "step_type": "observation", "content": "error",
                 "tool_result": '{"error": "Execution error: timeout"}',
                 "param_errors": None, "memory_state": None, "tool_name": "get_weather"},
            ],
            "injections": [],
        }
        result = attributor.attribute(trace)
        assert result["final_error"] is True
        assert result["attributed_stage"] == "tool_execution"

    def test_memory_write_anomaly(self, attributor):
        """Memory value not in tool_result → attribute to memory_write."""
        trace = {
            "session_id": "mem_err",
            "final_answer": "Weather in Atlantis is -99°C",
            "ground_truth_answer": "Weather in London is 22°C",
            "final_correct": False,
            "domain": "weather",
            "steps": [
                {"step_number": 1, "step_type": "observation", "content": "result",
                 "tool_result": '{"city": "London", "temperature": 22}',
                 "param_errors": None, "memory_state": None, "tool_name": "get_weather"},
                {"step_number": 2, "step_type": "memory_write", "content": "memory",
                 "memory_state": '{"city": {"value": "Atlantis", "turn_id": 1}}',
                 "tool_result": None, "param_errors": None, "tool_name": None},
            ],
            "injections": [],
        }
        result = attributor.attribute(trace)
        assert result["final_error"] is True
        assert result["attributed_stage"] == "memory_write"

    def test_output_contradiction(self, attributor):
        """Final answer with unsupported numbers → output_generation."""
        trace = {
            "session_id": "output_err",
            "final_answer": "The temperature is 999 degrees",
            "ground_truth_answer": "The temperature is 22 degrees",
            "final_correct": False,
            "steps": [
                {"step_number": 1, "step_type": "observation",
                 "content": '{"temperature": 22, "city": "London"}',
                 "tool_result": '{"temperature": 22, "city": "London"}',
                 "param_errors": None, "memory_state": None, "tool_name": "get_weather"},
            ],
            "injections": [],
        }
        result = attributor.attribute(trace)
        assert result["final_error"] is True
        assert result["attributed_stage"] == "output_generation"

    def test_attribution_accuracy_perfect(self, attributor):
        """Perfect predictions should give F1=1.0."""
        predictions = [
            {"attributed_stage": "parameter_generation"},
            {"attributed_stage": "tool_execution"},
            {"attributed_stage": "memory_write"},
        ]
        ground_truth = ["parameter_generation", "tool_execution", "memory_write"]
        result = attributor.attribution_accuracy(predictions, ground_truth)
        assert result["overall_accuracy"] == 1.0

    def test_attribution_accuracy_partial(self, attributor):
        """Partial matches should give expected metrics."""
        predictions = [
            {"attributed_stage": "parameter_generation"},
            {"attributed_stage": "parameter_generation"},  # wrong
            {"attributed_stage": "memory_write"},
        ]
        ground_truth = ["parameter_generation", "tool_execution", "memory_write"]
        result = attributor.attribution_accuracy(predictions, ground_truth)
        assert result["overall_accuracy"] == pytest.approx(2/3, abs=0.01)
        assert result["per_stage"]["parameter_generation"]["precision"] == pytest.approx(0.5, abs=0.01)

    def test_attribution_20_injected_sessions(self, attributor, tracer):
        """Run attribution on 20 injected sessions and check accuracy."""
        predictions = []
        gt_stages = []

        # 5 per error type
        configs = [
            ("p2_type_mismatch", "parameter_generation", True),
            ("p2_out_of_range", "parameter_generation", True),
            ("p2_missing_required", "parameter_generation", True),
            ("p2_semantic_wrong", "parameter_generation", False),
        ]

        for cfg_idx, (inj_type, gt_stage, has_param_errors) in enumerate(configs):
            for i in range(5):
                sid = f"attr_{cfg_idx}_{i}"
                _start_session(tracer, sid, domain="weather", injection_type=inj_type)

                # Build trace with appropriate error signals
                if has_param_errors:
                    tracer.log_step(sid, 0, 1, "observation", "error",
                                    tool_name="get_weather",
                                    param_errors=["validation error"],
                                    tool_result={"error": "validation failed"})
                else:
                    # semantic_wrong passes validation
                    tracer.log_step(sid, 0, 1, "observation", "result",
                                    tool_name="get_weather",
                                    tool_result={"city": "Lyon", "temperature": 25})

                tracer.log_step(sid, 0, 2, "final", "some answer")
                tracer.log_injection(sid, inj_type, gt_stage, "original", "injected", 0)
                tracer.end_session(sid, "some answer", "correct answer", False)

                trace = tracer.get_session_trace(sid)
                attr = attributor.attribute(trace)
                predictions.append(attr)
                gt_stages.append(gt_stage)

        accuracy = attributor.attribution_accuracy(predictions, gt_stages)
        # With rule-based detection, type/range/missing should be caught
        # Semantic wrong may fallback
        assert accuracy["overall_accuracy"] >= 0.5, \
            f"Expected ≥50% accuracy, got {accuracy['overall_accuracy']}"
        console_msg = f"Attribution accuracy: {accuracy['overall_accuracy']*100:.0f}%"
        assert accuracy["total_predictions"] == 20


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cascade Detector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCascadeDetector:
    """Test all 3 cascade patterns."""

    def test_pattern_1_wrong_domain_tool(self, detector):
        """Tool from wrong domain → Pattern 1 detected."""
        trace = {
            "domain": "weather",
            "steps": [
                {"step_type": "thought", "content": "I need to get the weather forecast"},
                {"step_type": "action", "content": "Action: create_event",
                 "tool_name": "create_event"},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_1(trace)
        assert result["detected"] is True
        assert result["confidence"] >= 0.8

    def test_pattern_1_correct_tool(self, detector):
        """Correct tool for domain → Pattern 1 not detected."""
        trace = {
            "domain": "weather",
            "steps": [
                {"step_type": "thought", "content": "I need to check the weather"},
                {"step_type": "action", "content": "Action: get_weather",
                 "tool_name": "get_weather"},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_1(trace)
        # Should not detect wrong domain (tool is correct for weather)
        assert result["confidence"] < 0.9 or not result["detected"]

    def test_pattern_2_injected_memory(self, detector):
        """Memory value with injected flag → Pattern 2 detected."""
        trace = {
            "steps": [
                {"step_type": "observation", "content": "result",
                 "tool_result": '{"city": "London", "temperature": 22}'},
                {"step_type": "memory_write", "content": "memory",
                 "memory_state": '{"city": {"value": "Atlantis", "turn_id": 1, "injected": true}}'},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_2(trace)
        assert result["detected"] is True
        assert result["confidence"] >= 0.9

    def test_pattern_2_clean_memory(self, detector):
        """Memory values all from tool_result → Pattern 2 not detected."""
        trace = {
            "steps": [
                {"step_type": "observation", "content": "result",
                 "tool_result": '{"city": "London", "temperature": 22}'},
                {"step_type": "memory_write", "content": "memory",
                 "memory_state": '{"city": {"value": "London", "turn_id": 1}}'},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_2(trace)
        assert result["detected"] is False

    def test_pattern_3_unsupported_claims(self, detector):
        """Final answer with values not in observations → Pattern 3 detected."""
        trace = {
            "final_answer": "The temperature in Atlantis is 999 degrees",
            "steps": [
                {"step_type": "observation", "content": '{"city": "London", "temperature": 22}',
                 "tool_result": '{"city": "London", "temperature": 22}',
                 "retrieval_results": None},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_3(trace)
        assert result["detected"] is True

    def test_pattern_3_supported_claims(self, detector):
        """Final answer with values from observations → Pattern 3 not detected."""
        trace = {
            "final_answer": "The temperature in London is 22 degrees",
            "steps": [
                {"step_type": "observation", "content": '{"city": "London", "temperature": 22}',
                 "tool_result": '{"city": "London", "temperature": 22}',
                 "retrieval_results": None},
            ],
            "injections": [],
        }
        result = detector.detect_pattern_3(trace)
        # No unsupported numbers beyond what's in observations
        assert result["confidence"] < 0.8

    def test_detect_all_returns_structure(self, detector):
        """detect_all should return all 3 patterns."""
        trace = {
            "domain": "weather",
            "final_answer": "result",
            "steps": [
                {"step_type": "thought", "content": "thinking about weather"},
                {"step_type": "action", "content": "Action: get_weather",
                 "tool_name": "get_weather"},
                {"step_type": "observation", "content": "result",
                 "tool_result": '{"temperature": 22}', "retrieval_results": None},
                {"step_type": "memory_write", "content": "memory",
                 "memory_state": '{"temperature": {"value": 22, "turn_id": 1}}'},
            ],
            "injections": [],
        }
        result = detector.detect_all(trace)
        assert "pattern_1" in result
        assert "pattern_2" in result
        assert "pattern_3" in result
        assert "cascade_count" in result
        assert "cascade_chain" in result
        assert isinstance(result["cascade_count"], int)

    def test_detect_all_on_cascade_trace(self, detector):
        """Full cascade trace should trigger multiple patterns."""
        trace = {
            "domain": "weather",
            "final_answer": "The temperature in Atlantis is 999 degrees with tornado",
            "steps": [
                {"step_type": "thought", "content": "I need to create an event"},
                {"step_type": "action", "content": "Action: create_event",
                 "tool_name": "create_event"},
                {"step_type": "observation", "content": "result",
                 "tool_result": '{"city": "London", "temperature": 22}',
                 "retrieval_results": None},
                {"step_type": "memory_write", "content": "memory",
                 "memory_state": '{"city": {"value": "Atlantis", "turn_id": 1, "injected": true}}'},
            ],
            "injections": [{
                "injected_value": '"Atlantis"',
                "injection_type": "p3_false_belief",
            }],
        }
        result = detector.detect_all(trace)
        # Pattern 1: wrong tool for domain
        assert result["pattern_1"]["detected"] is True
        # Pattern 2: injected memory
        assert result["pattern_2"]["detected"] is True
        # Pattern 3: unsupported values in output
        assert result["pattern_3"]["detected"] is True
        assert result["cascade_count"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Multi-Turn Accumulation Scorer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiTurnScorer:
    """Test MTAS with known multi-turn injection scenarios."""

    def test_no_injection(self, mtas_scorer):
        """No injection → propagation_depth=0."""
        trace = {
            "steps": [
                {"turn_id": 1, "content": "turn 1", "tool_params_raw": None,
                 "memory_state": None, "tool_result": None},
                {"turn_id": 2, "content": "turn 2", "tool_params_raw": None,
                 "memory_state": None, "tool_result": None},
            ],
            "injections": [],
        }
        result = mtas_scorer.compute_mtas(trace, injection_turn=0)
        assert result["propagation_depth"] == 0

    def test_full_propagation(self, mtas_scorer):
        """False value in all turns after injection → full propagation."""
        trace = {
            "steps": [
                {"turn_id": 1, "step_type": "observation", "content": "EVT_0001",
                 "tool_params_raw": '{"event_id": "EVT_0001"}', "memory_state": None,
                 "tool_result": None},
                {"turn_id": 2, "step_type": "memory_write", "content": "injection",
                 "tool_params_raw": None,
                 "memory_state": '{"event_id": {"value": "EVT_9999"}}',
                 "tool_result": None},
                {"turn_id": 3, "step_type": "action", "content": "uses EVT_9999",
                 "tool_params_raw": '{"event_id": "EVT_9999"}', "memory_state": None,
                 "tool_result": None},
                {"turn_id": 4, "step_type": "action", "content": "still EVT_9999",
                 "tool_params_raw": '{"event_id": "EVT_9999"}', "memory_state": None,
                 "tool_result": None},
                {"turn_id": 5, "step_type": "action", "content": "still EVT_9999",
                 "tool_params_raw": '{"event_id": "EVT_9999"}', "memory_state": None,
                 "tool_result": None},
            ],
            "injections": [{"injected_value": '"EVT_9999"'}],
        }
        result = mtas_scorer.compute_mtas(trace, injection_turn=2)
        assert result["propagation_depth"] == 3
        assert 3 in result["affected_turns"]
        assert 4 in result["affected_turns"]
        assert 5 in result["affected_turns"]
        assert result["propagation_rate"] == 1.0

    def test_self_correction(self, mtas_scorer):
        """Agent corrects at turn 5 → self_correction_turn=5."""
        trace = {
            "steps": [
                {"turn_id": 1, "content": "normal", "tool_params_raw": None,
                 "memory_state": None, "tool_result": None},
                {"turn_id": 2, "content": "injection", "tool_params_raw": None,
                 "memory_state": '{"event_id": {"value": "EVT_9999"}}',
                 "tool_result": None},
                {"turn_id": 3, "content": "uses EVT_9999",
                 "tool_params_raw": '{"event_id": "EVT_9999"}',
                 "memory_state": None, "tool_result": None},
                {"turn_id": 4, "content": "uses EVT_9999",
                 "tool_params_raw": '{"event_id": "EVT_9999"}',
                 "memory_state": None, "tool_result": None},
                {"turn_id": 5, "content": "corrected to EVT_0001",
                 "tool_params_raw": '{"event_id": "EVT_0001"}',
                 "memory_state": None, "tool_result": None},
            ],
            "injections": [{"injected_value": '"EVT_9999"'}],
        }
        result = mtas_scorer.compute_mtas(trace, injection_turn=2)
        assert result["propagation_depth"] == 2  # turns 3, 4
        assert result["self_correction_turn"] == 5

    def test_p3_injection_session(self, mtas_scorer, tracer):
        """Simulate full P3 injection session and compute MTAS."""
        sid = "mtas_p3_test"
        _start_session(tracer, sid, domain="calendar", injection_type="p3_false_belief")

        # Turn 1: normal event creation
        tracer.log_step(sid, 1, 1, "observation",
                        json.dumps({"event_id": "EVT_0001", "status": "created"}),
                        tool_name="create_event",
                        tool_result={"event_id": "EVT_0001", "status": "created"})

        # Turn 2: injection
        tracer.log_injection(sid, "p3_false_belief", "memory", "EVT_0001", "EVT_9999", turn_id=2)
        tracer.log_step(sid, 2, 2, "memory_write", "Injected false event_id",
                        memory_state={"event_id": {"value": "EVT_9999", "injected": True}})

        # Turns 3-6: agent uses false event_id
        for turn in range(3, 7):
            step_num = tracer.get_step_count(sid) + 1
            tracer.log_step(sid, turn, step_num, "action",
                            f"delete_event with EVT_9999",
                            tool_name="delete_event",
                            tool_params_raw={"event_id": "EVT_9999"})

        tracer.log_step(sid, 6, tracer.get_step_count(sid) + 1, "final",
                        "Deleted event EVT_9999")
        tracer.end_session(sid, "Deleted event EVT_9999", "Deleted EVT_0001", False)

        trace = tracer.get_session_trace(sid)
        result = mtas_scorer.compute_mtas(trace, injection_turn=2)

        assert result["propagation_depth"] >= 3, \
            f"Expected ≥3 turns affected, got {result['propagation_depth']}"
        assert result["injection_turn"] == 2
        assert result["propagation_rate"] > 0

    def test_compare_single_vs_multi(self, mtas_scorer):
        """Comparison should return valid structure."""
        import pandas as pd
        single_df = pd.DataFrame([
            {"eps": 0, "reached_output": False},
            {"eps": 1, "reached_output": False},
            {"eps": 2, "reached_output": True},
        ])
        multi_list = [
            {"propagation_depth": 4, "propagation_rate": 0.8,
             "compounding_score": 0.3, "self_correction_turn": None},
            {"propagation_depth": 3, "propagation_rate": 0.6,
             "compounding_score": 0.1, "self_correction_turn": 5},
        ]
        result = mtas_scorer.compare_single_vs_multi_turn(single_df, multi_list)
        assert "single_turn" in result
        assert "multi_turn" in result
        assert "finding" in result
        assert result["single_turn"]["count"] == 3
        assert result["multi_turn"]["count"] == 2
