"""Tests for tracer/trace_schema.py and tracer/trace_logger.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
import pytest
from sqlalchemy import create_engine, inspect

from tracer.trace_schema import Base, SessionRecord, StepRecord, GroundTruthRecord, InjectionRecord
from tracer.trace_logger import TraceLogger


@pytest.fixture
def tracer(tmp_path):
    db_path = str(tmp_path / "test_traces.db")
    t = TraceLogger(db_path)
    t.init_db()
    return t


class TestTraceSchema:
    def test_tables_created(self, tmp_path):
        db_path = str(tmp_path / "schema_test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "sessions" in tables
        assert "steps" in tables
        assert "ground_truth" in tables
        assert "injections" in tables

    def test_sessions_columns(self, tmp_path):
        db_path = str(tmp_path / "schema_cols.db")
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("sessions")}
        expected = {
            "session_id", "task_id", "model", "domain",
            "injection_type", "injection_stage",
            "started_at", "ended_at",
            "final_answer", "ground_truth_answer", "final_correct",
        }
        assert expected.issubset(cols)

    def test_steps_columns(self, tmp_path):
        db_path = str(tmp_path / "schema_steps.db")
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("steps")}
        expected = {
            "step_id", "session_id", "turn_id", "step_number",
            "step_type", "content", "tool_name",
            "tool_params_raw", "tool_params_validated",
            "tool_result", "param_errors",
            "retrieval_query", "retrieval_results",
            "memory_state", "token_count", "timestamp",
        }
        assert expected.issubset(cols)


class TestTraceLogger:
    def test_init_db(self, tracer):
        assert tracer._session is not None

    def test_start_session(self, tracer):
        tracer.start_session("s001", "t001", "gpt-4o", "weather")
        record = tracer.session.get(SessionRecord,"s001")
        assert record is not None
        assert record.task_id == "t001"
        assert record.model == "gpt-4o"
        assert record.domain == "weather"
        assert record.started_at is not None

    def test_end_session(self, tracer):
        tracer.start_session("s002", "t002", "gpt-4o", "calendar")
        tracer.end_session("s002", "Event created", "Event created successfully", True)
        record = tracer.session.get(SessionRecord,"s002")
        assert record.ended_at is not None
        assert record.final_answer == "Event created"
        assert record.final_correct is True

    def test_log_step(self, tracer):
        tracer.start_session("s003", "t003", "gpt-4o", "weather")
        tracer.log_step(
            session_id="s003",
            turn_id=0,
            step_number=1,
            step_type="thought",
            content="I need to check the weather",
            token_count=50,
        )
        steps = tracer.session.query(StepRecord).filter_by(session_id="s003").all()
        assert len(steps) == 1
        assert steps[0].step_type == "thought"
        assert steps[0].content == "I need to check the weather"

    def test_log_step_with_tool(self, tracer):
        tracer.start_session("s004", "t004", "gpt-4o", "weather")
        tracer.log_step(
            session_id="s004",
            turn_id=0,
            step_number=1,
            step_type="action",
            content="Calling get_weather",
            tool_name="get_weather",
            tool_params_raw={"city": "London", "date": "2024-06-15"},
            tool_params_validated={"city": "London", "date": "2024-06-15", "unit": "celsius"},
            tool_result={"temperature": 22, "humidity": 65},
            param_errors=[],
        )
        step = tracer.session.query(StepRecord).filter_by(session_id="s004").first()
        assert step.tool_name == "get_weather"
        assert json.loads(step.tool_params_raw)["city"] == "London"
        assert json.loads(step.tool_result)["temperature"] == 22

    def test_log_retrieval(self, tracer):
        tracer.start_session("s005", "t005", "gpt-4o", "weather")
        tracer.log_retrieval(
            session_id="s005",
            turn_id=0,
            query="weather in London",
            results=[{"text": "London has mild weather"}, {"text": "London rain forecast"}],
            scores=[0.95, 0.82],
        )
        steps = tracer.session.query(StepRecord).filter_by(session_id="s005").all()
        assert len(steps) == 1
        assert steps[0].step_type == "retrieval"
        assert steps[0].retrieval_query == "weather in London"
        results = json.loads(steps[0].retrieval_results)
        assert len(results) == 2
        assert results[0]["score"] == 0.95

    def test_log_memory_write(self, tracer):
        tracer.start_session("s006", "t006", "gpt-4o", "calendar")
        tracer.log_memory_write(
            session_id="s006",
            turn_id=1,
            key="event_id",
            value="EVT_001",
            source_step=3,
        )
        steps = tracer.session.query(StepRecord).filter_by(session_id="s006").all()
        assert len(steps) == 1
        assert steps[0].step_type == "memory_write"
        assert "event_id" in steps[0].content

    def test_log_injection(self, tracer):
        tracer.start_session("s007", "t007", "gpt-4o", "weather")
        tracer.log_injection(
            session_id="s007",
            injection_type="p2_type_mismatch",
            target_stage="tool_use",
            original={"days": 5},
            injected={"days": "five"},
            turn_id=0,
        )
        inj = tracer.session.query(InjectionRecord).filter_by(session_id="s007").first()
        assert inj is not None
        assert inj.injection_type == "p2_type_mismatch"
        assert json.loads(inj.original_value)["days"] == 5
        assert json.loads(inj.injected_value)["days"] == "five"

    def test_get_session_trace(self, tracer):
        tracer.start_session("s008", "t008", "gpt-4o", "medical")
        tracer.log_step("s008", 0, 1, "thought", "Checking patient record")
        tracer.log_step("s008", 0, 2, "action", "get_patient_record", tool_name="get_patient_record")
        tracer.log_step("s008", 0, 3, "observation", '{"name": "Alice"}')
        tracer.log_step("s008", 0, 4, "final", "Patient Alice found")
        tracer.end_session("s008", "Patient Alice found", "Alice Johnson", False)

        trace = tracer.get_session_trace("s008")
        assert trace["session_id"] == "s008"
        assert len(trace["steps"]) == 4
        assert trace["final_correct"] is False
        assert trace["steps"][0]["step_type"] == "thought"
        assert trace["steps"][-1]["step_type"] == "final"

    def test_get_session_trace_nonexistent(self, tracer):
        trace = tracer.get_session_trace("nonexistent")
        assert trace == {}

    def test_export_traces(self, tracer, tmp_path):
        tracer.start_session("s009", "t009", "gpt-4o", "weather")
        tracer.log_step("s009", 0, 1, "thought", "test")
        tracer.end_session("s009", "test answer", "correct", True)

        export_path = str(tmp_path / "export.json")
        tracer.export_traces(export_path)

        with open(export_path) as f:
            data = json.load(f)
        assert len(data) >= 1
        assert data[0]["session_id"] == "s009"

    def test_get_step_count(self, tracer):
        tracer.start_session("s010", "t010", "gpt-4o", "weather")
        assert tracer.get_step_count("s010") == 0
        tracer.log_step("s010", 0, 1, "thought", "step 1")
        tracer.log_step("s010", 0, 2, "action", "step 2")
        assert tracer.get_step_count("s010") == 2

    def test_multiple_sessions(self, tracer):
        for i in range(5):
            sid = f"multi_{i}"
            tracer.start_session(sid, f"task_{i}", "gpt-4o", "weather")
            tracer.log_step(sid, 0, 1, "thought", f"thinking {i}")
            tracer.end_session(sid, f"answer {i}", f"truth {i}", i % 2 == 0)

        sessions = tracer.session.query(SessionRecord).all()
        assert len(sessions) >= 5

    def test_session_with_injection_type(self, tracer):
        tracer.start_session(
            "s011", "t011", "gpt-4o", "weather",
            injection_type="p2_type_mismatch",
            injection_stage="tool_use",
        )
        record = tracer.session.get(SessionRecord,"s011")
        assert record.injection_type == "p2_type_mismatch"
        assert record.injection_stage == "tool_use"
