"""Tests for agent/tool_executor.py — validates all 9 tools with valid AND invalid params."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.tool_executor import ToolExecutor


@pytest.fixture
def executor():
    return ToolExecutor()


# ── Registry Tests ────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_tools_returns_9(self, executor):
        tools = executor.list_tools()
        assert len(tools) == 9

    def test_list_tools_have_required_fields(self, executor):
        for tool in executor.list_tools():
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_get_tools_for_domain(self, executor):
        for domain in ["weather", "calendar", "medical"]:
            tools = executor.get_tools_for_domain(domain)
            assert len(tools) == 3, f"Expected 3 tools for {domain}, got {len(tools)}"

    def test_unknown_tool(self, executor):
        result = executor.execute("nonexistent_tool", {})
        assert "error" in result["result"]
        assert len(result["validation_errors"]) > 0


# ── Weather Domain Tests ──────────────────────────────────────────────────────

class TestWeatherTools:
    def test_get_weather_valid(self, executor):
        result = executor.execute("get_weather", {"city": "London", "date": "2024-06-15"})
        assert result["validation_errors"] == []
        r = result["result"]
        assert "temperature" in r
        assert "humidity" in r
        assert "condition" in r

    def test_get_weather_fahrenheit(self, executor):
        result = executor.execute("get_weather", {"city": "London", "date": "2024-06-15", "unit": "fahrenheit"})
        assert result["result"]["unit"] == "fahrenheit"

    def test_get_weather_missing_city(self, executor):
        result = executor.execute("get_weather", {"date": "2024-06-15"})
        assert "error" in result["result"]
        assert any("city" in e for e in result["validation_errors"])

    def test_get_weather_invalid_unit(self, executor):
        result = executor.execute("get_weather", {"city": "London", "date": "2024-06-15", "unit": "kelvin"})
        assert "error" in result["result"]

    def test_get_weather_wrong_type(self, executor):
        result = executor.execute("get_weather", {"city": 123, "date": "2024-06-15"})
        assert "error" in result["result"]

    def test_get_weather_deterministic(self, executor):
        r1 = executor.execute("get_weather", {"city": "London", "date": "2024-06-15"})
        r2 = executor.execute("get_weather", {"city": "London", "date": "2024-06-15"})
        assert r1["result"]["temperature"] == r2["result"]["temperature"]

    def test_get_forecast_valid(self, executor):
        result = executor.execute("get_forecast", {"city": "Tokyo", "days": 3})
        assert result["validation_errors"] == []
        assert len(result["result"]["forecast"]) == 3

    def test_get_forecast_with_hourly(self, executor):
        result = executor.execute("get_forecast", {"city": "Tokyo", "days": 1, "include_hourly": True})
        assert "hourly" in result["result"]["forecast"][0]

    def test_get_forecast_days_out_of_range(self, executor):
        result = executor.execute("get_forecast", {"city": "Tokyo", "days": 30})
        assert "error" in result["result"]

    def test_get_forecast_days_wrong_type(self, executor):
        result = executor.execute("get_forecast", {"city": "Tokyo", "days": "three"})
        assert "error" in result["result"]

    def test_get_historical_valid(self, executor):
        result = executor.execute("get_historical", {
            "city": "Paris", "start_date": "2024-01-01", "end_date": "2024-01-03"
        })
        assert result["validation_errors"] == []
        assert len(result["result"]["data"]) == 3

    def test_get_historical_bad_date_order(self, executor):
        result = executor.execute("get_historical", {
            "city": "Paris", "start_date": "2024-01-10", "end_date": "2024-01-01"
        })
        assert "error" in result["result"]

    def test_get_historical_missing_field(self, executor):
        result = executor.execute("get_historical", {"city": "Paris", "start_date": "2024-01-01"})
        assert "error" in result["result"]


# ── Calendar Domain Tests ─────────────────────────────────────────────────────

class TestCalendarTools:
    def test_create_event_valid(self, executor):
        result = executor.execute("create_event", {
            "title": "Team Meeting",
            "date": "2024-07-01",
            "time": "10:00",
            "duration_minutes": 60,
        })
        assert result["validation_errors"] == []
        assert "event_id" in result["result"]

    def test_create_event_with_attendees(self, executor):
        result = executor.execute("create_event", {
            "title": "Review",
            "date": "2024-07-01",
            "time": "14:00",
            "duration_minutes": 30,
            "attendees": ["alice@test.com", "bob@test.com"],
        })
        assert result["result"]["status"] == "created"

    def test_create_event_missing_time(self, executor):
        result = executor.execute("create_event", {
            "title": "Team Meeting",
            "date": "2024-07-01",
            "duration_minutes": 60,
        })
        assert "error" in result["result"]

    def test_create_event_bad_duration(self, executor):
        result = executor.execute("create_event", {
            "title": "Team Meeting",
            "date": "2024-07-01",
            "time": "10:00",
            "duration_minutes": 2000,
        })
        assert "error" in result["result"]

    def test_get_events_valid(self, executor):
        result = executor.execute("get_events", {"date": "2024-07-01"})
        assert result["validation_errors"] == []
        assert "events" in result["result"]

    def test_get_events_missing_date(self, executor):
        result = executor.execute("get_events", {})
        assert "error" in result["result"]

    def test_delete_event_valid(self, executor):
        # First create, then delete
        create_result = executor.execute("create_event", {
            "title": "To Delete",
            "date": "2024-08-01",
            "time": "09:00",
            "duration_minutes": 30,
        })
        eid = create_result["result"]["event_id"]
        del_result = executor.execute("delete_event", {"event_id": eid})
        assert del_result["result"]["success"] is True

    def test_delete_event_not_found(self, executor):
        result = executor.execute("delete_event", {"event_id": "EVT_NONEXISTENT"})
        assert result["result"]["success"] is False

    def test_delete_event_missing_id(self, executor):
        result = executor.execute("delete_event", {})
        assert "error" in result["result"]


# ── Medical Domain Tests ──────────────────────────────────────────────────────

class TestMedicalTools:
    def test_get_patient_valid(self, executor):
        result = executor.execute("get_patient_record", {"patient_id": "P001"})
        assert result["validation_errors"] == []
        r = result["result"]
        assert r["name"] == "Alice Johnson"
        assert "conditions" in r
        assert "medications" in r

    def test_get_patient_specific_fields(self, executor):
        result = executor.execute("get_patient_record", {"patient_id": "P001", "fields": ["name", "dob"]})
        r = result["result"]
        assert "name" in r
        assert "conditions" not in r

    def test_get_patient_not_found(self, executor):
        result = executor.execute("get_patient_record", {"patient_id": "P999"})
        assert "error" in result["result"]

    def test_get_patient_missing_id(self, executor):
        result = executor.execute("get_patient_record", {})
        assert "error" in result["result"]

    def test_drug_interaction_known(self, executor):
        result = executor.execute("check_drug_interaction", {"drug_a": "ibuprofen", "drug_b": "lisinopril"})
        assert result["validation_errors"] == []
        assert result["result"]["interaction"] is True
        assert result["result"]["severity"] == "moderate"

    def test_drug_interaction_none(self, executor):
        result = executor.execute("check_drug_interaction", {"drug_a": "metformin", "drug_b": "lisinopril"})
        assert result["result"]["interaction"] is False

    def test_drug_interaction_unknown_pair(self, executor):
        result = executor.execute("check_drug_interaction", {"drug_a": "aspirin", "drug_b": "vitamin_c"})
        assert result["result"]["severity"] == "unknown"

    def test_drug_interaction_missing_param(self, executor):
        result = executor.execute("check_drug_interaction", {"drug_a": "aspirin"})
        assert "error" in result["result"]

    def test_schedule_appointment_valid(self, executor):
        result = executor.execute("schedule_appointment", {
            "patient_id": "P001",
            "doctor_id": "D001",
            "date": "2024-08-15",
            "appointment_type": "checkup",
        })
        assert result["validation_errors"] == []
        assert "appointment_id" in result["result"]

    def test_schedule_appointment_bad_type(self, executor):
        result = executor.execute("schedule_appointment", {
            "patient_id": "P001",
            "doctor_id": "D001",
            "date": "2024-08-15",
            "appointment_type": "surgery",
        })
        assert "error" in result["result"]

    def test_schedule_appointment_missing_fields(self, executor):
        result = executor.execute("schedule_appointment", {"patient_id": "P001"})
        assert "error" in result["result"]
