"""Unit tests for the structured validator (V_core).  Hand-constructed traces, one or more per outcome code,
plus normalisation / extraction / gold-construction tests.  Run:  python3 -I -m pytest -q test_V_validator.py
(gold values below are computed by the copied simulator, i.e. they are what the environment returns)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_core as V  # noqa: E402


def T(query, domain, tid="t"):
    return {"task_id": tid, "domain": domain, "query": query, "correct_tool_sequence": [], "required_tools": []}


def C(tool, params, result=None, errors=None, emitted=None):
    if result is None:
        if tool in V.STATELESS:
            result = V.sim(tool, params)
        else:
            result = {}
    return V.call_record(tool, emitted if emitted is not None else params, params, result, errors or [])


W1 = T("What is the weather in Dubai on 2024-10-12?", "weather")
GW = V.sim("get_weather", {"city": "Dubai", "date": "2024-10-12"})


# ------------------------------------------------------------------ simulator determinism
def test_simulator_deterministic():
    a = V.sim("get_weather", {"city": "Toronto", "date": "2024-07-08"})
    b = V.sim("get_weather", {"city": "Toronto", "date": "2024-07-08"})
    assert a == b
    # value recorded in baseline_day6.db for this exact call (session bl_gpt4o_mini_weather_097_9f0902)
    assert (a["temperature"], a["humidity"], a["condition"]) == (34, 27, "cloudy")


# ------------------------------------------------------------------ outcome codes
def test_correct_read():
    ans = f"The weather in Dubai on 2024-10-12 is {GW['temperature']}°C and {GW['condition']}, humidity {GW['humidity']}%."
    s = V.score_trace(W1, [C("get_weather", {"city": "Dubai", "date": "2024-10-12"})], ans)
    assert s["outcome"] == "correct" and s["CORRECT"] and s["T"] and s["P"] and s["A"]


def test_wrong_params():
    obs = V.sim("get_weather", {"city": "Abu Dhabi", "date": "2024-10-12"})
    ans = f"It is {obs['temperature']}°C and {obs['condition']}."
    s = V.score_trace(W1, [C("get_weather", {"city": "Abu Dhabi", "date": "2024-10-12"})], ans)
    assert s["outcome"] == "wrong_params" and s["T"] and not s["P"]
    assert s["per_slot"]["get_weather.city"] is False and s["per_slot"]["get_weather.date"] is True


def test_wrong_answer_values():
    wrong_t = GW["temperature"] + 7
    ans = f"The weather in Dubai on 2024-10-12 is {wrong_t}°C and {GW['condition']}."
    s = V.score_trace(W1, [C("get_weather", {"city": "Dubai", "date": "2024-10-12"})], ans)
    assert s["outcome"] == "wrong_answer_values" and s["T"] and s["P"] and not s["A"]
    assert "temp" in s["ungrounded"]


def test_injected_observation_propagated_is_wrong_answer():
    # P2-style: agent emitted correct params, executor corrupted the city; agent re-called correctly but reported
    # the corrupted observation's values -> wrong answer values (gold observation != received corrupted one)
    bad = V.sim("get_weather", {"city": "Reykjavik", "date": "2024-10-12"})
    calls = [C("get_weather", {"city": "Reykjavik", "date": "2024-10-12"}, emitted={"city": "Dubai", "date": "2024-10-12"}),
             C("get_weather", {"city": "Dubai", "date": "2024-10-12"})]
    ans = f"It is {bad['temperature']}°C and {bad['condition']} in Dubai."
    s = V.score_trace(W1, calls, ans)
    if bad["temperature"] != GW["temperature"]:
        assert s["outcome"] == "wrong_answer_values" and s["P"] and not s["A"]


def test_fabricated_no_tool_call():
    s = V.score_trace(W1, [], "The weather in Dubai on 2024-10-12 is 31°C and sunny.")
    assert s["outcome"] == "fabricated" and not s["CORRECT"]


def test_fabricated_write_claim_without_execution():
    t = T("Delete event EVT_4986.", "calendar")
    s = V.score_trace(t, [], "Event EVT_4986 deleted.")
    assert s["outcome"] == "fabricated"


def test_refused():
    s = V.score_trace(W1, [], "I'm sorry, I cannot access weather data for that date.")
    assert s["outcome"] == "refused"


def test_incomplete_placeholder():
    s = V.score_trace(W1, [], "[weather data for Dubai on 2024-10-12]")
    assert s["outcome"] == "incomplete"


def test_incomplete_missing_required_tool():
    t = T("What's the weather in Rome on 2024-04-05, and also give me a 5-day forecast?", "weather")
    g = V.sim("get_weather", {"city": "Rome", "date": "2024-04-05"})
    s = V.score_trace(t, [C("get_weather", {"city": "Rome", "date": "2024-04-05"})],
                      f"Rome is {g['temperature']}°C and {g['condition']}.")
    assert s["outcome"] == "incomplete" and not s["T"] and s["T_missing"] == "get_forecast"


def test_schema_error():
    errs = ["Missing required parameter: 'city'", "Unknown parameter: 'param'"]
    c = V.call_record("get_weather", {"param": "Dubai"}, {"param": "Dubai"}, {"error": "; ".join(errs)}, errs)
    s = V.score_trace(W1, [c], "I could not get the weather due to a tool error.")
    assert s["outcome"] == "schema_error"


def test_infra_error():
    s = V.score_trace(W1, [], "ERROR: 404 This model is no longer available")
    assert s["outcome"] == "infra_error"


def test_not_verifiable():
    s = V.score_trace(T("Is it going to rain in Berlin tomorrow?", "weather"), [], "Yes.")
    assert s["outcome"] == "not_verifiable" and s["CORRECT"] is None
    k = {"task_id": "h", "domain": "knowledge", "query": "Use the search_knowledge tool to answer: x?"}
    assert V.score_trace(k, [], "x")["outcome"] == "not_verifiable"


# ------------------------------------------------------------------ write tasks / state
CR = T("Create a meeting called 'Training Session' on 2024-08-12 at 11:00 for 90 minutes.", "calendar")
CRP = {"title": "Training Session", "date": "2024-08-12", "time": "11:00", "duration_minutes": 90}


def test_write_correct():
    c = C("create_event", CRP, {"event_id": "EVT_0007", "status": "created"})
    s = V.score_trace(CR, [c], "The 'Training Session' event has been created (EVT_0007).")
    assert s["outcome"] == "correct" and s["S"]


def test_wrong_state_duplicate_write():
    c1 = C("create_event", CRP, {"event_id": "EVT_0007", "status": "created"})
    c2 = C("create_event", dict(CRP, title="Training"), {"event_id": "EVT_0008", "status": "created"})
    s = V.score_trace(CR, [c1, c2], "The 'Training Session' event has been created.")
    assert s["S"] is False and s["outcome"] == "wrong_state"
    s2 = V.score_trace(CR, [c1, c2], "The 'Training Session' event has been created.", cfg={"state": "lenient"})
    assert s2["S"] is True


def test_write_claim_must_match_state():
    c = C("create_event", CRP, {"event_id": "EVT_0007", "status": "created"})
    s = V.score_trace(CR, [c], "I could not create the event.")
    assert not s["A"] and not s["CORRECT"]


def test_delete_not_found_reported_faithfully_is_correct():
    t = T("Delete event EVT_7883.", "calendar")
    c = C("delete_event", {"event_id": "EVT_7883"}, {"success": False, "error": "Event 'EVT_7883' not found"})
    assert V.score_trace(t, [c], "Event EVT_7883 was not found, so nothing was deleted.")["outcome"] == "correct"
    s = V.score_trace(t, [c], "Event EVT_7883 has been deleted.")
    # claims a successful write although no write succeeded -> fabricated (asserted result without successful execution)
    assert s["outcome"] == "fabricated" and not s["claim_ok"]


def test_conditional_write_slot_busy():
    t = T("Check my schedule for 2024-07-10 and if the 09:00 slot is free, create a '1:1 with Manager' for 30 minutes.",
          "calendar")
    ev = {"date": "2024-07-10", "calendar_id": "primary",
          "events": [{"title": "Budget Review", "time": "09:00", "duration": 45, "event_id": "EVT_0012"}]}
    g = C("get_events", {"date": "2024-07-10"}, ev)
    s = V.score_trace(t, [g], "The 09:00 slot on 2024-07-10 is already booked (Budget Review), so I did not create it.")
    assert s["outcome"] == "correct"
    cr = C("create_event", {"title": "1:1 with Manager", "date": "2024-07-10", "time": "09:00", "duration_minutes": 30},
           {"event_id": "EVT_0099", "status": "created"})
    s2 = V.score_trace(t, [g, cr], "Created the '1:1 with Manager' meeting at 09:00.")
    assert not s2["CORRECT"] and s2["S"] is False


def test_ecommerce_cheaper_is_computed_from_simulator():
    t = T("Compare product PRD_9379 and product PRD_2894, then order 3 of the cheaper one to 321 Maple Ln, Boston, MA 02108.",
          "ecommerce")
    spec = V.build_spec(t)
    p1 = V.sim("get_product", {"product_id": "PRD_9379"})["price_usd"]
    p2 = V.sim("get_product", {"product_id": "PRD_2894"})["price_usd"]
    assert spec.calls[2].params["product_id"] == ("PRD_9379" if p1 <= p2 else "PRD_2894")


# ------------------------------------------------------------------ normalisation
def test_date_time_normalisation():
    assert V.norm_date("July 8, 2024") == "2024-07-08" and V.norm_date("2024/07/08") == "2024-07-08"
    assert V.norm_time("10am") == "10:00" and V.norm_time("1:30 PM") == "13:30"
    assert V.slot_equal("city", "São Paulo", "sao paulo")
    assert not V.slot_equal("city", "São Paulo", "sao paulo", dict(V.DEFAULT_CFG, casefold=False))
    assert V.slot_equal("attendees", ["a@x.com", "b@x.com"], ["B@x.com", "a@x.com"])


def test_fahrenheit_conversion_path():
    t = T("What's the temperature in Tokyo on 2024-10-12 in fahrenheit?", "weather")
    c_obs = V.sim("get_weather", {"city": "Tokyo", "date": "2024-10-12"})
    fval = V.c2f(c_obs["temperature"])
    s = V.score_trace(t, [C("get_weather", {"city": "Tokyo", "date": "2024-10-12"})], f"It is {fval}°F in Tokyo.")
    assert s["P"] and s["A"] and s["CORRECT"]          # celsius call + correct conversion accepted
    s2 = V.score_trace(t, [C("get_weather", {"city": "Tokyo", "date": "2024-10-12"})], f"It is {fval}°F in Tokyo.",
                       cfg={"unit_conv": False})
    assert not s2["P"]


def test_derived_average_grounded():
    t = T("Compare today's weather in Toronto (2024-07-08) with the historical average from 2023-01-01 to 2023-01-07.",
          "weather")
    calls = [C("get_weather", {"city": "Toronto", "date": "2024-07-08"}),
             C("get_historical", {"city": "Toronto", "start_date": "2023-01-01", "end_date": "2023-01-07"})]
    ans = "Today in Toronto is 34°C and cloudy, warmer than the historical average of approximately 16.29°C."
    assert V.score_trace(t, calls, ans)["CORRECT"]
    assert not V.score_trace(t, calls, ans, cfg={"derived": False})["CORRECT"]


def test_extraction_ignores_years_and_ids():
    e = V.extract("April 16, 2026 – 22°C, partly cloudy; order ORD_123 total $45", domain="weather")
    assert (22.0, "C") in e["temp"] and all(abs(v) < 200 for v, _ in e["temp"])
    assert "ORD_123" in e["ids"] and 45.0 in e["money"] and "partly cloudy" in e["condition"]


def test_query_grounded_gold_overrides_taskfile_contradiction():
    # generator bug: easy delete template draws the gold event_id independently of the one in the query
    task = {"task_id": "calendar_010", "domain": "calendar", "query": "Delete event EVT_7883.",
            "correct_tool_sequence": [{"tool": "delete_event", "params": {"event_id": "EVT_7381"}}]}
    spec = V.build_spec(task)
    assert spec.calls[0].params["event_id"] == "EVT_7883"
    assert V.audit_taskfile_vs_query(spec) == [("delete_event.event_id", "EVT_7381", "EVT_7883")]
