"""V_core.py -- AgentProp structured (programmatic) correctness validator, version V1.0.

No LLM, no human input.  Everything is derived from (i) the task query text, (ii) the task-file gold
tool sequence, (iii) a faithful re-implementation of the AgentProp tool simulator (scripts/C_tools.py,
a text copy of agent/tool_executor.py; verified byte-identical to the original repo file), and
(iv) the logged trace (executed tool calls, observations, final answer).

Components (see VALIDATOR_SPEC.txt for the full rubric and justification):
  T  tool correctness   : every required gold call is matched by an executed (schema-valid) call of the
                          same tool (one-to-one assignment) and per-template order constraints hold.
  P  parameter corr.    : every matched executed call agrees with the gold call on every scored slot,
                          after slot normalisation (case/whitespace/accents, date formats, time formats,
                          unordered sets, enum spelling).  Params are those ACTUALLY EXECUTED (post-injection).
  S  state/action       : write tasks only. Multiset of state-changing successful writes == multiset of gold
                          writes (strict, tau-bench-style); gold deletes are satisfied by an executed attempt
                          with the gold id (the simulator decides found/not-found).
  A  answer faithfulness: typed values in the final answer vs the GOLD observation(s):
                          recall  = required key values present; precision = typed value mentions grounded in
                          gold observations / query / derived aggregates; write outcome claims must match S.
  CORRECT = T and P and S and A   (S is vacuously true for read-only tasks; A for write-only tasks reduces to
                                   outcome-claim faithfulness + precision).
"""
import copy
import json
import math
import re
import statistics
import sys
import os
import unicodedata
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C_tools as SIM  # noqa: E402  (our own text copy of the simulator; no repo code imported)

VALIDATOR_VERSION = "V1.0"

# ------------------------------------------------------------------------------------------------
# Configuration (every normalisation choice is a switch so robustness can be measured)
# ------------------------------------------------------------------------------------------------
DEFAULT_CFG = {
    "casefold": True,          # case-insensitive string slots / entity matching
    "accent_fold": True,       # 'Sao Paulo' == 'São Paulo'
    "date_norm": True,         # 'July 8, 2024', '2024/07/08' -> '2024-07-08' (slots + answer)
    "time_norm": True,         # '10am', '10:00 AM' -> '10:00'
    "temp_tol": 1.0,           # absolute tolerance on temperatures (degrees)
    "num_tol": 0.0,            # absolute tolerance on other numbers (humidity %, prices)
    "unit_conv": True,         # F<->C conversion accepted in answers; celsius call + correct F answer passes unit slot
    "derived": True,           # mean/min/max/median/differences of list observations are grounded values
    "precision_required": 1.0, # A requires typed-mention precision >= this (None = ignore precision)
    "state": "strict",         # 'strict' (state multiset equality) | 'lenient' (gold writes subset of executed)
    "gold_source": "query",    # 'query' (query-grounded gold; default) | 'taskfile' (raw correct_tool_sequence)
    "equiv_obs": True,         # observation of an executed call that matches gold after normalisation is gold too
    "cond_synonyms": True,     # 'clear'->sunny, 'rain'->rainy, 'snow'->snowy, 'clouds'->cloudy
    "hist_lenient": False,     # 'last January' accepts any sub-range of a January (default: full month 2023/2024)
    "aux_obs": True,           # observations of extra on-task calls (same entities as query/gold) are admissible
    "recall_required": 1.0,    # A requires key-value recall >= this
    "title_lenient": False,    # 'All Hands Meeting' accepted for 'All Hands' (extra generic tokens only)
}

WRITE_TOOLS = {"create_event", "delete_event", "schedule_appointment", "place_order"}
STORE_DEPENDENT = {"get_events", "create_event", "delete_event"}  # depend on shared in-process calendar store

DATE_RE = r"\d{4}-\d{2}-\d{2}"
TIME_RE = r"\d{2}:\d{2}"


# ------------------------------------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------------------------------------
def fold(s, cfg=DEFAULT_CFG):
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    for ch in "‐‑‒–—−":
        s = s.replace(ch, "-")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    if cfg.get("accent_fold", True):
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = " ".join(s.split()).strip().strip("'\"").strip()
    if cfg.get("casefold", True):
        s = s.casefold()
    return s


_DATE_FMTS = ["%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y",
              "%d %b %Y", "%m/%d/%Y", "%Y.%m.%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]


def norm_date(s, cfg=DEFAULT_CFG):
    s0 = "" if s is None else str(s).strip()
    if not cfg.get("date_norm", True):
        return s0
    s1 = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s0)
    for f in _DATE_FMTS:
        try:
            return datetime.strptime(s1, f).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s0


def norm_time(s, cfg=DEFAULT_CFG):
    s0 = "" if s is None else str(s).strip()
    if not cfg.get("time_norm", True):
        return s0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", s0.lower())
    if not m:
        return s0
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap:
        ap = ap.replace(".", "")
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
    elif m.group(2) is None:
        return s0
    return f"{h:02d}:{mi:02d}"


def slot_equal(slot, gold, got, cfg=DEFAULT_CFG):
    """Normalised slot comparison.  Returns bool."""
    if got is None:
        return False
    if slot in ("date", "start_date", "end_date"):
        return norm_date(got, cfg) == norm_date(gold, cfg)
    if slot == "time":
        return norm_time(got, cfg) == norm_time(gold, cfg)
    if slot in ("attendees",):
        if not isinstance(got, list):
            return False
        return sorted(fold(x, cfg) for x in got) == sorted(fold(x, cfg) for x in gold)
    if slot == "drug_pair":
        return sorted(fold(x, cfg) for x in got) == sorted(fold(x, cfg) for x in gold)
    if slot in ("days", "duration_minutes", "quantity", "max_price"):
        try:
            return float(got) == float(gold) and not isinstance(got, bool)
        except (TypeError, ValueError):
            return False
    if slot == "include_hourly":
        return got is True or (isinstance(got, str) and got.lower() == "true")
    if slot == "appointment_type":
        return fold(str(got).replace("-", "_").replace(" ", "_"), cfg) == fold(gold, cfg)
    if slot in ("event_id", "product_id", "patient_id", "doctor_id"):
        return fold(got, cfg).upper() == fold(gold, cfg).upper()
    if slot == "shipping_address":
        f = lambda x: re.sub(r"[^\w]+", " ", fold(x, cfg)).strip()
        return f(got) == f(gold)
    return fold(got, cfg) == fold(gold, cfg)


# ------------------------------------------------------------------------------------------------
# Simulator replay (stateless tools are deterministic: md5-seeded random.Random, see SPEC)
# ------------------------------------------------------------------------------------------------
STATELESS = {"get_weather", "get_forecast", "get_historical", "get_patient_record",
             "check_drug_interaction", "schedule_appointment", "search_products", "get_product",
             "place_order"}


def sim(tool, params):
    """Replay a stateless tool through the copied simulator (validation + defaults + executor)."""
    assert tool in STATELESS, tool
    schema = SIM.TOOL_SCHEMAS[tool]["parameters"]
    errs = SIM._validate_params(params, schema)
    if errs:
        return {"error": "; ".join(errs)}
    vp = SIM._apply_defaults(params, schema)
    return SIM._EXECUTORS[tool](vp)


def mock_events(date):
    """get_events output on an EMPTY store (deterministic fallback; used only when no received obs)."""
    saved = dict(SIM._calendar_store)
    SIM._calendar_store.clear()
    try:
        return SIM._exec_get_events({"date": date, "calendar_id": "primary"})
    finally:
        SIM._calendar_store.update(saved)


# ------------------------------------------------------------------------------------------------
# Task gold: query-grounded specification
# ------------------------------------------------------------------------------------------------
PATIENTS = SIM.PATIENTS


class GoldCall:
    def __init__(self, tool, params, scored, alts=None, keys=None, write=False, conditional=None):
        self.tool = tool
        self.params = params            # canonical params (used for replay)
        self.scored = scored            # list of slot names scored in P
        self.alts = alts or [params]    # acceptable param variants (ambiguity), first = canonical
        self.keys = keys or []          # answer key requirements: list of (kind, mode)
        self.write = write
        self.conditional = conditional  # None or a string tag (condition resolved per trace)

    def to_dict(self):
        return {"tool": self.tool, "params": self.params, "scored": self.scored,
                "alts": self.alts, "keys": self.keys, "write": self.write, "conditional": self.conditional}


class TaskSpec:
    def __init__(self, task, template, calls, order=(), verifiable=True, reason="", extra=None):
        self.task = task
        self.template = template
        self.calls = calls
        self.order = list(order)         # pairs (i, j): call i must precede call j
        self.verifiable = verifiable
        self.reason = reason
        self.extra = extra or {}
        self.taskfile_mismatch = []      # slots where task-file gold contradicts the query

    @property
    def is_write(self):
        return any(c.write for c in self.calls)


def _gc_weather(city, date, unit=None):
    p = {"city": city, "date": date}
    sc = ["city", "date"]
    if unit:
        p["unit"] = unit
        sc.append("unit")
    keys = [("temp", "all")] if unit else [("temp", "all"), ("condition", "all")]
    return GoldCall("get_weather", p, sc, keys=keys)


def _gc_forecast(city, days, hourly=False):
    p = {"city": city, "days": int(days)}
    sc = ["city", "days"]
    if hourly:
        p["include_hourly"] = True
        sc.append("include_hourly")
    return GoldCall("get_forecast", p, sc, keys=[("list_any", "any")])


def _gc_hist(city, s, e, alts=None):
    p = {"city": city, "start_date": s, "end_date": e}
    return GoldCall("get_historical", p, ["city", "start_date", "end_date"],
                    alts=alts or [p], keys=[("list_any", "any")])


def _gc_record(pid, need):
    return GoldCall("get_patient_record", {"patient_id": pid}, ["patient_id"], keys=[(k, "all") for k in need])


def _gc_drug(a, b):
    return GoldCall("check_drug_interaction", {"drug_a": a, "drug_b": b}, ["drug_pair"],
                    keys=[("interaction", "all")])


def _gc_sched(pid, did, date, atype):
    return GoldCall("schedule_appointment", {"patient_id": pid, "doctor_id": did, "date": date,
                                             "appointment_type": atype},
                    ["patient_id", "doctor_id", "date", "appointment_type"], keys=[("claim", "all")], write=True)


def _gc_create(title, date, time, dur, atts=None):
    p = {"title": title, "date": date, "time": time, "duration_minutes": int(dur)}
    sc = ["title", "date", "time", "duration_minutes"]
    if atts:
        p["attendees"] = atts
        sc.append("attendees")
    return GoldCall("create_event", p, sc, keys=[("claim", "all")], write=True)


def _gc_get_events(date):
    return GoldCall("get_events", {"date": date}, ["date"], keys=[("events", "any")])


def _gc_search(q, cat=None, maxp=None, keys=True):
    p = {"query": q}
    sc = ["query"]
    if cat:
        p["category"] = cat
        sc.append("category")
    if maxp is not None:
        p["max_price"] = int(maxp)
        sc.append("max_price")
    return GoldCall("search_products", p, sc, keys=[("search", "any")] if keys else [])


def _gc_product(pid, keys=True):
    return GoldCall("get_product", {"product_id": pid}, ["product_id"],
                    keys=[("product", "all")] if keys else [])


def _gc_order(pid, qty, addr, pay=None, conditional=None):
    p = {"product_id": pid, "quantity": int(qty), "shipping_address": addr}
    sc = ["product_id", "quantity", "shipping_address"]
    if pay:
        p["payment_method"] = pay
        sc.append("payment_method")
    return GoldCall("place_order", p, sc, keys=[("claim", "all")], write=True, conditional=conditional)


def build_spec(task, cfg=DEFAULT_CFG):
    """Parse the query into a gold specification.  Returns TaskSpec."""
    q = task["query"].strip()
    dom = task["domain"]
    m = None

    def M(pat):
        nonlocal m
        m = re.fullmatch(pat, q)
        return m

    if dom == "knowledge":
        return TaskSpec(task, "knowledge", [], verifiable=False,
                        reason="knowledge (HaluEval) task: empty correct_final_answer, no tool structure")
    # ---------------- weather
    if dom == "weather":
        if M(rf"What is the weather in (?P<c>.+?) on (?P<d>{DATE_RE})\?"):
            return TaskSpec(task, "W1_weather", [_gc_weather(m["c"], m["d"])])
        if M(rf"What's the temperature in (?P<c>.+?) on (?P<d>{DATE_RE}) in fahrenheit\?"):
            return TaskSpec(task, "W2_weather_F", [_gc_weather(m["c"], m["d"], "fahrenheit")])
        if M(r"Give me a (?P<n>\d+)-day forecast for (?P<c>.+?)\."):
            return TaskSpec(task, "W3_forecast", [_gc_forecast(m["c"], m["n"])])
        if M(rf"What events are on (?P<c>.+?)'s weather station for (?P<d>{DATE_RE})\? I mean weather conditions\."):
            return TaskSpec(task, "W4_weather_conditions", [_gc_weather(m["c"], m["d"])])
        if M(r"Show me the hourly forecast for (?P<c>.+?) for the next day\."):
            return TaskSpec(task, "W5_hourly", [_gc_forecast(m["c"], 1, hourly=True)])
        if M(rf"What's the weather in (?P<c>.+?) on (?P<d>{DATE_RE}), and also give me a (?P<n>\d+)-day forecast\?"):
            return TaskSpec(task, "W6_weather_forecast", [_gc_weather(m["c"], m["d"]), _gc_forecast(m["c"], m["n"])])
        if M(rf"Compare today's weather in (?P<c>.+?) \((?P<d>{DATE_RE})\) with the historical average from (?P<s>{DATE_RE}) to (?P<e>{DATE_RE})\."):
            w = _gc_weather(m["c"], m["d"])
            w.keys = [("temp", "all")]
            return TaskSpec(task, "W7_weather_vs_hist", [w, _gc_hist(m["c"], m["s"], m["e"])])
        if M(rf"Get the weather for (?P<c>.+?) on (?P<d>{DATE_RE}) and check if it will change in the next (?P<n>\d+) days\."):
            return TaskSpec(task, "W8_weather_change", [_gc_weather(m["c"], m["d"]), _gc_forecast(m["c"], m["n"])])
        if M(rf"I'm planning a trip to (?P<c>.+?)\. Get today's weather \((?P<d>{DATE_RE})\), a (?P<n>\d+)-day forecast, and historical data for last January to understand climate patterns\."):
            c = m["c"]
            alts = [{"city": c, "start_date": "2024-01-01", "end_date": "2024-01-31"},
                    {"city": c, "start_date": "2023-01-01", "end_date": "2023-01-31"}]
            h = _gc_hist(c, "2024-01-01", "2024-01-31", alts=alts)
            h.scored = ["city", "jan_range"]
            return TaskSpec(task, "W9_trip", [_gc_weather(c, m["d"]), _gc_forecast(c, m["n"]), h])
        if M(rf"Compare the current weather in (?P<c1>.+?) and (?P<c2>.+?) on (?P<d>{DATE_RE}), then get forecasts for both cities for the next (?P<n>\d+) days\."):
            return TaskSpec(task, "W10_two_cities",
                            [_gc_weather(m["c1"], m["d"]), _gc_weather(m["c2"], m["d"]),
                             _gc_forecast(m["c1"], m["n"]), _gc_forecast(m["c2"], m["n"])])
        if M(r"Is it going to rain in (?P<c>.+?) tomorrow\?") or M(r"What was the temperature in (?P<c>.+?) last week\?") \
                or M(r"What's the current weather in (?P<c>.+?)\?"):
            return TaskSpec(task, "WU_relative_date", [], verifiable=False,
                            reason="underspecified: relative date ('tomorrow'/'last week'/'current') with no anchor date "
                                   "in the query, and every weather tool requires an explicit date or range")
    # ---------------- calendar
    if dom == "calendar":
        if M(rf"Create a meeting called '(?P<t>.+)' on (?P<d>{DATE_RE}) at (?P<h>{TIME_RE}) for (?P<n>\d+) minutes\."):
            return TaskSpec(task, "C1_create", [_gc_create(m["t"], m["d"], m["h"], m["n"])])
        if M(rf"What meetings do I have on (?P<d>{DATE_RE})\?") or M(rf"Show me my calendar for (?P<d>{DATE_RE})\."):
            return TaskSpec(task, "C2_get_events", [_gc_get_events(m["d"])])
        if M(rf"Schedule a (?P<n>\d+)-minute '(?P<t>.+)' for (?P<d>{DATE_RE}) at (?P<h>{TIME_RE})\."):
            return TaskSpec(task, "C3_create", [_gc_create(m["t"], m["d"], m["h"], m["n"])])
        if M(r"Delete event (?P<e>EVT_\d+)\."):
            g = GoldCall("delete_event", {"event_id": m["e"]}, ["event_id"], keys=[("claim", "all")], write=True)
            return TaskSpec(task, "C4_delete", [g])
        if M(rf"Create a '(?P<t>.+)' meeting on (?P<d>{DATE_RE}) at (?P<h>{TIME_RE}) for (?P<n>\d+) minutes with (?P<a1>\S+) and (?P<a2>\S+), then show me all events for that day\."):
            return TaskSpec(task, "C5_create_then_list",
                            [_gc_create(m["t"], m["d"], m["h"], m["n"], [m["a1"], m["a2"]]), _gc_get_events(m["d"])],
                            order=[(0, 1)])
        if M(rf"Check my schedule for (?P<d>{DATE_RE}) and if the (?P<h>{TIME_RE}) slot is free, create a '(?P<t>.+)' for (?P<n>\d+) minutes\."):
            g0 = _gc_get_events(m["d"])
            g0.keys = []
            g1 = _gc_create(m["t"], m["d"], m["h"], m["n"])
            g1.conditional = "slot_free"
            return TaskSpec(task, "C6_check_then_create", [g0, g1], order=[(0, 1)],
                            extra={"slot_time": m["h"], "date": m["d"]})
        if M(rf"I need to reorganize my (?P<d>{DATE_RE}) schedule\. First check what's there, then delete event (?P<e>EVT_\d+), and create a new '(?P<t>.+)' at (?P<h>{TIME_RE}) for (?P<n>\d+) minutes with (?P<a1>[^,\s]+), (?P<a2>[^,\s]+), and (?P<a3>[^,\s]+)\."):
            g0 = _gc_get_events(m["d"])
            g0.keys = []
            g1 = GoldCall("delete_event", {"event_id": m["e"]}, ["event_id"], keys=[("claim", "all")], write=True)
            g2 = _gc_create(m["t"], m["d"], m["h"], m["n"], [m["a1"], m["a2"], m["a3"]])
            return TaskSpec(task, "C7_reorganize", [g0, g1, g2], order=[(0, 1), (0, 2)])
        if M(rf"Create three meetings on (?P<d>{DATE_RE}): '(?P<t1>.+?)' at (?P<h1>{TIME_RE}) for (?P<n1>\d+) min, '(?P<t2>.+?)' at (?P<h2>{TIME_RE}) for (?P<n2>\d+) min, and '(?P<t3>.+?)' at (?P<h3>{TIME_RE}) for (?P<n3>\d+) min\."):
            d = m["d"]
            return TaskSpec(task, "C8_three_creates",
                            [_gc_create(m["t1"], d, m["h1"], m["n1"]), _gc_create(m["t2"], d, m["h2"], m["n2"]),
                             _gc_create(m["t3"], d, m["h3"], m["n3"])])
        if M(rf"Cancel my (?P<t>.+?) on (?P<d>{DATE_RE})\."):
            g0 = _gc_get_events(m["d"])
            g0.keys = []
            g1 = GoldCall("delete_event", {"event_id": None}, ["event_id"], keys=[("claim", "all")], write=True,
                          conditional="title_present")
            return TaskSpec(task, "C9_cancel_by_title", [g0, g1], order=[(0, 1)],
                            extra={"title": m["t"], "date": m["d"]})
        if M(r"Create a recurring weekly (?P<t>.+?)\.") or M(rf"Schedule a (?P<t>.+?) for next Monday at (?P<h>{TIME_RE})\."):
            return TaskSpec(task, "CU_underspecified", [], verifiable=False,
                            reason="underspecified: no explicit date ('next Monday' has no anchor / recurring) and no "
                                   "duration, both required by create_event; simulator has no recurrence support")
    # ---------------- medical
    if dom == "medical":
        if M(r"Get the medical record for patient (?P<p>P\d{3})\."):
            return TaskSpec(task, "M1_record", [_gc_record(m["p"], ["name", "conditions", "medications"])])
        if M(r"Check if (?P<a>[a-z]+) and (?P<b>[a-z]+) interact with each other\.") or \
                M(r"Check if (?P<a>[a-z]+) interacts with (?P<b>[a-z]+)\."):
            return TaskSpec(task, "M2_interaction", [_gc_drug(m["a"], m["b"])])
        if M(rf"Schedule a (?P<t>\w+) appointment for patient (?P<p>P\d{{3}}) with doctor (?P<doc>D\d{{3}}) on (?P<d>{DATE_RE})\."):
            return TaskSpec(task, "M3_schedule", [_gc_sched(m["p"], m["doc"], m["d"], m["t"])])
        if M(r"What conditions does patient (?P<p>P\d{3}) have\?"):
            return TaskSpec(task, "M4_conditions", [_gc_record(m["p"], ["conditions"])])
        if M(r"What medications is patient (?P<p>P\d{3}) currently taking\?") or \
                M(r"Look up patient (?P<p>P\d{3})'s medication list\."):
            return TaskSpec(task, "M5_medications", [_gc_record(m["p"], ["medications"])])
        if M(r"Get patient (?P<p>P\d{3})'s record and check if their medications (?P<a>[a-z]+) and (?P<b>[a-z]+) have any interactions\."):
            r = _gc_record(m["p"], [])
            d = _gc_drug(m["a"], m["b"])
            meds = PATIENTS[m["p"]]["medications"]
            # 'their medications A and B': A/B are often NOT the patient's drugs (generator draws them at random);
            # checking the patient's actual medication pair is an equally valid reading of the query.
            if len(meds) >= 2 and sorted([m["a"], m["b"]]) != sorted(meds[:2]):
                d.alts = [d.params, {"drug_a": meds[0], "drug_b": meds[1]}]
            return TaskSpec(task, "M6_record_interaction", [r, d])
        if M(rf"Look up patient (?P<p>P\d{{3}})'s conditions and schedule a (?P<t>\w+) with doctor (?P<doc>D\d{{3}}) on (?P<d>{DATE_RE})\."):
            return TaskSpec(task, "M7_conditions_schedule",
                            [_gc_record(m["p"], ["conditions"]), _gc_sched(m["p"], m["doc"], m["d"], m["t"])])
        if M(rf"Review patient (?P<p>P\d{{3}})'s full record, check interactions between all their medications, and schedule a (?P<t>\w+) follow-up with (?P<doc>D\d{{3}}) on (?P<d>{DATE_RE})\."):
            meds = PATIENTS[m["p"]]["medications"]
            calls = [_gc_record(m["p"], [])]
            order = []
            if len(meds) >= 2:
                for i in range(len(meds)):
                    for j in range(i + 1, len(meds)):
                        calls.append(_gc_drug(meds[i], meds[j]))
                        order.append((0, len(calls) - 1))
            calls.append(_gc_sched(m["p"], m["doc"], m["d"], m["t"]))
            return TaskSpec(task, "M8_full_review", calls, order=order)
        if M(rf"Patient (?P<p>P\d{{3}}) is being prescribed (?P<x>[a-z]+)\. Check their current medications, verify no interactions with (?P=x), and schedule a specialist appointment with (?P<doc>D\d{{3}}) on (?P<d>{DATE_RE})\."):
            meds = [x for x in PATIENTS[m["p"]]["medications"] if x != m["x"]]
            calls = [_gc_record(m["p"], [])]
            order = []
            for md in meds:
                calls.append(_gc_drug(m["x"], md))
                order.append((0, len(calls) - 1))
            calls.append(_gc_sched(m["p"], m["doc"], m["d"], "specialist"))
            return TaskSpec(task, "M9_new_prescription", calls, order=order)
        if M(r"Schedule a (?P<t>\w+) appointment for patient (?P<p>P\d{3})\."):
            return TaskSpec(task, "MU_schedule_missing_slots", [], verifiable=False,
                            reason="underspecified: doctor_id and date are required by schedule_appointment but absent "
                                   "from the query")
        if M(r"What are the side effects of (?P<x>[a-z]+)\?"):
            return TaskSpec(task, "MU_side_effects", [], verifiable=False,
                            reason="no tool in the medical toolset returns side effects; no gold observation exists")
    # ---------------- ecommerce
    if dom == "ecommerce":
        A = r"(?P<addr>\d+ [^,]+, [^,]+, [A-Z]{2} \d{5})"
        if M(r"Search the catalog for '(?P<q>.+)'\."):
            return TaskSpec(task, "E1_search", [_gc_search(m["q"])])
        if M(r"Find (?P<c>\w+) items matching '(?P<q>.+)' under \$(?P<x>\d+)\."):
            return TaskSpec(task, "E2_search_filtered", [_gc_search(m["q"], m["c"], m["x"])])
        if M(r"Get details for product (?P<p>PRD_\d+)\.") or M(r"Show me info on product (?P<p>PRD_\d+)\."):
            return TaskSpec(task, "E3_product", [_gc_product(m["p"])])
        if M(rf"Order (?P<n>\d+) unit of product (?P<p>PRD_\d+) to {A}\."):
            return TaskSpec(task, "E4_order", [_gc_order(m["p"], m["n"], m["addr"])])
        if M(r"Search for '(?P<q>.+)' in (?P<c>\w+), then get details on the first result\."):
            first = sim("search_products", {"query": m["q"], "category": m["c"]})["results"][0]["product_id"]
            return TaskSpec(task, "E5_search_first", [_gc_search(m["q"], m["c"], keys=False), _gc_product(first)],
                            order=[(0, 1)])
        if M(rf"Get details on product (?P<p>PRD_\d+), then if it is in stock order (?P<n>\d+) to {A}\."):
            return TaskSpec(task, "E6_product_if_stock",
                            [_gc_product(m["p"], keys=False), _gc_order(m["p"], m["n"], m["addr"], conditional="in_stock")],
                            order=[(0, 1)])
        if M(rf"Search for '(?P<q>.+)' in (?P<c>\w+), and order (?P<n>\d+) of product (?P<p>PRD_\d+) to {A}\."):
            return TaskSpec(task, "E7_search_order",
                            [_gc_search(m["q"], m["c"], keys=False), _gc_order(m["p"], m["n"], m["addr"])])
        if M(rf"I want to buy a .+?\. Search for '(?P<q>.+)' in (?P<c>\w+), check details on product (?P<p>PRD_\d+), and if it's in stock and costs under \$(?P<x>\d+), order (?P<n>\d+) to {A} with (?P<pay>\w+)\."):
            return TaskSpec(task, "E8_buy_if",
                            [_gc_search(m["q"], m["c"], keys=False), _gc_product(m["p"], keys=False),
                             _gc_order(m["p"], m["n"], m["addr"], m["pay"], conditional="in_stock_under")],
                            order=[(1, 2)], extra={"limit": int(m["x"])})
        if M(rf"Compare product (?P<p1>PRD_\d+) and product (?P<p2>PRD_\d+), then order (?P<n>\d+) of the cheaper one to {A}\."):
            pr1 = sim("get_product", {"product_id": m["p1"]})["price_usd"]
            pr2 = sim("get_product", {"product_id": m["p2"]})["price_usd"]
            cheaper = m["p1"] if pr1 <= pr2 else m["p2"]
            o = _gc_order(cheaper, m["n"], m["addr"])
            if pr1 == pr2:
                o.alts = [o.params, dict(o.params, product_id=m["p2"])]
            return TaskSpec(task, "E9_compare_cheaper", [_gc_product(m["p1"], keys=False), _gc_product(m["p2"], keys=False), o],
                            order=[(0, 2), (1, 2)])
        if M(rf"Find '(?P<q>.+)' items under \$(?P<x>\d+) in (?P<c>\w+), inspect product (?P<p>PRD_\d+), and order (?P<n>\d+) units to {A} with (?P<pay>\w+)\."):
            return TaskSpec(task, "E10_find_inspect_order",
                            [_gc_search(m["q"], m["c"], m["x"], keys=False), _gc_product(m["p"], keys=False),
                             _gc_order(m["p"], m["n"], m["addr"], m["pay"])])
    return TaskSpec(task, "UNPARSED", [], verifiable=False, reason="query matches no known template")


def taskfile_spec(task):
    """Raw task-file gold (for the gold_source='taskfile' robustness arm and mismatch audit)."""
    calls = []
    for c in task.get("correct_tool_sequence", []):
        p = dict(c["params"])
        tool = c["tool"]
        if tool not in SIM.TOOL_SCHEMAS:
            return None
        req = SIM.TOOL_SCHEMAS[tool]["parameters"]["required"]
        sc = [k for k in p if k in req or k in ("unit", "include_hourly", "attendees", "category", "max_price",
                                                  "payment_method")]
        if tool == "check_drug_interaction":
            sc = ["drug_pair"]
        calls.append(GoldCall(tool, p, sc, write=tool in WRITE_TOOLS))
    return calls


def audit_taskfile_vs_query(spec):
    """List slot disagreements between the task-file gold sequence and the query-grounded gold."""
    tf = taskfile_spec(spec.task)
    out = []
    if tf is None or not spec.verifiable:
        return out
    tf_by_tool = {}
    for c in tf:
        tf_by_tool.setdefault(c.tool, []).append(c)
    qg_tools = sorted(c.tool for c in spec.calls if not c.conditional)
    tf_tools = sorted(c.tool for c in tf)
    if qg_tools != tf_tools and sorted(c.tool for c in spec.calls) != tf_tools:
        out.append(("tool_multiset", tf_tools, sorted(c.tool for c in spec.calls)))
    used = set()
    for g in spec.calls:
        cands = [c for c in tf_by_tool.get(g.tool, []) if id(c) not in used]
        if not cands:
            continue
        best, bestbad = None, None
        for c in cands:
            bad = [s for s in g.scored if s not in ("drug_pair", "jan_range") and s in g.params and
                   g.params[s] is not None and not slot_equal(s, g.params[s], c.params.get(s))]
            if g.tool == "check_drug_interaction":
                if not slot_equal("drug_pair", [g.params["drug_a"], g.params["drug_b"]],
                                  [c.params.get("drug_a", ""), c.params.get("drug_b", "")]):
                    bad = ["drug_pair"]
            if best is None or len(bad) < len(bestbad):
                best, bestbad = c, bad
        used.add(id(best))
        for s in bestbad:
            gv = [g.params["drug_a"], g.params["drug_b"]] if s == "drug_pair" else g.params.get(s)
            tv = [best.params.get("drug_a"), best.params.get("drug_b")] if s == "drug_pair" else best.params.get(s)
            out.append((g.tool + "." + s, tv, gv))
    return out


# ------------------------------------------------------------------------------------------------
# Answer value extraction (deterministic regex parsing)
# ------------------------------------------------------------------------------------------------
CONDITIONS = SIM.CONDITIONS
COND_SYNONYMS = [  # (regex, canonical); order matters ('partly cloudy' before 'cloudy')
    (r"partly[\s-]+cloudy|partly[\s-]+sunny|partial(?:ly)? cloud", "partly cloudy"),
    (r"overcast", "overcast"),
    (r"cloudy", "cloudy"),
    (r"sunny", "sunny"),
    (r"rainy", "rainy"),
    (r"snowy", "snowy"),
]
COND_SYNONYMS_LENIENT = [
    (r"clouds|cloud cover", "cloudy"),
    (r"clear(?:\s+skies?)?|sunshine", "sunny"),
    (r"rain(?:ing|fall|s)?|showers?", "rainy"),
    (r"snow(?:ing|fall|s)?", "snowy"),
]
OTHER_WEATHER = r"thunder(?:storms?)?|storms?|drizzle|fog(?:gy)?|mist(?:y)?|haz(?:e|y)|hail|sleet|windy|humid and hot|tornado"
NEG_BEFORE = re.compile(r"(?:\bno\b|\bnot\b|n't\b|\bwithout\b|\bnor\b|\bfree of\b)\W+(?:\w+\W+){0,2}$", re.I)

DRUG_VOCAB = sorted(set(["albuterol", "amlodipine", "aspirin", "atorvastatin", "cetirizine", "fluticasone", "ibuprofen",
                         "levothyroxine", "lisinopril", "metformin", "metoprolol", "omeprazole", "sertraline",
                         "sumatriptan", "warfarin", "acetaminophen", "glipizide", "losartan", "naproxen",
                         "fluoxetine", "rizatriptan", "ipratropium", "heparin", "simvastatin", "loratadine",
                         "budesonide", "insulin", "prednisone", "amoxicillin", "gabapentin", "hydrochlorothiazide",
                         "clopidogrel", "montelukast", "escitalopram", "pantoprazole", "furosemide"]))
COND_VOCAB = sorted(set(["hypertension", "diabetes type 2", "type 2 diabetes", "diabetes", "asthma", "migraine",
                         "anxiety", "arthritis", "high cholesterol", "allergies", "depression", "copd",
                         "heart disease", "obesity", "hypothyroidism", "gerd", "osteoporosis",
                         "chronic kidney disease", "high blood pressure"]), key=len, reverse=True)
COND_CANON = {"type 2 diabetes": "diabetes type 2", "high blood pressure": "hypertension"}
PRODUCT_NAMES = sorted({n for v in SIM.ECOM_NAMES.values() for n in v}, key=len, reverse=True)
EVENT_TITLE_VOCAB = sorted(["Team Standup", "Sprint Review", "1:1 with Manager", "Design Review", "Client Call",
                            "Lunch Meeting", "Board Meeting", "Code Review Session", "Product Demo", "Budget Review",
                            "Training Session", "Interview", "Strategy Meeting", "All Hands", "Retrospective",
                            "Lunch Break", "Project Review", "1:1 Meeting", "Code Review", "Workshop", "Team Meeting"],
                           key=len, reverse=True)

ID_RE = re.compile(r"\b(EVT_MOCK_\d+|EVT_\d+|APPT_\d+|ORD_\d+|PRD_\d+)\b", re.I)
MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")
PCT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
HUM_RE = re.compile(r"humidity\s*(?:level\s*)?(?:of|is|at|:|around|about|will be|was)?\s*(?:about|around|approximately|~)?\s*(\d{1,3})(?!\s*°)", re.I)
NUM = r"[-−–]?\d+(?:\.\d+)?"
TEMP_UNIT_RE = re.compile(
    rf"(?<![\w.$])({NUM})(?:\s*(?:to|and|-|–|/)\s*({NUM}))?\s*(?:°\s*([CF])\b|°|º\s*([CF])?|\s*degrees?(?:\s+(celsius|fahrenheit|c|f)\b)?|\s+(celsius|fahrenheit)\b|\s?([CF])\b(?![\w'’]))",
    re.I)
TEMP_WORD_RE = re.compile(
    rf"\b(?:temp(?:erature)?s?|highs?|lows?|average)\b[^.\d\n]{{0,25}}?(?:of|is|was|at|around|about|approximately|be|reach(?:ing)?|:|=|~)\s*(?:about|around|approximately|~)?\s*({NUM})(?!\s*(?:%|percent|days?|-day|hours?|am|pm|:\d))",
    re.I)
ISO_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
NAT_DATE_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
                         r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")
TIME_ANS_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?|\b(\d{1,2})\s*([ap]\.?m\.?)", re.I)
SEVERITY_RE = re.compile(r"\b(high|severe|serious|major|moderate|mild|minor|low)\b(?:[\s-]+(?:severity|risk|level|interaction))?", re.I)

REFUSAL_RE = re.compile(
    r"(cannot reliably answer|max steps reached|^error:|unable to|not able to|could not|couldn't|can't|cannot|"
    r"can not|please provide|please specify|need (?:more|additional) (?:information|details)|"
    r"could you (?:please )?(?:provide|specify|clarify)|i (?:do not|don't) have (?:access|enough)|"
    r"not (?:currently )?(?:possible|available)|no (?:tool|data|information) (?:is )?available|failed to|encountered an error)",
    re.I)
INFRA_RE = re.compile(r"^\s*ERROR:\s", re.I)  # harness/API failure text written by the runner, not by the model
PLACEHOLDER_RE = re.compile(r"^\s*\[[^\]]*\]\s*\.?\s*$")
SUCCESS_RE = re.compile(
    r"\b(has been|have been|was|were|is now|are now|successfully|is|are)\s+(?:successfully\s+)?"
    r"(created|scheduled|booked|placed|deleted|removed|cancel+ed|added|confirmed|set up)\b|"
    r"\b(successfully|i(?:'ve| have)?)\s+(created|scheduled|booked|placed|deleted|removed|cancel+ed|added|ordered|confirmed)\b|"
    r"\b(order (?:id|number)|appointment id|event id|confirmation)\b|\b(done|all set|completed)\b|"
    r"^\s*(?:event|meeting|appointment|order)?\s*[^.]{0,80}?\b(created|scheduled|booked|placed|deleted|cancel+ed|added|confirmed)\b",
    re.I)
FAIL_RE = re.compile(
    r"\b(not found|does not exist|doesn't exist|no (?:such|matching) event|could not|couldn't|unable to|failed|"
    r"cannot|can't|was not|wasn't|were not|weren't|not (?:been )?(?:created|scheduled|deleted|placed|ordered|booked)|"
    r"no event|did not|didn't|error|out of stock|not in stock|unavailable|exceeds|over (?:the|your) (?:budget|limit)|"
    r"more than \$|above \$|not (?:under|below|less than)|isn't (?:under|below)|is not free|not free|already (?:booked|occupied|taken|scheduled)|"
    r"occupied|conflict|is booked|busy|taken|not available|isn't available)\b", re.I)
NO_EVENTS_RE = re.compile(r"\b(no (?:events|meetings|appointments)|nothing scheduled|calendar is (?:empty|clear|free)|"
                          r"(?:don't|do not) have any (?:events|meetings))\b", re.I)
INTERACT_NEG_RE = re.compile(
    r"\b(no (?:known |significant |reported |documented )?(?:drug[- ])?interactions?|(?:do|does|did) not (?:have any |have a )?(?:known )?interact|"
    r"don't interact|doesn't interact|not (?:known to )?interact|no interaction data|no data (?:is )?available|"
    r"interaction (?:data )?(?:is )?(?:not available|unknown)|severity (?:is )?(?:unknown|none)|safe(?:ly)? (?:to (?:be )?(?:use|take)n? )?together|"
    r"commonly used together safely|without (?:any )?interaction|no interactions? (?:were |was )?found|not interact)\b", re.I)
INTERACT_POS_RE = re.compile(
    r"\b((?:there is|there's|have|has|with) an? (?:\w+ )?interaction|(?:do|does) interact|interact with each other(?! ?(?:\?|safely))|"
    r"(?:moderate|high|severe|serious|major)[\s-]+(?:severity|risk|interaction)|risk of|may reduce|can reduce|"
    r"increased risk|interaction (?:was |is )?(?:found|detected|identified|present)|potential interaction)\b", re.I)


def _f(x):
    return float(str(x).replace("−", "-").replace("–", "-"))


def extract(answer, cfg=DEFAULT_CFG, domain=None):
    """Typed value mentions in an answer.  Returns dict kind -> list.
    Weather-typed mentions (temperature, humidity, sky condition) are only extracted for weather tasks."""
    out = _extract(answer, cfg)
    if domain is not None and domain != "weather":
        for k in ("temp", "humidity", "condition", "other_weather"):
            out[k] = []
    return out


def _extract(answer, cfg=DEFAULT_CFG):
    a = unicodedata.normalize("NFKC", answer or "")
    out = {"temp": [], "humidity": [], "condition": [], "other_weather": [], "ids": [], "money": [], "dates": [],
           "times": [], "drugs": [], "med_conditions": [], "severity": [], "products": [], "event_titles": []}
    spans = []
    # temperatures with an explicit unit / degree sign
    for mm in TEMP_UNIT_RE.finditer(a):
        unit_tok = (mm.group(3) or mm.group(4) or mm.group(5) or mm.group(6) or mm.group(7) or "")
        unit = unit_tok[:1].upper() if unit_tok else None
        if mm.group(7) and not re.search(r"°|degree", a[max(0, mm.start() - 3):mm.end() + 2], re.I):
            # bare 'C'/'F' after a number: accept only if it looks like a temperature context
            ctx = a[max(0, mm.start() - 40):mm.start()].lower()
            if not re.search(r"temp|high|low|weather|°|degree|average", ctx):
                continue
        vals = [_f(mm.group(1))] + ([_f(mm.group(2))] if mm.group(2) else [])
        for v in vals:
            if abs(v) <= 150:  # drop years / ids captured by range syntax ('2026 - 22°C')
                out["temp"].append((v, unit))
        spans.append(mm.span())
    for mm in TEMP_WORD_RE.finditer(a):
        if any(s <= mm.start(1) < e for s, e in spans):
            continue
        v = _f(mm.group(1))
        if 1900 <= abs(v) <= 2100:
            continue
        out["temp"].append((v, None))
    for mm in PCT_RE.finditer(a):
        out["humidity"].append(_f(mm.group(1)))
    for mm in HUM_RE.finditer(a):
        v = _f(mm.group(1))
        if v not in out["humidity"]:
            out["humidity"].append(v)
    low = a.lower()
    taken = []
    syn = COND_SYNONYMS + (COND_SYNONYMS_LENIENT if cfg.get("cond_synonyms", True) else [])
    for pat, canon in syn:
        for mm in re.finditer(rf"\b(?:{pat})\b", low):
            if any(s <= mm.start() < e for s, e in taken):
                continue
            taken.append(mm.span())
            if NEG_BEFORE.search(low[max(0, mm.start() - 30):mm.start()]):
                continue
            out["condition"].append(canon)
    for mm in re.finditer(rf"\b(?:{OTHER_WEATHER})\b", low):
        if NEG_BEFORE.search(low[max(0, mm.start() - 30):mm.start()]):
            continue
        out["other_weather"].append(mm.group(0))
    out["ids"] = [x.upper() for x in ID_RE.findall(a)]
    out["money"] = [_f(x.replace(",", "")) for x in MONEY_RE.findall(a)]
    out["money"] += [_f(x) for x in re.findall(r"\b(?:price|cost|total)(?:_usd)?\"?\s*[:=]\s*(\d+(?:\.\d+)?)", a, re.I)]
    out["money"] += [_f(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:USD|dollars)\b", a, re.I)]
    for y, mo, d in ISO_DATE_RE.findall(a):
        out["dates"].append(f"{y}-{mo}-{d}")
    if cfg.get("date_norm", True):
        for mon, day, yr in NAT_DATE_RE.findall(a):
            mon3 = mon[:3].title()
            try:
                if yr:
                    out["dates"].append(datetime.strptime(f"{mon3} {int(day)} {yr}", "%b %d %Y").strftime("%Y-%m-%d"))
                else:
                    out["dates"].append("*-" + datetime.strptime(f"{mon3} {int(day)} 2000", "%b %d %Y").strftime("%m-%d"))
            except ValueError:
                pass
    for mm in TIME_ANS_RE.finditer(a):
        if mm.group(1):
            t = f"{mm.group(1)}:{mm.group(2)}" + (f" {mm.group(3)}" if mm.group(3) else "")
        else:
            t = f"{mm.group(4)} {mm.group(5)}"
        out["times"].append(norm_time(t, cfg) if cfg.get("time_norm", True) else t)
    for d in DRUG_VOCAB:
        if re.search(rf"\b{d}\b", low):
            out["drugs"].append(d)
    tmp = low
    for c in COND_VOCAB:
        if re.search(rf"\b{re.escape(c)}\b", tmp):
            out["med_conditions"].append(COND_CANON.get(c, c))
            tmp = re.sub(rf"\b{re.escape(c)}\b", " ", tmp)
    for mm in SEVERITY_RE.finditer(a):
        ctx = a[max(0, mm.start() - 60):mm.end() + 60].lower()
        if "interact" in ctx or "severity" in ctx or "risk" in ctx:
            out["severity"].append(_sev(mm.group(1)))
    for n in PRODUCT_NAMES:
        if re.search(rf"\b{re.escape(n.lower())}\b", low):
            out["products"].append(n.lower())
    tl = low
    for n in EVENT_TITLE_VOCAB:
        pat = rf"(?<![\w]){re.escape(n.lower())}(?![\w])"
        if re.search(pat, tl):
            out["event_titles"].append(fold(n))
            tl = re.sub(pat, " ", tl)
    return out


def _sev(w):
    w = w.lower()
    return {"severe": "high", "serious": "high", "major": "high", "mild": "low", "minor": "low"}.get(w, w)


def c2f(c):
    return int(c * 9 / 5 + 32)


def f2c(f):
    return (f - 32) * 5 / 9


# ------------------------------------------------------------------------------------------------
# Grounded value sets from observations
# ------------------------------------------------------------------------------------------------
def _temps_of(obs):
    """(list of temperatures, unit) contained in a weather observation."""
    if not isinstance(obs, dict):
        return [], "C"
    unit = "F" if obs.get("unit") == "fahrenheit" else "C"
    t = []
    if "temperature" in obs:
        t.append(obs["temperature"])
    for e in obs.get("forecast", []) or []:
        if "temperature" in e:
            t.append(e["temperature"])
        for h in e.get("hourly", []) or []:
            t.append(h.get("temp"))
    for e in obs.get("data", []) or []:
        t.append(e.get("temperature"))
    return [x for x in t if isinstance(x, (int, float))], unit


def grounded_from_obs(obs, cfg=DEFAULT_CFG):
    """Grounded typed values from one observation dict."""
    g = {"temp": set(), "temp_lists": [], "humidity": set(), "condition": set(), "ids": set(), "money": set(),
         "dates": set(), "times": set(), "drugs": set(), "med_conditions": set(), "severity": set(),
         "products": set(), "event_titles": set(), "event_ids": set()}
    if not isinstance(obs, dict):
        return g
    temps, unit = _temps_of(obs)
    lists = []
    if obs.get("forecast"):
        lists.append([e["temperature"] for e in obs["forecast"]])
        hrs = [h["temp"] for e in obs["forecast"] for h in (e.get("hourly") or [])]
        if hrs:
            lists.append(hrs)
    if obs.get("data"):
        lists.append([e["temperature"] for e in obs["data"]])
    for t in temps:
        g["temp"].add((float(t), unit))
    for L in lists:
        g["temp_lists"].append((L, unit))
    if "humidity" in obs:
        g["humidity"].add(float(obs["humidity"]))
    if "condition" in obs:
        g["condition"].add(obs["condition"])
    for e in obs.get("forecast", []) or []:
        if "condition" in e:
            g["condition"].add(e["condition"])
        if "date" in e:
            g["dates"].add(e["date"])
    for e in obs.get("data", []) or []:
        g["dates"].add(e.get("date"))
    for k in ("date",):
        if k in obs:
            g["dates"].add(str(obs[k]))
    if "period" in obs:
        for d in re.findall(DATE_RE, obs["period"]):
            g["dates"].add(d)
    for k in ("event_id", "appointment_id", "order_id", "product_id"):
        if k in obs and obs[k]:
            g["ids"].add(str(obs[k]).upper())
    for e in obs.get("events", []) or []:
        g["ids"].add(str(e.get("event_id", "")).upper())
        g["event_ids"].add(str(e.get("event_id", "")).upper())
        g["times"].add(e.get("time"))
        try:  # derived end time (start + duration)
            g["times"].add((datetime.strptime(e["time"], "%H:%M") + timedelta(minutes=int(e.get("duration", 0)))).strftime("%H:%M"))
        except Exception:
            pass
        g["event_titles"].add(fold(e.get("title", "")))
    for r in obs.get("results", []) or []:
        g["ids"].add(str(r.get("product_id", "")).upper())
        g["money"].add(float(r.get("price_usd", -1)))
        g["products"].add(r.get("name", "").lower())
    for k in ("price_usd", "unit_price_usd", "total_usd"):
        if k in obs:
            g["money"].add(float(obs[k]))
    if "name" in obs and "price_usd" in obs:
        g["products"].add(obs["name"].lower())
    for k in ("conditions",):
        for c in obs.get(k, []) or []:
            g["med_conditions"].add(c.lower())
    for k in ("medications",):
        for c in obs.get(k, []) or []:
            g["drugs"].add(c.lower())
    for k in ("drug_a", "drug_b"):
        if k in obs:
            g["drugs"].add(str(obs[k]).lower())
    if "severity" in obs:
        g["severity"].add(_sev(obs["severity"]))
    if "dob" in obs:
        g["dates"].add(obs["dob"])
    if "patient_id" in obs:
        g["ids"].add(obs["patient_id"])
    if "doctor_id" in obs:
        g["ids"].add(obs["doctor_id"])
    if "time" in obs:
        g["times"].add(obs["time"])
    if "title" in obs:
        g["event_titles"].add(fold(obs["title"]))
    return g


def merge_grounded(gs):
    out = {}
    for g in gs:
        for k, v in g.items():
            if k == "temp_lists":
                out.setdefault(k, []).extend(v)
            else:
                out.setdefault(k, set()).update(v)
    return out


def temp_candidates(G, cfg=DEFAULT_CFG):
    """All grounded temperatures (value, unit) incl. derived aggregates and conversions."""
    cands = set(G.get("temp", set()))
    if cfg.get("derived", True):
        for L, unit in G.get("temp_lists", []):
            if not L:
                continue
            mean = statistics.mean(L)
            for v in (mean, round(mean), round(mean, 1), round(mean, 2), math.floor(mean), min(L), max(L),
                      statistics.median(L), max(L) - min(L)):
                cands.add((float(v), unit))
        base = [v for v in cands]
        for (a, ua) in base:
            for (b, ub) in base:
                if ua == ub and a != b:
                    cands.add((abs(a - b), ua))
    if cfg.get("unit_conv", True):
        for (v, u) in list(cands):
            if u == "C":
                cands.add((float(c2f(v)), "F"))
                cands.add((float(round(v * 9 / 5 + 32, 1)), "F"))
            else:
                cands.add((round(f2c(v), 1), "C"))
    return cands


def temp_match(v, unit, cands, cfg=DEFAULT_CFG):
    tol = cfg.get("temp_tol", 1.0)
    for (c, cu) in cands:
        if unit is not None and cu != unit:
            continue
        if abs(v - c) <= tol + 1e-9:
            return True
    return False


# ------------------------------------------------------------------------------------------------
# Trace representation
# ------------------------------------------------------------------------------------------------
def call_record(tool, params_emitted, params_exec, result, errors):
    """Normalised executed-call record."""
    errors = errors or []
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            result = {"error": result}
    if result is None:
        result = {}
    err_s = str(result.get("error", "")) if isinstance(result, dict) else ""
    schema_err = bool(errors) or err_s.startswith("Unknown tool") or bool(
        re.match(r"(Missing required parameter|Unknown parameter|Parameter ')", err_s))
    executed = not schema_err
    success = executed and isinstance(result, dict) and "error" not in result
    return {"tool": tool, "params_emitted": params_emitted, "params": params_exec, "result": result,
            "schema_error": schema_err, "executed": executed, "success": success}


# ------------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------------
def _call_slot_ok(g, params, slot, cfg, answer_ext=None, alt=None):
    p = alt or g.params
    if slot == "drug_pair":
        return slot_equal("drug_pair", [p["drug_a"], p["drug_b"]],
                          [params.get("drug_a", ""), params.get("drug_b", "")], cfg)
    if slot == "jan_range":
        s, e = norm_date(params.get("start_date", ""), cfg), norm_date(params.get("end_date", ""), cfg)
        okfull = any(s == a["start_date"] and e == a["end_date"] for a in g.alts)
        if okfull:
            return True
        if cfg.get("hist_lenient"):
            return bool(re.fullmatch(r"202[34]-01-\d\d", s) and re.fullmatch(r"202[34]-01-\d\d", e) and s[:4] == e[:4])
        return False
    if slot == "unit":
        if params.get("unit") == p["unit"]:
            return True
        if cfg.get("unit_conv", True) and params.get("unit", "celsius") == "celsius" and answer_ext is not None:
            # celsius call + correct fahrenheit value in the answer is an acceptable conversion path
            obs = sim("get_weather", {k: v for k, v in params.items() if k != "unit"}) \
                if "city" in params and "date" in params else None
            if obs and "temperature" in obs:
                fv = c2f(obs["temperature"])
                return any(u == "F" and abs(v - fv) <= cfg.get("temp_tol", 1.0) for v, u in answer_ext["temp"])
        return False
    if slot == "include_hourly":
        return slot_equal(slot, True, params.get(slot), cfg)
    if slot == "title" and cfg.get("title_lenient") and isinstance(params.get("title"), str):
        gt, et = set(fold(p["title"], cfg).split()), set(fold(params["title"], cfg).split())
        if gt <= et and (et - gt) <= {"meeting", "session", "call", "event"}:
            return True
    gv = p.get(slot)
    if gv is None:
        return True
    return slot_equal(slot, gv, params.get(slot), cfg)


def _slots_ok(g, params, cfg, answer_ext):
    """Per-slot correctness against the best-matching alternative."""
    best = None
    for alt in g.alts:
        r = {s: _call_slot_ok(g, params, s, cfg, answer_ext, alt) for s in g.scored}
        if best is None or sum(r.values()) > sum(best.values()):
            best = r
    return best


def resolve_conditionals(spec, calls, cfg):
    """Decide which conditional gold calls are required; fill dynamic params. Returns (gold_calls, info)."""
    info = {}
    gold = []
    for g in spec.calls:
        g2 = copy.copy(g)
        g2.params = copy.deepcopy(g.params)
        g2.alts = copy.deepcopy(g.alts)
        if g.conditional == "in_stock":
            prod = sim("get_product", {"product_id": g.params["product_id"]})
            info["in_stock"] = prod["in_stock"]
            g2.required = prod["in_stock"]
        elif g.conditional == "in_stock_under":
            prod = sim("get_product", {"product_id": g.params["product_id"]})
            info["in_stock"] = prod["in_stock"]
            info["price"] = prod["price_usd"]
            g2.required = bool(prod["in_stock"] and prod["price_usd"] < spec.extra["limit"])
        elif g.conditional == "slot_free":
            ev = _received_events(calls, spec.extra["date"], cfg)
            src = "received"
            if ev is None:
                ev, src = mock_events(spec.extra["date"]), "mock_empty_store"
            t0 = spec.extra["slot_time"]
            busy = False
            for e in ev.get("events", []):
                try:
                    st = datetime.strptime(e["time"], "%H:%M")
                    s0 = datetime.strptime(t0, "%H:%M")
                    dur = int(e.get("duration", 0) or 0)
                    if st == s0 or st <= s0 < st + timedelta(minutes=dur):
                        busy = True
                except Exception:
                    pass
            info["slot_free"] = not busy
            info["condition_source"] = src
            g2.required = not busy
        elif g.conditional == "title_present":
            ev = _received_events(calls, spec.extra["date"], cfg)
            src = "received"
            if ev is None:
                ev, src = mock_events(spec.extra["date"]), "mock_empty_store"
            hits = [e["event_id"] for e in ev.get("events", []) if fold(e.get("title", "")) == fold(spec.extra["title"])]
            info["title_present"] = bool(hits)
            info["condition_source"] = src
            g2.required = bool(hits)
            if hits:
                g2.params["event_id"] = hits[0]
                g2.alts = [{"event_id": h} for h in hits]
        else:
            g2.required = True
        gold.append(g2)
    return gold, info


def _received_events(calls, date, cfg):
    for c in calls:
        if c["tool"] == "get_events" and c["success"] and slot_equal("date", date, c["params"].get("date"), cfg):
            return c["result"]
    return None


def assign_calls(gold, calls, cfg, answer_ext):
    """One-to-one assignment of executed calls to required gold calls (full matches first)."""
    used = set()
    assign = {}
    slotres = {}
    for full_pass in (True, False):
        for gi, g in enumerate(gold):
            if not g.required or gi in assign:
                continue
            best, bestscore, bestres = None, -1, None
            for ci, c in enumerate(calls):
                if ci in used or c["tool"] != g.tool or not c["executed"]:
                    continue
                res = _slots_ok(g, c["params"], cfg, answer_ext)
                sc = sum(res.values())
                if full_pass and sc < len(res):
                    continue
                if sc > bestscore:
                    best, bestscore, bestres = ci, sc, res
            if best is not None:
                assign[gi] = best
                used.add(best)
                slotres[gi] = bestres
    return assign, slotres


def gold_observations(g, calls, assign_ci, cfg, full_match=None):
    """Gold observation(s) for gold call g.
    Stateless tools: deterministic replay.  If the matched executed call fully matches an acceptable variant, the
    replay of THAT variant (+ the executed call's own observation when equiv_obs) is gold; otherwise the union of
    the replays of all acceptable variants.  Store-dependent tools (get_events/create/delete): the observation the
    agent received from the matched call, only if that call fully matches the gold slots (else: unknown)."""
    obs = []
    matched = assign_ci is not None and full_match
    if g.tool in STATELESS:
        if matched:
            p = calls[assign_ci]["params"]
            hit = [a for a in g.alts if all(_call_slot_ok(g, p, s, cfg, None, a) for s in g.scored if s != "unit")]
            obs.append(sim(g.tool, hit[0] if hit else g.alts[0]))
            if cfg.get("equiv_obs", True) and calls[assign_ci]["success"]:
                obs.append(calls[assign_ci]["result"])
        else:
            for a in g.alts:
                obs.append(sim(g.tool, a))
    else:
        if matched and calls[assign_ci]["executed"]:
            obs.append(calls[assign_ci]["result"])
    good = [o for o in obs if isinstance(o, dict) and "error" not in o]
    return good or [o for o in obs if isinstance(o, dict)]


def score_trace(task, calls, final_answer, cfg=None, spec=None, answer_only=False, n_exec_hint=None):
    """Score one trace.  Returns dict of components and outcome code."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    spec = spec or build_spec(task, cfg)
    ans = final_answer or ""
    res = {"validator_version": VALIDATOR_VERSION, "template": spec.template, "verifiable": spec.verifiable,
           "reason": spec.reason}
    if not spec.verifiable:
        res.update(outcome="not_verifiable", T=None, P=None, S=None, A=None, CORRECT=None)
        return res
    if cfg["gold_source"] == "taskfile":
        tf = taskfile_spec(task)
        if not tf:
            res.update(outcome="not_verifiable", T=None, P=None, S=None, A=None, CORRECT=None,
                       reason="taskfile gold unusable")
            return res
        for g in tf:
            g.keys = next((x.keys for x in spec.calls if x.tool == g.tool), [])
        spec = TaskSpec(task, spec.template + "|taskfile", tf, order=[])
    ext = extract(ans, cfg, task["domain"])
    if answer_only:
        return _score_answer_only(spec, ext, ans, cfg, res, n_exec_hint)

    gold, cinfo = resolve_conditionals(spec, calls, cfg)
    assign, slotres = assign_calls(gold, calls, cfg, ext)
    req = [i for i, g in enumerate(gold) if g.required]

    # ---- T
    T_missing = [gold[i].tool for i in req if i not in assign]
    order_ok = all(assign[i] < assign[j] for i, j in spec.order if i in assign and j in assign
                   and gold[i].required and gold[j].required)
    T = (not T_missing) and order_ok
    # ---- P (per slot, on the matched executed call)
    per_slot = {}
    for gi, r in slotres.items():
        for s, ok in r.items():
            per_slot[f"{gold[gi].tool}.{s}"] = per_slot.get(f"{gold[gi].tool}.{s}", True) and ok
    for gi in req:
        if gi not in assign:
            for s in gold[gi].scored:
                per_slot.setdefault(f"{gold[gi].tool}.{s}", None)
    P = T and all(all(slotres[gi].values()) for gi in req if gi in slotres)
    # emitted-params diagnostic (pre-injection)
    P_emitted = None
    if any(c.get("params_emitted") is not None and c["params_emitted"] != c["params"] for c in calls):
        alt_calls = [dict(c, params=c["params_emitted"]) for c in calls]
        a2, s2 = assign_calls(gold, alt_calls, cfg, ext)
        P_emitted = all(gi in a2 for gi in req) and all(all(s2[gi].values()) for gi in req if gi in s2)

    # ---- S (state)
    S = None
    s_detail = ""
    if spec.is_write:
        gold_writes = [g for g in gold if g.write and g.required]
        succ_writes = [(ci, c) for ci, c in enumerate(calls) if c["tool"] in WRITE_TOOLS and c["success"]]
        matched_ci = {assign[i] for i, g in enumerate(gold) if g.write and g.required and i in assign
                      and all(slotres[i].values())}
        gold_ok = all((i in assign and all(slotres[i].values())) for i, g in enumerate(gold) if g.write and g.required)
        extra = [c for ci, c in succ_writes if ci not in matched_ci]
        if cfg["state"] == "strict":
            S = gold_ok and not extra
        else:
            S = gold_ok
        s_detail = f"gold_writes={len(gold_writes)};gold_ok={gold_ok};extra_state_changing_writes={len(extra)}"

    # ---- A (answer faithfulness)
    gobs_all = []
    recall_hits, recall_total = 0, 0
    key_fail = []
    for gi, g in enumerate(gold):
        ci = assign.get(gi)
        gobs = gold_observations(g, calls, ci, cfg, ci is not None and all(slotres[gi].values())) if g.required else []
        gobs_all.extend(gobs)
    aux = [c["result"] for ci, c in enumerate(calls) if c["success"] and ci not in assign.values()
           and _on_task(c, task, gold, cfg)] if cfg.get("aux_obs", True) else []
    G = merge_grounded([grounded_from_obs(o, cfg) for o in gobs_all + aux] + [_grounded_from_query(task, spec, gold)])
    # forecast dates are generated from datetime.now() at run time (not deterministic, entity-independent):
    # dates printed in ANY received forecast observation are neutral (grounded), never evidence of error.
    for c in calls:
        if c["tool"] == "get_forecast" and c["success"]:
            G.setdefault("dates", set()).update(e.get("date") for e in c["result"].get("forecast", []) or [])
    tcands = temp_candidates(G, cfg)
    point = _point_obs(gold, calls, assign, cfg, {gi: all(r.values()) for gi, r in slotres.items()})
    for gi, g in enumerate(gold):
        ci = assign.get(gi)
        if g.write:
            continue
        if not g.required:
            continue
        gobs = gold_observations(g, calls, ci, cfg, ci is not None and all(slotres[gi].values()))
        for kind, mode in g.keys:
            recall_total += 1
            ok = _key_present(kind, g, gobs, ext, ans, cfg, task, point)
            recall_hits += int(ok)
            if not ok:
                key_fail.append(f"{g.tool}:{kind}")
    # write-outcome claims
    claim_ok = True
    claim_detail = []
    if spec.is_write:
        claim_ok, claim_detail = _claims_faithful(spec, gold, calls, assign, slotres, ans, cinfo, cfg)
        recall_total += 1
        recall_hits += int(claim_ok)
        if not claim_ok:
            key_fail.append("write_claim")
    ext_p = extract(_assertive_text(ans), cfg, task["domain"])
    prec_n, prec_ok, unground = _precision(ext_p, G, tcands, cfg, task, spec)
    recall = recall_hits / recall_total if recall_total else 1.0
    precision = prec_ok / prec_n if prec_n else 1.0
    pr = cfg.get("precision_required", 1.0)
    A = (recall >= cfg.get("recall_required", 1.0) - 1e-9) and claim_ok and (pr is None or precision >= pr - 1e-9)
    S_eff = True if S is None else S
    CORRECT = bool(T and P and S_eff and A)

    # ---- trace-level facts for outcome codes
    n_calls = len(calls)
    n_exec = sum(c["executed"] for c in calls)
    n_succ = sum(c["success"] for c in calls)
    n_schema = sum(c["schema_error"] for c in calls)
    req_tools = {gold[i].tool for i in req}
    req_succ = any(c["success"] and c["tool"] in req_tools for c in calls)
    # values unsupported by ANY observation received in the trace (nor gold nor query)
    G_trace = merge_grounded([G] + [grounded_from_obs(c["result"], cfg) for c in calls if isinstance(c["result"], dict)])
    _, _, unsupported = _precision(ext_p, G_trace, temp_candidates(G_trace, cfg), cfg, task, spec)
    asserts = _asserts(ext, ans, task, spec) or (spec.is_write and _success_claim(ans))
    succ_write_tools = {c["tool"] for c in calls if c["success"] and c["tool"] in WRITE_TOOLS}
    write_claim_unexecuted = bool(spec.is_write and _success_claim(ans) and any(
        g.write and g.required and g.tool not in succ_write_tools for g in gold))
    refusal = bool(REFUSAL_RE.search(ans)) or bool(PLACEHOLDER_RE.match(ans)) or not ans.strip()
    if INFRA_RE.match(ans):
        outcome = "infra_error"
    elif CORRECT:
        outcome = "correct"
    elif asserts and (n_exec == 0 or write_claim_unexecuted):
        outcome = "fabricated"
    elif n_calls > 0 and n_exec == 0:
        outcome = "schema_error"
    elif refusal and recall_hits <= (1 if spec.is_write and claim_ok else 0) and not asserts:
        outcome = "refused" if not PLACEHOLDER_RE.match(ans) and "max steps reached" not in ans.lower() else "incomplete"
    elif not T:
        outcome = "incomplete"
    elif not P:
        outcome = "wrong_params"
    elif S is False:
        outcome = "wrong_state"
    else:
        outcome = "wrong_answer_values"
    res.update({
        "T": T, "P": P, "S": S, "A": A, "CORRECT": CORRECT, "outcome": outcome,
        "P_emitted": P_emitted, "T_missing": ";".join(T_missing), "order_ok": order_ok,
        "per_slot": per_slot, "S_detail": s_detail, "A_recall": round(recall, 4), "A_precision": round(precision, 4),
        "A_recall_n": recall_total, "A_prec_n": prec_n, "A_key_fail": ";".join(key_fail),
        "ungrounded": ";".join(unground)[:300], "unsupported_by_trace": ";".join(unsupported)[:300],
        "n_unsupported": len(unsupported), "claim_ok": claim_ok, "claim_detail": ";".join(claim_detail),
        "n_calls": n_calls, "n_exec": n_exec, "n_success": n_succ, "n_schema_err": n_schema,
        "conditions": json.dumps(cinfo), "asserts": asserts, "refusal_lang": refusal,
        "write_claim_unexecuted": write_claim_unexecuted,
    })
    return res


def _grounded_from_query(task, spec, gold):
    """Values legitimately echoed from the query / gold inputs."""
    g = {"temp": set(), "temp_lists": [], "humidity": set(), "condition": set(), "ids": set(), "money": set(),
         "dates": set(), "times": set(), "drugs": set(), "med_conditions": set(), "severity": set(),
         "products": set(), "event_titles": set(), "event_ids": set()}
    q = task["query"]
    ext = extract(q)
    g["ids"].update(ext["ids"])
    g["money"].update(ext["money"])
    g["dates"].update(ext["dates"])
    g["times"].update(ext["times"])
    g["drugs"].update(ext["drugs"])
    g["products"].update(ext["products"])
    for gc in gold:
        for a in gc.alts:
            for k, v in a.items():
                if k in ("date", "start_date", "end_date") and v:
                    g["dates"].add(v)
                if k == "time" and v:
                    g["times"].add(v)
                if k in ("drug_a", "drug_b"):
                    g["drugs"].add(str(v).lower())
                if k in ("event_id", "product_id", "patient_id", "doctor_id") and v:
                    g["ids"].add(str(v).upper())
                if k == "title" and v:
                    g["event_titles"].add(fold(v))
        if gc.tool == "get_events":
            pass
    for m in re.findall(r"\bP\d{3}\b|\bD\d{3}\b", q):
        g["ids"].add(m)
    g["event_titles"].update(ext["event_titles"])
    if spec.extra.get("title"):
        g["event_titles"].add(fold(spec.extra["title"]))
    return g


def _key_present(kind, g, gobs, ext, ans, cfg, task, point_obs=None):
    if not gobs:
        return False
    if kind == "temp":
        unit_req = "F" if g.params.get("unit") == "fahrenheit" else None
        for o in gobs:
            t, u = o.get("temperature"), ("F" if o.get("unit") == "fahrenheit" else "C")
            if t is None:
                continue
            cands = {(float(t), u)}
            if cfg.get("unit_conv", True):
                cands |= {(float(c2f(t)), "F"), (round(t * 9 / 5 + 32, 1), "F")} if u == "C" else {(round(f2c(t), 1), "C")}
            for v, mu in ext["temp"]:
                if unit_req == "F":
                    if mu == "F" or (mu is None and "fahrenheit" in ans.lower()):
                        if temp_match(v, "F", cands, cfg):
                            return True
                    continue
                if temp_match(v, mu if mu else None, cands, cfg):
                    return True
        return False
    if kind == "condition":
        conds = {o.get("condition") for o in gobs if o.get("condition")}
        return bool(conds & set(ext["condition"]))
    if kind == "list_any":
        # >=1 value that comes from the LIST observation (forecast / historical), not from a point observation
        point_t = {float(t) for t in (point_obs or {}).get("temps", set())}
        point_c = set((point_obs or {}).get("conds", set()))
        G = merge_grounded([grounded_from_obs(o, cfg) for o in gobs])
        G["temp"] = set()  # list values only
        lst = set()
        for L, u in G.get("temp_lists", []):
            for x in L:
                lst.add((float(x), u))
        Gc = {"temp": lst, "temp_lists": G.get("temp_lists", [])}
        tc = temp_candidates(Gc, cfg)
        pc = temp_candidates({"temp": {(t, "C") for t in point_t}, "temp_lists": []}, cfg) if point_t else set()
        for v, u in ext["temp"]:
            if temp_match(v, u, tc, cfg) and not temp_match(v, u, pc, dict(cfg, temp_tol=0.0)):
                return True
        conds = set()
        for o in gobs:
            for e in o.get("forecast", []) or []:
                conds.add(e.get("condition"))
        return bool((conds - point_c) & set(ext["condition"]))
    if kind in ("name", "conditions", "medications"):
        o = gobs[0]
        if kind == "name":
            return fold(o.get("name", "#")) in fold(ans)
        if kind == "conditions":
            need = {c.lower() for c in o.get("conditions", [])}
            return bool(need) and need <= set(ext["med_conditions"])
        need = {c.lower() for c in o.get("medications", [])}
        return bool(need) and need <= set(ext["drugs"])
    if kind == "interaction":
        o = gobs[0]
        neg = bool(INTERACT_NEG_RE.search(ans))
        pos = bool(INTERACT_POS_RE.search(ans)) or any(s in ("high", "moderate") for s in ext["severity"])
        if o.get("interaction"):
            sev = _sev(o.get("severity", ""))
            return pos and (sev in ext["severity"] or sev not in ("high", "moderate"))
        return neg and not (pos and not neg)
    if kind == "events":
        o = gobs[0]
        evs = o.get("events", [])
        if not evs:
            return bool(NO_EVENTS_RE.search(ans))
        titles = {fold(e["title"]) for e in evs}
        times = {e["time"] for e in evs}
        ids = {str(e["event_id"]).upper() for e in evs}
        return bool(any(t in fold(ans) for t in titles) or (times & set(ext["times"])) or (ids & set(ext["ids"])))
    if kind == "search":
        o = gobs[0]
        names = {r["name"].lower() for r in o.get("results", [])}
        ids = {r["product_id"].upper() for r in o.get("results", [])}
        counts = {str(o.get("total_matches")), str(len(o.get("results", [])))}
        cnt = re.findall(r"\b(\d+)\s+(?:\w+\s+){0,2}(?:items?|products?|results?|matches)\b", ans, re.I)
        return bool(names & set(ext["products"]) or ids & set(ext["ids"]) or (counts & set(cnt)) or
                    (not o.get("results") and re.search(r"\bno (?:results|products|items|matches)\b", ans, re.I)))
    if kind == "product":
        o = gobs[0]
        return (o.get("name", "").lower() in set(ext["products"])) and any(
            abs(m - o.get("price_usd", -999)) <= cfg.get("num_tol", 0.0) for m in ext["money"])
    return False


WRITE_VERB = r"(?:created|scheduled|booked|placed|deleted|removed|cancel+ed|added|ordered|confirmed|set up|reorganized)"
CLAIM_PATTERNS = [
    re.compile(rf"\b(?:has|have|had)\s+(?:now\s+|also\s+|just\s+)?(?:been\s+)?(?:successfully\s+)?{WRITE_VERB}\b", re.I),
    re.compile(rf"\b(?:was|were|is|are)\s+(?:now\s+|also\s+)?(?:successfully\s+)?{WRITE_VERB}\b", re.I),
    re.compile(rf"\b(?:I|I've|we|we've)\s+(?:\w+\s+){{0,2}}?{WRITE_VERB}\b", re.I),
    re.compile(rf"\bsuccessfully\s+{WRITE_VERB}\b|\b{WRITE_VERB}\s+successfully\b", re.I),
    re.compile(rf"(?:^|[.!\n:]\s*)(?:[\w'\"’:-]+\s+){{0,5}}?{WRITE_VERB}\b", re.I),  # 'Event EVT_1 deleted.' 'Order placed ...'
]
NEG_NEAR_RE = re.compile(r"(\bno\b|\bnot\b|n't\b|\bnever\b|\bunable\b|\bcannot\b|\bfailed\b|\bwithout\b|"
                         r"\balready\b|\bif\b|\bwould\b|\bwill\b|\bshould\b|\bcan\b|\bplease\b|\bto be\b|\bwant\b|"
                         r"\bneed\b|\bnor\b|\bthere (?:is|are)\b|\bobserved\b|\bexisting\b|\blet me\b)", re.I)
CONFIRM_RE = re.compile(r"\b(done|all set|status[:\s]+(?:confirmed|scheduled|created))\b|\b(APPT|ORD)_\d+\b", re.I)


def _success_claim(ans):
    """Affirmative (non-negated, non-hypothetical) statement that a write happened."""
    a = ans or ""
    for pat in CLAIM_PATTERNS:
        for m in pat.finditer(a):
            sent_start = max(a.rfind(".", 0, m.start() + 1), a.rfind("\n", 0, m.start() + 1), a.rfind("!", 0, m.start() + 1))
            pre = a[sent_start + 1:m.end()]
            if NEG_NEAR_RE.search(pre):
                continue
            return True
    for m in CONFIRM_RE.finditer(a):
        sent_start = max(a.rfind(".", 0, m.start() + 1), a.rfind("\n", 0, m.start() + 1))
        if not NEG_NEAR_RE.search(a[sent_start + 1:m.end()]):
            return True
    return False


ID_SLOTS = ("city", "date", "start_date", "end_date", "patient_id", "product_id", "event_id", "drug_a", "drug_b",
            "doctor_id", "query", "title")


def _on_task(c, task, gold, cfg):
    """Auxiliary (non-gold) executed call is admissible evidence iff every identifying slot value it used appears in
    the query or in a gold call (so injected / wrong-entity observations are never admissible)."""
    q = fold(task["query"], cfg)
    gvals = {fold(v, cfg) for g in gold for a in g.alts for v in a.values() if isinstance(v, (str, int))}
    p = c["params"] if isinstance(c["params"], dict) else {}
    for k in ID_SLOTS:
        if k in p:
            v = fold(p[k], cfg)
            if k in ("date", "start_date", "end_date"):
                v = norm_date(p[k], cfg)
            if v not in gvals and v not in q:
                return False
    return True


def _point_obs(gold, calls, assign, cfg, full=None):
    """Temperatures / conditions of single-day weather observations (used to keep list keys honest)."""
    full = full or {}
    temps, conds = set(), set()
    for gi, g in enumerate(gold):
        if g.tool == "get_weather" and getattr(g, "required", True):
            for o in gold_observations(g, calls, assign.get(gi), cfg, full.get(gi, False)):
                if "temperature" in o:
                    temps.add(o["temperature"] if o.get("unit") != "fahrenheit" else round(f2c(o["temperature"])))
                if o.get("condition"):
                    conds.add(o["condition"])
    return {"temps": temps, "conds": conds}


HEDGE_RE = re.compile(r"\b(consider|suggest|alternative|would you like|do you want|could try|you could|you might|"
                      r"if (?:open|available|free)|perhaps|maybe|for example|e\.g\.)\b|\?\s*$", re.I)


def _assertive_text(ans):
    """Answer with hedged / suggestion / question sentences removed (precision only scores assertions)."""
    sents = re.split(r"(?<=[.!?])\s+|\n+", ans or "")
    return " ".join(x for x in sents if not HEDGE_RE.search(x))


def _failure_claim(ans):
    return bool(FAIL_RE.search(ans or "")) or bool(re.search(r"\b(no|not)\b[^.]{0,40}\b(event|meeting|appointment|order|"
                                                              r"scheduled|found|exist|created|placed|booked)", ans or "", re.I))


def _claims_faithful(spec, gold, calls, assign, slotres, ans, cinfo, cfg):
    """Outcome claims for write tasks must match what actually happened."""
    detail = []
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", ans) if s.strip()]
    ok = True
    for gi, g in enumerate(gold):
        if not g.write:
            continue
        ci = assign.get(gi)
        if g.required:
            matched = ci is not None and all(slotres[gi].values())
            if g.tool == "delete_event":
                if not matched:
                    # never executed with the gold id: the required action did not happen
                    detail.append("delete_not_executed")
                    ok = False
                    continue
                r = calls[ci]["result"]
                if r.get("success"):
                    good = _success_claim(ans) and not re.search(r"not found|does not exist|doesn't exist", ans, re.I)
                else:
                    good = bool(re.search(r"not found|does not exist|doesn't exist|no (?:such|matching) event|"
                                          r"could not|couldn't|unable|failed|wasn't|was not", ans, re.I))
                detail.append(f"delete_result_success={r.get('success')};claim_ok={good}")
                ok &= good
            else:
                if matched and calls[ci]["success"]:
                    mention = g.tool == "create_event" and fold(g.params.get("title", "#")) in fold(ans)
                    good = (_success_claim(ans) or mention) and not _all_fail(sentences, g)
                else:
                    good = False
                detail.append(f"{g.tool}_claim_ok={good}")
                ok &= good
        else:
            # gold says: do NOT perform this write (condition false) -> answer must say so and not claim it was done
            claimed = _success_claim(ans)
            good = (not claimed) and _failure_claim(ans)
            detail.append(f"{g.tool}_not_required;claim_ok={good}")
            ok &= good
    return ok, detail


def _all_fail(sentences, g):
    """True if every sentence mentioning the write is a failure statement."""
    rel = [s for s in sentences if re.search(r"creat|schedul|book|order|placed|appointment|event|meeting", s, re.I)]
    return bool(rel) and all(FAIL_RE.search(s) for s in rel)


def _asserts(ext, ans, task, spec):
    """Does the answer assert concrete results (typed values not merely echoed from the query)?"""
    if PLACEHOLDER_RE.match(ans or ""):
        return False
    qext = extract(task["query"])
    if ext["temp"] or ext["humidity"] or ext["condition"] or ext["other_weather"] or ext["severity"] or ext["money"]:
        return True
    if set(ext["ids"]) - set(qext["ids"]):
        return True
    if set(ext["med_conditions"]) or (set(ext["drugs"]) - set(qext["drugs"])):
        return True
    if set(ext["products"]) - set(qext["products"]) or (set(ext["event_titles"]) - set(qext["event_titles"])):
        return True
    if spec.template.startswith("M2") or spec.template in ("M6_record_interaction",):
        if INTERACT_NEG_RE.search(ans) or INTERACT_POS_RE.search(ans):
            return True
    if NO_EVENTS_RE.search(ans):
        return True
    return False


def _precision(ext, G, tcands, cfg, task, spec):
    """Typed mentions grounded in G.  Returns (n_mentions, n_grounded, list_of_ungrounded)."""
    n, ok, bad = 0, 0, []
    for v, u in ext["temp"]:
        n += 1
        if temp_match(v, u, tcands, cfg):
            ok += 1
        else:
            bad.append(f"temp:{v}{u or ''}")
    for v in ext["humidity"]:
        n += 1
        if any(abs(v - h) <= cfg.get("num_tol", 0.0) for h in G.get("humidity", set())):
            ok += 1
        else:
            bad.append(f"humidity:{v}")
    for c in ext["condition"]:
        n += 1
        if c in G.get("condition", set()):
            ok += 1
        else:
            bad.append(f"cond:{c}")
    for w in ext["other_weather"]:
        n += 1
        bad.append(f"cond:{w}")
    qids = set(extract(task["query"])["ids"])
    for i in ext["ids"]:
        n += 1
        if i in G.get("ids", set()) or i in qids:
            ok += 1
        else:
            bad.append(f"id:{i}")
    for mny in ext["money"]:
        n += 1
        if any(abs(mny - x) <= max(cfg.get("num_tol", 0.0), 0.005) for x in G.get("money", set())):
            ok += 1
        else:
            bad.append(f"money:{mny}")
    for d in ext["dates"]:
        n += 1
        gd = G.get("dates", set())
        if d.startswith("*-"):
            hit = any(str(x)[4:] == d[1:] for x in gd if x)
        else:
            hit = d in gd
        if hit:
            ok += 1
        else:
            bad.append(f"date:{d}")
    if spec.template.startswith("C"):
        for t in ext["times"]:
            n += 1
            if t in G.get("times", set()):
                ok += 1
            else:
                bad.append(f"time:{t}")
    for d in ext["drugs"]:
        n += 1
        if d in G.get("drugs", set()):
            ok += 1
        else:
            bad.append(f"drug:{d}")
    for c in ext["med_conditions"]:
        n += 1
        if c in G.get("med_conditions", set()):
            ok += 1
        else:
            bad.append(f"medcond:{c}")
    for s in ext["severity"]:
        n += 1
        if s in G.get("severity", set()) or (s == "low" and "unknown" in G.get("severity", set()) and False):
            ok += 1
        else:
            bad.append(f"severity:{s}")
    for p in ext["products"]:
        n += 1
        if p in G.get("products", set()):
            ok += 1
        else:
            bad.append(f"product:{p}")
    for t in ext["event_titles"]:
        n += 1
        if t in G.get("event_titles", set()):
            ok += 1
        else:
            bad.append(f"title:{t}")
    return n, ok, bad


def _score_answer_only(spec, ext, ans, cfg, res, n_exec_hint):
    """Open-weight CSVs carry only the final answer: score A (vs deterministic gold observation) only."""
    if any(g.tool in STORE_DEPENDENT for g in spec.calls) or spec.is_write:
        res.update(outcome="not_verifiable_answer_only", T=None, P=None, S=None, A=None, CORRECT=None,
                   reason="answer-only trace: task involves write or store-dependent tools; state not observable")
        return res
    gold, _ = resolve_conditionals(spec, [], cfg)
    gobs_all = []
    hits = tot = 0
    point = _point_obs(gold, [], {}, cfg)
    for g in gold:
        gobs = gold_observations(g, [], None, cfg)
        gobs_all.extend(gobs)
        for kind, mode in g.keys:
            tot += 1
            hits += int(_key_present(kind, g, gobs, ext, ans, cfg, spec.task, point))
    G = merge_grounded([grounded_from_obs(o, cfg) for o in gobs_all] + [_grounded_from_query(spec.task, spec, gold)])
    ext = dict(ext, dates=[d for d in ext["dates"] if d in G.get("dates", set())])  # run-time dates unknowable here
    pn, pok, bad = _precision(ext, G, temp_candidates(G, cfg), cfg, spec.task, spec)
    prec = pok / pn if pn else 1.0
    pr = cfg.get("precision_required", 1.0)
    A = hits == tot and (pr is None or prec >= pr - 1e-9)
    asserts = _asserts(ext, ans, spec.task, spec)
    if INFRA_RE.match(ans):
        outcome = "infra_error"
    elif A:
        outcome = "answer_correct"
    elif n_exec_hint == 0 and asserts:
        outcome = "fabricated"
    elif n_exec_hint == 0:
        outcome = "refused" if REFUSAL_RE.search(ans) and not PLACEHOLDER_RE.match(ans) else "incomplete"
    else:
        outcome = "wrong_answer_values"
    res.update(outcome=outcome, T=None, P=None, S=None, A=A, CORRECT=None, A_answer_only=A,
               A_recall=round(hits / tot, 4) if tot else 1.0, A_precision=round(prec, 4),
               ungrounded=";".join(bad)[:300], asserts=asserts, n_exec=n_exec_hint)
    return res
