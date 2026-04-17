"""Tool registry and executor with 9 domain tools (3 per domain).

Every tool has JSON schema validation, returns structured dicts,
never raises exceptions — returns {"error": "..."} on bad input.
"""

import json
import logging
import hashlib
import random
from datetime import datetime, timedelta
from typing import Any

from config import RANDOM_SEED

logger = logging.getLogger(__name__)

# ── JSON Schemas ──────────────────────────────────────────────────────────────

TOOL_SCHEMAS: dict[str, dict] = {
    # ── Weather Domain ────────────────────────────────────────────────────────
    "get_weather": {
        "name": "get_weather",
        "domain": "weather",
        "description": "Get current weather for a city on a specific date.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature unit",
                },
            },
            "required": ["city", "date"],
        },
    },
    "get_forecast": {
        "name": "get_forecast",
        "domain": "weather",
        "description": "Get weather forecast for a city for upcoming days.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 14,
                    "description": "Number of forecast days (1-14)",
                },
                "include_hourly": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include hourly breakdown",
                },
            },
            "required": ["city", "days"],
        },
    },
    "get_historical": {
        "name": "get_historical",
        "domain": "weather",
        "description": "Get historical weather data for a city between two dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            },
            "required": ["city", "start_date", "end_date"],
        },
    },
    # ── Calendar Domain ───────────────────────────────────────────────────────
    "create_event": {
        "name": "create_event",
        "domain": "calendar",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM format"},
                "duration_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1440,
                    "description": "Duration in minutes",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee emails",
                },
            },
            "required": ["title", "date", "time", "duration_minutes"],
        },
    },
    "get_events": {
        "name": "get_events",
        "domain": "calendar",
        "description": "Get all events for a specific date.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "calendar_id": {
                    "type": "string",
                    "default": "primary",
                    "description": "Calendar identifier",
                },
            },
            "required": ["date"],
        },
    },
    "delete_event": {
        "name": "delete_event",
        "domain": "calendar",
        "description": "Delete a calendar event by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Event ID to delete"},
                "notify_attendees": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to notify attendees",
                },
            },
            "required": ["event_id"],
        },
    },
    # ── Medical Domain ────────────────────────────────────────────────────────
    "get_patient_record": {
        "name": "get_patient_record",
        "domain": "medical",
        "description": "Retrieve patient medical record.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID (e.g. P001)"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["all"],
                    "description": "Fields to retrieve",
                },
            },
            "required": ["patient_id"],
        },
    },
    "check_drug_interaction": {
        "name": "check_drug_interaction",
        "domain": "medical",
        "description": "Check interaction between two drugs.",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_a": {"type": "string", "description": "First drug name"},
                "drug_b": {"type": "string", "description": "Second drug name"},
            },
            "required": ["drug_a", "drug_b"],
        },
    },
    "schedule_appointment": {
        "name": "schedule_appointment",
        "domain": "medical",
        "description": "Schedule a medical appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID"},
                "doctor_id": {"type": "string", "description": "Doctor ID"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "appointment_type": {
                    "type": "string",
                    "enum": ["checkup", "follow_up", "specialist", "emergency"],
                    "description": "Type of appointment",
                },
            },
            "required": ["patient_id", "doctor_id", "date", "appointment_type"],
        },
    },
    # ── E-commerce / Retail Domain ────────────────────────────────────────────
    "search_products": {
        "name": "search_products",
        "domain": "ecommerce",
        "description": "Search the retail catalog by keyword and optional category/price.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
                "category": {
                    "type": "string",
                    "enum": ["electronics", "books", "clothing", "home", "sports", "any"],
                    "default": "any",
                    "description": "Product category filter",
                },
                "max_price": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100000,
                    "description": "Max price in USD",
                },
            },
            "required": ["query"],
        },
    },
    "get_product": {
        "name": "get_product",
        "domain": "ecommerce",
        "description": "Get full product details by product_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product ID (e.g. PRD_1234)"},
            },
            "required": ["product_id"],
        },
    },
    "place_order": {
        "name": "place_order",
        "domain": "ecommerce",
        "description": "Place an order for a product at a given shipping address.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product ID"},
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Quantity to order",
                },
                "shipping_address": {"type": "string", "description": "Full shipping address"},
                "payment_method": {
                    "type": "string",
                    "enum": ["credit_card", "paypal", "apple_pay", "gift_card"],
                    "default": "credit_card",
                    "description": "Payment method",
                },
            },
            "required": ["product_id", "quantity", "shipping_address"],
        },
    },
}


# ── Schema Validation ─────────────────────────────────────────────────────────

def _validate_params(params: dict, schema: dict) -> list[str]:
    """Validate params against JSON schema. Returns list of error strings."""
    errors = []
    props = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for field in required:
        if field not in params:
            errors.append(f"Missing required parameter: '{field}'")

    # Check types and constraints
    for key, value in params.items():
        if key not in props:
            errors.append(f"Unknown parameter: '{key}'")
            continue

        prop_schema = props[key]
        expected_type = prop_schema.get("type")

        # Type check
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Parameter '{key}' must be string, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Parameter '{key}' must be integer, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Parameter '{key}' must be boolean, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Parameter '{key}' must be array, got {type(value).__name__}")

        # Range check for integers
        if expected_type == "integer" and isinstance(value, int):
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(
                    f"Parameter '{key}' value {value} below minimum {prop_schema['minimum']}"
                )
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(
                    f"Parameter '{key}' value {value} above maximum {prop_schema['maximum']}"
                )

        # Enum check
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            errors.append(
                f"Parameter '{key}' value '{value}' not in allowed values: {prop_schema['enum']}"
            )

    return errors


def _apply_defaults(params: dict, schema: dict) -> dict:
    """Apply default values for missing optional parameters."""
    result = dict(params)
    props = schema.get("properties", {})
    for key, prop_schema in props.items():
        if key not in result and "default" in prop_schema:
            result[key] = prop_schema["default"]
    return result


# ── Deterministic Mock Implementations ────────────────────────────────────────

def _seed_hash(seed_str: str) -> int:
    """Deterministic hash for seeded randomness."""
    return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)


CONDITIONS = ["sunny", "cloudy", "rainy", "partly cloudy", "overcast", "snowy"]

# Deterministic patient database
PATIENTS = {
    "P001": {"name": "Alice Johnson", "dob": "1985-03-15", "conditions": ["hypertension", "diabetes type 2"], "medications": ["metformin", "lisinopril"]},
    "P002": {"name": "Bob Smith", "dob": "1972-08-22", "conditions": ["asthma"], "medications": ["albuterol", "fluticasone"]},
    "P003": {"name": "Carol Davis", "dob": "1990-11-05", "conditions": ["migraine", "anxiety"], "medications": ["sumatriptan", "sertraline"]},
    "P004": {"name": "David Lee", "dob": "1968-01-30", "conditions": ["arthritis", "high cholesterol"], "medications": ["ibuprofen", "atorvastatin"]},
    "P005": {"name": "Eva Martinez", "dob": "1995-06-18", "conditions": ["allergies"], "medications": ["cetirizine"]},
}

DRUG_INTERACTIONS = {
    frozenset({"metformin", "lisinopril"}): {"interaction": False, "severity": "none", "details": "No known interaction"},
    frozenset({"ibuprofen", "lisinopril"}): {"interaction": True, "severity": "moderate", "details": "NSAIDs may reduce the effect of ACE inhibitors"},
    frozenset({"sertraline", "sumatriptan"}): {"interaction": True, "severity": "high", "details": "Risk of serotonin syndrome"},
    frozenset({"warfarin", "aspirin"}): {"interaction": True, "severity": "high", "details": "Increased risk of bleeding"},
    frozenset({"metformin", "atorvastatin"}): {"interaction": False, "severity": "none", "details": "No known interaction"},
    frozenset({"albuterol", "fluticasone"}): {"interaction": False, "severity": "none", "details": "Commonly used together safely"},
}

# Calendar mock store
_calendar_store: dict[str, dict] = {}
_event_counter = 0


def _exec_get_weather(params: dict) -> dict:
    rng = random.Random(RANDOM_SEED + _seed_hash(f"{params['city']}_{params['date']}"))
    unit = params.get("unit", "celsius")
    temp = rng.randint(-10, 40)
    if unit == "fahrenheit":
        temp = int(temp * 9 / 5 + 32)
    return {
        "city": params["city"],
        "date": params["date"],
        "temperature": temp,
        "unit": unit,
        "humidity": rng.randint(20, 95),
        "condition": rng.choice(CONDITIONS),
    }


def _exec_get_forecast(params: dict) -> dict:
    rng = random.Random(RANDOM_SEED + _seed_hash(f"{params['city']}_forecast"))
    days = params["days"]
    base_date = datetime.now()
    forecast = []
    for i in range(days):
        d = base_date + timedelta(days=i + 1)
        entry = {
            "date": d.strftime("%Y-%m-%d"),
            "temperature": rng.randint(-5, 38),
            "condition": rng.choice(CONDITIONS),
        }
        if params.get("include_hourly", False):
            entry["hourly"] = [
                {"hour": h, "temp": rng.randint(-5, 38)} for h in range(0, 24, 3)
            ]
        forecast.append(entry)
    return {"city": params["city"], "days": days, "forecast": forecast}


def _exec_get_historical(params: dict) -> dict:
    rng = random.Random(
        RANDOM_SEED + _seed_hash(f"{params['city']}_{params['start_date']}")
    )
    try:
        start = datetime.strptime(params["start_date"], "%Y-%m-%d")
        end = datetime.strptime(params["end_date"], "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    if end < start:
        return {"error": "end_date must be after start_date"}
    data = []
    current = start
    while current <= end:
        data.append({
            "date": current.strftime("%Y-%m-%d"),
            "temperature": rng.randint(-10, 40),
        })
        current += timedelta(days=1)
    return {"city": params["city"], "period": f"{params['start_date']} to {params['end_date']}", "data": data}


def _exec_create_event(params: dict) -> dict:
    global _event_counter
    _event_counter += 1
    event_id = f"EVT_{_event_counter:04d}"
    event = {
        "event_id": event_id,
        "title": params["title"],
        "date": params["date"],
        "time": params["time"],
        "duration_minutes": params["duration_minutes"],
        "attendees": params.get("attendees", []),
    }
    _calendar_store[event_id] = event
    return {"event_id": event_id, "status": "created"}


def _exec_get_events(params: dict) -> dict:
    cal_id = params.get("calendar_id", "primary")
    events = [
        {"title": e["title"], "time": e["time"], "duration": e["duration_minutes"], "event_id": eid}
        for eid, e in _calendar_store.items()
        if e["date"] == params["date"]
    ]
    # Always return some deterministic mock events if store is empty
    if not events:
        rng = random.Random(RANDOM_SEED + _seed_hash(params["date"]))
        titles = ["Team Standup", "Lunch Break", "Project Review", "1:1 Meeting", "Code Review"]
        n = rng.randint(1, 3)
        for i in range(n):
            events.append({
                "title": rng.choice(titles),
                "time": f"{rng.randint(8, 17):02d}:00",
                "duration": rng.choice([30, 60, 90]),
                "event_id": f"EVT_MOCK_{rng.randint(1000,9999)}",
            })
    return {"date": params["date"], "calendar_id": cal_id, "events": events}


def _exec_delete_event(params: dict) -> dict:
    event_id = params["event_id"]
    if event_id in _calendar_store:
        del _calendar_store[event_id]
        return {"success": True, "event_id": event_id, "notified": params.get("notify_attendees", True)}
    return {"success": False, "error": f"Event '{event_id}' not found"}


def _exec_get_patient_record(params: dict) -> dict:
    pid = params["patient_id"]
    if pid not in PATIENTS:
        return {"error": f"Patient '{pid}' not found"}
    record = dict(PATIENTS[pid])
    record["patient_id"] = pid
    fields = params.get("fields", ["all"])
    if "all" not in fields:
        record = {k: v for k, v in record.items() if k in fields or k == "patient_id"}
    return record


def _exec_check_drug_interaction(params: dict) -> dict:
    pair = frozenset({params["drug_a"].lower(), params["drug_b"].lower()})
    if pair in DRUG_INTERACTIONS:
        result = dict(DRUG_INTERACTIONS[pair])
        result["drug_a"] = params["drug_a"]
        result["drug_b"] = params["drug_b"]
        return result
    return {
        "drug_a": params["drug_a"],
        "drug_b": params["drug_b"],
        "interaction": False,
        "severity": "unknown",
        "details": "No interaction data available for this combination",
    }


def _exec_schedule_appointment(params: dict) -> dict:
    rng = random.Random(
        RANDOM_SEED + _seed_hash(f"{params['patient_id']}_{params['date']}")
    )
    appt_id = f"APPT_{rng.randint(10000, 99999)}"
    return {
        "appointment_id": appt_id,
        "patient_id": params["patient_id"],
        "doctor_id": params["doctor_id"],
        "date": params["date"],
        "appointment_type": params["appointment_type"],
        "status": "scheduled",
    }


# ── E-commerce Domain Executors ───────────────────────────────────────────────

ECOM_CATEGORIES = ["electronics", "books", "clothing", "home", "sports"]
ECOM_NAMES = {
    "electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "USB-C Cable", "Laptop Stand"],
    "books": ["Python Cookbook", "The Pragmatic Programmer", "Clean Code", "Deep Work", "Atomic Habits"],
    "clothing": ["Cotton T-Shirt", "Running Shoes", "Wool Sweater", "Denim Jeans", "Rain Jacket"],
    "home": ["Ceramic Mug", "LED Desk Lamp", "Throw Blanket", "Knife Set", "Cast Iron Pan"],
    "sports": ["Yoga Mat", "Resistance Bands", "Water Bottle", "Tennis Racket", "Jump Rope"],
}


def _exec_search_products(params: dict) -> dict:
    rng = random.Random(RANDOM_SEED + _seed_hash(f"search_{params['query']}_{params.get('category','any')}"))
    cat = params.get("category", "any")
    cats = ECOM_CATEGORIES if cat == "any" else [cat]
    results = []
    for c in cats:
        for name in ECOM_NAMES.get(c, []):
            price = rng.randint(10, 500)
            if "max_price" in params and price > params["max_price"]:
                continue
            results.append({
                "product_id": f"PRD_{_seed_hash(c + name) % 10000:04d}",
                "name": name,
                "category": c,
                "price_usd": price,
                "in_stock": rng.random() > 0.15,
            })
    return {
        "query": params["query"],
        "results": results[:10],
        "total_matches": len(results),
    }


def _exec_get_product(params: dict) -> dict:
    pid = params["product_id"]
    rng = random.Random(RANDOM_SEED + _seed_hash(pid))
    cat = rng.choice(ECOM_CATEGORIES)
    name = rng.choice(ECOM_NAMES[cat])
    return {
        "product_id": pid,
        "name": name,
        "category": cat,
        "price_usd": rng.randint(10, 500),
        "rating": round(rng.uniform(2.5, 5.0), 1),
        "num_reviews": rng.randint(5, 2000),
        "in_stock": rng.random() > 0.1,
        "description": f"High-quality {name.lower()} with 30-day return policy.",
    }


def _exec_place_order(params: dict) -> dict:
    rng = random.Random(RANDOM_SEED + _seed_hash(f"order_{params['product_id']}_{params['shipping_address']}"))
    order_id = f"ORD_{rng.randint(100000, 999999)}"
    unit_price = rng.randint(10, 500)
    total = unit_price * params["quantity"]
    return {
        "order_id": order_id,
        "product_id": params["product_id"],
        "quantity": params["quantity"],
        "shipping_address": params["shipping_address"],
        "payment_method": params.get("payment_method", "credit_card"),
        "unit_price_usd": unit_price,
        "total_usd": total,
        "status": "confirmed",
        "estimated_delivery_days": rng.randint(2, 10),
    }


# ── Tool Registry ─────────────────────────────────────────────────────────────

_EXECUTORS = {
    "get_weather": _exec_get_weather,
    "get_forecast": _exec_get_forecast,
    "get_historical": _exec_get_historical,
    "create_event": _exec_create_event,
    "get_events": _exec_get_events,
    "delete_event": _exec_delete_event,
    "get_patient_record": _exec_get_patient_record,
    "check_drug_interaction": _exec_check_drug_interaction,
    "schedule_appointment": _exec_schedule_appointment,
    "search_products": _exec_search_products,
    "get_product": _exec_get_product,
    "place_order": _exec_place_order,
}


class ToolExecutor:
    """Registry + executor for all domain tools."""

    def __init__(self, injector=None):
        self.schemas = dict(TOOL_SCHEMAS)
        self.executors = dict(_EXECUTORS)
        self.injector = injector  # Optional ParameterInjector instance
        logger.info("ToolExecutor initialized with %d tools (injector=%s)", len(self.schemas), injector is not None)

    def list_tools(self) -> list[dict]:
        """Return list of tool descriptions for the agent prompt."""
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            }
            for s in self.schemas.values()
        ]

    def get_tools_for_domain(self, domain: str) -> list[dict]:
        """Return tools for a specific domain."""
        return [
            {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}
            for s in self.schemas.values()
            if s.get("domain") == domain
        ]

    def execute(
        self,
        tool_name: str,
        params: dict,
        injector=None,
        injection_error_type: str | None = None,
        tracer=None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict:
        """Execute a tool with full validation. Never raises — returns error dict on failure.

        When injector is provided with injection_error_type, params are corrupted
        BEFORE validation/execution and the injection is logged to tracer.
        """
        logger.info("Executing tool=%s params=%s", tool_name, json.dumps(params, default=str))

        # Use instance injector if no explicit one passed
        active_injector = injector or self.injector

        # ── Inject parameter error BEFORE execution if requested ─────────
        if active_injector and injection_error_type and tool_name in self.schemas:
            schema = self.schemas[tool_name]
            params = active_injector.inject(
                tool_name=tool_name,
                params=params,
                error_type=injection_error_type,
                schema=schema,
                tracer=tracer,
                session_id=session_id,
                turn_id=turn_id,
            )
            logger.info("Post-injection params for %s: %s", tool_name, json.dumps(params, default=str))

        # Unknown tool
        if tool_name not in self.schemas:
            err = {"error": f"Unknown tool: '{tool_name}'"}
            logger.warning("Unknown tool: %s", tool_name)
            return {
                "result": err,
                "tool_name": tool_name,
                "raw_params": params,
                "validated_params": None,
                "validation_errors": [f"Unknown tool: '{tool_name}'"],
            }

        schema = self.schemas[tool_name]["parameters"]

        # Validate
        validation_errors = _validate_params(params, schema)
        if validation_errors:
            logger.warning("Validation errors for %s: %s", tool_name, validation_errors)
            return {
                "result": {"error": "; ".join(validation_errors)},
                "tool_name": tool_name,
                "raw_params": params,
                "validated_params": None,
                "validation_errors": validation_errors,
            }

        # Apply defaults
        validated_params = _apply_defaults(params, schema)

        # Execute
        try:
            result = self.executors[tool_name](validated_params)
        except Exception as e:
            logger.error("Tool execution error for %s: %s", tool_name, str(e))
            result = {"error": f"Execution error: {str(e)}"}

        logger.info("Tool %s result: %s", tool_name, json.dumps(result, default=str)[:200])
        return {
            "result": result,
            "tool_name": tool_name,
            "raw_params": params,
            "validated_params": validated_params,
            "validation_errors": [],
        }
