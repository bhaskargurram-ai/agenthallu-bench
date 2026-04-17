"""P2: Parameter perturbation injector.

Intercepts tool calls BEFORE execution and corrupts parameters in 4 ways:
- type_mismatch: change int→str, str→int, bool→str
- out_of_range: set numeric outside bounds, enum outside list
- missing_required: remove a required parameter
- semantic_wrong: correct type/range but wrong meaning (different city, shifted date, etc.)

All injections are logged to the tracer.
"""

import json
import logging
import random
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Optional

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


# ── Semantic substitutions for semantic_wrong injection ──────────────────────

_CITY_ALTERNATIVES = {
    "Paris": "Lyon", "London": "Manchester", "Tokyo": "Osaka",
    "New York": "Los Angeles", "São Paulo": "Rio de Janeiro",
    "Berlin": "Munich", "Sydney": "Melbourne", "Toronto": "Vancouver",
    "Chicago": "Houston", "Mumbai": "Delhi", "Lagos": "Abuja",
}
_CITY_FALLBACK = "Reykjavik"

_PATIENT_ALTERNATIVES = {
    "P001": "P002", "P002": "P003", "P003": "P004", "P004": "P005", "P005": "P001",
}

_APPOINTMENT_TYPE_ALTERNATIVES = {
    "checkup": "follow_up", "follow_up": "specialist",
    "specialist": "checkup", "emergency": "checkup",
}

_DRUG_ALTERNATIVES = {
    "metformin": "glipizide", "lisinopril": "losartan",
    "ibuprofen": "naproxen", "sertraline": "fluoxetine",
    "sumatriptan": "rizatriptan", "albuterol": "ipratropium",
    "aspirin": "acetaminophen", "warfarin": "heparin",
    "atorvastatin": "simvastatin", "cetirizine": "loratadine",
    "fluticasone": "budesonide",
}


class ParameterInjector:
    """Intercepts tool calls BEFORE execution and corrupts parameters.

    Called by tool_executor after agent produces Action Input but before execute().
    Logs injection to trace.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        logger.info("ParameterInjector initialized with seed=%d", seed)

    def inject(
        self,
        tool_name: str,
        params: dict,
        error_type: str,
        schema: dict,
        tracer: Optional[Any] = None,
        session_id: Optional[str] = None,
        turn_id: int = 0,
    ) -> dict:
        """Return corrupted params dict. Logs original + corrupted to injections table."""
        original_params = deepcopy(params)
        param_schema = schema.get("parameters", schema)

        if error_type == "type_mismatch":
            corrupted = self._type_mismatch(params, param_schema)
        elif error_type == "out_of_range":
            corrupted = self._out_of_range(params, param_schema)
        elif error_type == "missing_required":
            corrupted = self._missing_required(params, param_schema)
        elif error_type == "semantic_wrong":
            corrupted = self._semantic_wrong(params, param_schema, tool_name)
        else:
            raise ValueError(f"Unknown error type: {error_type}")

        logger.info(
            "P2 injection [%s] on %s: %s → %s",
            error_type, tool_name,
            json.dumps(original_params, default=str),
            json.dumps(corrupted, default=str),
        )

        # Log to tracer
        if tracer and session_id:
            tracer.log_injection(
                session_id=session_id,
                injection_type=f"p2_{error_type}",
                target_stage="parameter_generation",
                original=original_params,
                injected=corrupted,
                turn_id=turn_id,
            )

        return corrupted

    # ── Error Type Implementations ───────────────────────────────────────────

    def _type_mismatch(self, params: dict, schema: dict) -> dict:
        """Find a param that expects int/float → set to string version.
        Find a param that expects string → set to int 0.
        Find a param that expects bool → set to string "true".

        Priority: int→str first, then bool→str, then str→int (spec order).
        """
        result = deepcopy(params)
        props = schema.get("properties", {})

        word_map = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        }

        # Pass 1: prefer int → string (most common P2 error)
        for key, value in list(result.items()):
            if key not in props:
                continue
            if props[key].get("type") == "integer" and isinstance(value, int):
                result[key] = word_map.get(value, str(value))
                logger.info("type_mismatch: %s: %r → %r", key, value, result[key])
                return result

        # Pass 2: bool → string
        for key, value in list(result.items()):
            if key not in props:
                continue
            if props[key].get("type") == "boolean" and isinstance(value, bool):
                result[key] = "true" if value else "false"
                logger.info("type_mismatch: %s: %r → %r", key, value, result[key])
                return result

        # Pass 3: string → int
        for key, value in list(result.items()):
            if key not in props:
                continue
            if props[key].get("type") == "string" and isinstance(value, str):
                result[key] = 0
                logger.info("type_mismatch: %s: %r → %r", key, value, result[key])
                return result

        # Fallback: corrupt first param
        if result:
            first_key = next(iter(result))
            result[first_key] = "CORRUPTED_TYPE"
            logger.info("type_mismatch fallback: %s → 'CORRUPTED_TYPE'", first_key)
        return result

    def _out_of_range(self, params: dict, schema: dict) -> dict:
        """Find numeric param with min/max → set to max*10 or min-100.
        Find enum param → replace with value NOT in enum list.
        """
        result = deepcopy(params)
        props = schema.get("properties", {})

        # First try numeric with bounds
        for key, value in list(result.items()):
            if key not in props:
                continue
            prop = props[key]
            if prop.get("type") == "integer" and isinstance(value, int):
                if "maximum" in prop:
                    result[key] = prop["maximum"] * 10
                    logger.info("out_of_range: %s: %r → %r (max*10)", key, value, result[key])
                    return result
                elif "minimum" in prop:
                    result[key] = prop["minimum"] - 100
                    logger.info("out_of_range: %s: %r → %r (min-100)", key, value, result[key])
                    return result

        # Then try enum
        for key, value in list(result.items()):
            if key not in props:
                continue
            prop = props[key]
            if "enum" in prop:
                invalid_values = ["INVALID_ENUM", "kelvin", "unknown_type", "surgery"]
                for iv in invalid_values:
                    if iv not in prop["enum"]:
                        result[key] = iv
                        logger.info("out_of_range: %s: %r → %r (invalid enum)", key, value, result[key])
                        return result

        # Fallback: add a very large number to first numeric
        for key, value in list(result.items()):
            if isinstance(value, int):
                result[key] = value * 1000
                logger.info("out_of_range fallback: %s: %r → %r", key, value, result[key])
                return result

        return result

    def _missing_required(self, params: dict, schema: dict) -> dict:
        """Remove the first required param from dict."""
        result = deepcopy(params)
        required = schema.get("required", [])

        # Choose the most semantically important required param (first one present)
        for field in required:
            if field in result:
                removed_value = result.pop(field)
                logger.info("missing_required: removed '%s' (was %r)", field, removed_value)
                return result

        return result

    def _semantic_wrong(self, params: dict, schema: dict, tool_name: str) -> dict:
        """Correct type, correct range, but wrong meaning.
        - date shifted by 180 days
        - city swapped to different city
        - patient_id swapped to different patient
        """
        result = deepcopy(params)

        # Try city swap
        if "city" in result and isinstance(result["city"], str):
            original = result["city"]
            result["city"] = _CITY_ALTERNATIVES.get(original, _CITY_FALLBACK)
            logger.info("semantic_wrong: city: %r → %r", original, result["city"])
            return result

        # Try date shift (180 days)
        for key in ["date", "start_date", "end_date"]:
            if key in result and isinstance(result[key], str):
                try:
                    dt = datetime.strptime(result[key], "%Y-%m-%d")
                    shifted = dt + timedelta(days=180)
                    original = result[key]
                    result[key] = shifted.strftime("%Y-%m-%d")
                    logger.info("semantic_wrong: %s: %r → %r", key, original, result[key])
                    return result
                except ValueError:
                    continue

        # Try patient_id swap
        if "patient_id" in result:
            original = result["patient_id"]
            result["patient_id"] = _PATIENT_ALTERNATIVES.get(original, "P999")
            logger.info("semantic_wrong: patient_id: %r → %r", original, result["patient_id"])
            return result

        # Try appointment_type swap
        if "appointment_type" in result:
            original = result["appointment_type"]
            result["appointment_type"] = _APPOINTMENT_TYPE_ALTERNATIVES.get(original, "checkup")
            logger.info("semantic_wrong: appointment_type: %r → %r", original, result["appointment_type"])
            return result

        # Try drug swap
        for key in ["drug_a", "drug_b"]:
            if key in result and isinstance(result[key], str):
                original = result[key]
                result[key] = _DRUG_ALTERNATIVES.get(original.lower(), "acetaminophen")
                logger.info("semantic_wrong: %s: %r → %r", key, original, result[key])
                return result

        # Try event_id swap
        if "event_id" in result:
            original = result["event_id"]
            result["event_id"] = "EVT_9999"
            logger.info("semantic_wrong: event_id: %r → %r", original, result["event_id"])
            return result

        # Fallback: modify first string param
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = value + "_WRONG"
                logger.info("semantic_wrong fallback: %s: %r → %r", key, value, result[key])
                return result

        return result
