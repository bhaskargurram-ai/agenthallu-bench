"""Error Propagation Score (EPS) — novel metric defined in this paper.

Definition:
EPS measures how far a hallucination travels through the agent pipeline
before becoming detectable or manifesting in final output.

EPS = 0: Error is detectable at the stage where it originates (best case)
EPS = 1: Error propagates one stage (e.g., planning → tool use)
EPS = 2: Error propagates two stages (e.g., planning → tool use → memory)
EPS = 3: Error only appears in final output (worst case, hardest to detect)

Weighted EPS (wEPS):
wEPS = EPS * severity_weight * detectability_weight

severity_weight:
- type_mismatch: 0.5 (usually caught by API)
- out_of_range: 0.7 (sometimes caught)
- missing_required: 0.9 (often silent failure)
- semantic_wrong: 1.0 (never caught by schema)

detectability_weight:
- If error appears in tool return value: 0.6
- If error is silent (tool succeeds with wrong result): 1.0
"""

import json
import logging
from typing import Any, Optional

import pandas as pd

from config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Stage order as defined in the spec
STAGE_ORDER = [
    "planning",
    "tool_selection",
    "parameter_generation",
    "tool_execution",
    "memory_write",
    "memory_retrieval",
    "output_generation",
]

SEVERITY_WEIGHTS = {
    "type_mismatch": 0.5,
    "out_of_range": 0.7,
    "missing_required": 0.9,
    "semantic_wrong": 1.0,
}

# Step type → stage mapping
_STEP_TYPE_TO_STAGE = {
    "thought": "planning",
    "action": "tool_selection",
    "observation": "tool_execution",
    "retrieval": "memory_retrieval",
    "memory_write": "memory_write",
    "final": "output_generation",
}


def _safe_json_loads(val: Any) -> Any:
    """Parse JSON string, or return as-is if already parsed or None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def _extract_error_type(injection_type: str) -> str:
    """Extract base error type from injection_type string like 'p2_type_mismatch'."""
    for et in SEVERITY_WEIGHTS:
        if et in injection_type:
            return et
    return injection_type


def _error_value_in_step(step: dict, injected_value: Any) -> bool:
    """Check if the injected error value appears in a step's trace fields."""
    injected_str = str(injected_value)
    if not injected_str or injected_str in ("None", "null", ""):
        return False

    fields_to_check = [
        step.get("content", ""),
        step.get("tool_params_raw", ""),
        step.get("tool_params_validated", ""),
        step.get("tool_result", ""),
        step.get("param_errors", ""),
        step.get("memory_state", ""),
        step.get("retrieval_results", ""),
    ]
    for field in fields_to_check:
        if field and injected_str in str(field):
            return True
    return False


def _tool_result_has_error(step: dict) -> bool:
    """Check if a step's tool_result contains an error dict."""
    result = _safe_json_loads(step.get("tool_result"))
    if isinstance(result, dict) and "error" in result:
        return True
    return False


def _step_to_stage(step: dict) -> str:
    """Map a step to its pipeline stage."""
    step_type = step.get("step_type", "")

    # Action steps with tool_name → tool_selection or parameter_generation
    if step_type == "action":
        params_raw = step.get("tool_params_raw")
        if params_raw:
            return "parameter_generation"
        return "tool_selection"

    return _STEP_TYPE_TO_STAGE.get(step_type, "planning")


class EPSScorer:
    """Computes Error Propagation Score for traced agent sessions."""

    def compute_eps(self, session_trace: dict, injection_record: dict) -> dict:
        """Compute EPS for a single session.

        Inputs:
        - session_trace: full trace dict from TraceLogger.get_session_trace()
        - injection_record: {injection_type, target_stage, injected_value, ...}

        Algorithm — impact-based propagation tracking:
        1. Origin = parameter_generation (where P2 injection occurs)
        2. Check tool_execution: did validation/tool catch the error?
        3. Check planning: did model reason about the error in subsequent thoughts?
        4. Check output_generation: did the final answer end up wrong?
        5. EPS = number of stages beyond origin that show error impact

        Error impact signals (not just literal string matching):
        - tool_execution: param_errors or tool error in result
        - planning: model mentions error/invalid/wrong in post-injection thoughts
        - output_generation: final_correct == False (wrong answer due to injection)

        Returns dict with eps, weps, origin_stage, manifest_stage, etc.
        """
        steps = session_trace.get("steps", [])
        injections = session_trace.get("injections", [])

        # Use provided injection_record or first from trace
        if not injection_record and injections:
            injection_record = injections[0]

        if not injection_record:
            return {
                "eps": 0, "weps": 0.0,
                "origin_stage": None, "manifest_stage": None,
                "stages_traversed": [], "detectable_at_step": None,
                "reached_output": False, "error_type": None,
            }

        injection_type = injection_record.get("injection_type", "")
        error_type = _extract_error_type(injection_type)
        target_stage = injection_record.get("target_stage", "parameter_generation")
        injected_value = _safe_json_loads(injection_record.get("injected_value"))

        # Find the injection step number
        injection_step = injection_record.get("step_number", 0)

        # Determine origin stage
        origin_stage = target_stage
        if origin_stage not in STAGE_ORDER:
            origin_stage = "parameter_generation"
        origin_idx = STAGE_ORDER.index(origin_stage)

        # Track error impact across stages
        stages_with_error = set()
        stages_with_error.add(origin_stage)
        detectable_at_step = None
        reached_output = False

        # Signals we collect from the trace
        has_param_errors = False
        has_tool_error = False
        has_error_in_thought = False
        model_retried_tool = False
        final_correct = session_trace.get("final_correct", None)

        # Error-related keywords in model reasoning
        ERROR_KEYWORDS = {"error", "invalid", "failed", "missing", "wrong", "unable",
                          "cannot", "couldn't", "not found", "not valid", "apologize",
                          "unfortunately", "don't have", "no result", "unavailable"}

        post_injection = False
        tool_call_count = 0

        for step in steps:
            step_num = step.get("step_number", 0)
            step_type = step.get("step_type", "")

            # Track whether we're past the injection point
            if step_num > injection_step or injection_step == 0:
                post_injection = True

            if not post_injection:
                continue

            stage = _step_to_stage(step)

            # --- Literal value match (original approach, still useful) ---
            if _error_value_in_step(step, injected_value):
                stages_with_error.add(stage)

            # --- Impact-based detection ---

            # Tool execution stage: validation errors or tool errors
            param_errors = _safe_json_loads(step.get("param_errors"))
            if param_errors and isinstance(param_errors, list) and len(param_errors) > 0:
                has_param_errors = True
                stages_with_error.add("tool_execution")
                if detectable_at_step is None:
                    detectable_at_step = step_num

            if _tool_result_has_error(step):
                has_tool_error = True
                stages_with_error.add("tool_execution")
                if detectable_at_step is None:
                    detectable_at_step = step_num

            # Planning stage: model reasoning about errors post-injection
            if step_type == "thought" and post_injection:
                content = str(step.get("content", "")).lower()
                if any(kw in content for kw in ERROR_KEYWORDS):
                    has_error_in_thought = True
                    stages_with_error.add("planning")

            # Track tool retries (model calling tools again after error)
            if step_type == "action":
                tool_call_count += 1
                if tool_call_count > 1:
                    model_retried_tool = True

        # Output generation: if final answer is wrong, error propagated to output
        if final_correct is False:
            stages_with_error.add("output_generation")
            reached_output = True

        # Also check literal match in final answer
        final_answer = session_trace.get("final_answer", "")
        if final_answer and injected_value and str(injected_value) in str(final_answer):
            stages_with_error.add("output_generation")
            reached_output = True

        # Compute EPS: count stages traversed from origin
        stages_traversed = []
        for stage in STAGE_ORDER:
            if stage in stages_with_error:
                stage_idx = STAGE_ORDER.index(stage)
                if stage_idx >= origin_idx:
                    stages_traversed.append(stage)

        # EPS = number of stages beyond origin
        eps = max(0, len(stages_traversed) - 1)

        # Determine manifest stage (last stage with error)
        manifest_stage = stages_traversed[-1] if stages_traversed else origin_stage

        # Compute weighted EPS
        severity_weight = SEVERITY_WEIGHTS.get(error_type, 1.0)
        detectability_weight = 0.6 if detectable_at_step is not None else 1.0
        weps = eps * severity_weight * detectability_weight

        result = {
            "eps": eps,
            "weps": round(weps, 3),
            "origin_stage": origin_stage,
            "manifest_stage": manifest_stage,
            "stages_traversed": stages_traversed,
            "detectable_at_step": detectable_at_step,
            "reached_output": reached_output,
            "error_type": error_type,
        }

        logger.info(
            "EPS computed: session=%s eps=%d weps=%.3f origin=%s manifest=%s",
            session_trace.get("session_id", "?"), eps, weps, origin_stage, manifest_stage,
        )
        return result

    def compute_batch_eps(self, session_ids: list, db_path: str) -> pd.DataFrame:
        """Compute EPS for all sessions. Returns DataFrame with one row per session.

        Columns: session_id, model, domain, error_type, eps, weps, origin_stage,
                 manifest_stage, reached_output, difficulty
        """
        from tracer.trace_logger import TraceLogger
        from tracer.trace_schema import GroundTruthRecord

        tracer = TraceLogger(db_path)
        tracer.init_db()

        rows = []
        for sid in session_ids:
            trace = tracer.get_session_trace(sid)
            if not trace:
                logger.warning("No trace found for session %s", sid)
                continue

            injections = trace.get("injections", [])
            injection_record = injections[0] if injections else {}

            eps_result = self.compute_eps(trace, injection_record)

            # Look up difficulty from ground_truth
            task_id = trace.get("task_id", "")
            gt = tracer.session.query(GroundTruthRecord).filter_by(task_id=task_id).first()
            difficulty = gt.difficulty if gt else "unknown"

            rows.append({
                "session_id": sid,
                "model": trace.get("model", ""),
                "domain": trace.get("domain", ""),
                "error_type": eps_result["error_type"],
                "eps": eps_result["eps"],
                "weps": eps_result["weps"],
                "origin_stage": eps_result["origin_stage"],
                "manifest_stage": eps_result["manifest_stage"],
                "reached_output": eps_result["reached_output"],
                "difficulty": difficulty,
                "stages_traversed": ",".join(eps_result["stages_traversed"]),
            })

        tracer.close()

        df = pd.DataFrame(rows)
        logger.info("Batch EPS computed for %d sessions", len(df))
        return df

    def summarize_eps_by_error_type(self, eps_df: pd.DataFrame) -> dict:
        """Return mean/std EPS broken down by error_type, model, domain, difficulty."""
        if eps_df.empty:
            return {"by_error_type": {}, "by_model": {}, "by_domain": {}, "by_difficulty": {}}

        summary = {}

        for group_col in ["error_type", "model", "domain", "difficulty"]:
            if group_col not in eps_df.columns:
                summary[f"by_{group_col}"] = {}
                continue
            grouped = eps_df.groupby(group_col)["eps"].agg(["mean", "std", "count"])
            summary[f"by_{group_col}"] = {
                idx: {"mean": round(row["mean"], 3), "std": round(row["std"], 3) if pd.notna(row["std"]) else 0.0, "count": int(row["count"])}
                for idx, row in grouped.iterrows()
            }

        logger.info("EPS summary: %s", json.dumps(summary, default=str)[:500])
        return summary
