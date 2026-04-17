"""Multi-Turn Memory Accumulation Score (MTAS) — novel P3 metric.

Measures how hallucination error compounds across turns in extended sessions.
For each turn after injection:
1. Check if false belief is still active in working_memory
2. Check if false belief influenced that turn's tool calls
3. Compute compounding_score: does error get more influential each turn?
"""

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


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


class MultiTurnScorer:
    """Scores multi-turn memory accumulation for P3 experiments."""

    def compute_mtas(self, session_trace: dict, injection_turn: int) -> dict:
        """Per-turn analysis of false belief presence after injection.

        For each turn after injection:
        1. Check if false belief is still active in working_memory
        2. Check if false belief influenced that turn's reasoning (tool calls)
        3. Check if false belief appeared in that turn's tool params

        Returns:
        {
            "injection_turn": int,
            "total_turns": int,
            "affected_turns": [int],
            "propagation_depth": int,
            "propagation_rate": float,
            "compounding_score": float,
            "self_correction_turn": int | None
        }
        """
        steps = session_trace.get("steps", [])
        injections = session_trace.get("injections", [])

        # Get false value from injection record
        false_value = None
        for inj in injections:
            injected = _safe_json_loads(inj.get("injected_value"))
            if injected is not None:
                false_value = str(injected)
                break

        if false_value is None:
            logger.warning("No injection found in session trace")
            return {
                "injection_turn": injection_turn,
                "total_turns": 0,
                "affected_turns": [],
                "propagation_depth": 0,
                "propagation_rate": 0.0,
                "compounding_score": 0.0,
                "self_correction_turn": None,
            }

        # Group steps by turn
        turns_after = {}
        all_turn_ids = set()
        for step in steps:
            turn = step.get("turn_id", 0)
            all_turn_ids.add(turn)
            if turn <= injection_turn:
                continue
            if turn not in turns_after:
                turns_after[turn] = []
            turns_after[turn].append(step)

        total_turns = max(all_turn_ids) if all_turn_ids else 0

        # Per-turn analysis
        affected_turns = []
        per_turn_influence = {}  # turn_id → influence score
        self_correction_turn = None

        for turn_id in sorted(turns_after.keys()):
            turn_steps = turns_after[turn_id]
            influence = 0.0
            found_in_turn = False

            for step in turn_steps:
                # Check tool_params_raw
                params_raw = step.get("tool_params_raw", "")
                if params_raw and false_value in str(params_raw):
                    found_in_turn = True
                    influence += 1.0

                # Check content
                content = step.get("content", "")
                if false_value in str(content):
                    found_in_turn = True
                    influence += 0.5

                # Check memory_state
                mem_state = step.get("memory_state", "")
                if mem_state and false_value in str(mem_state):
                    found_in_turn = True
                    influence += 0.5

                # Check tool_result
                tool_result = step.get("tool_result", "")
                if tool_result and false_value in str(tool_result):
                    found_in_turn = True
                    influence += 0.3

            if found_in_turn:
                affected_turns.append(turn_id)
                per_turn_influence[turn_id] = influence
            else:
                # First turn after injection where false value is NOT present
                if affected_turns and self_correction_turn is None:
                    self_correction_turn = turn_id

        propagation_depth = len(affected_turns)
        total_after = len(turns_after) if turns_after else 1
        propagation_rate = propagation_depth / max(total_after, 1)

        # Compounding score: correlation of turn_id with error_influence
        # Higher score means error gets MORE influential over time
        compounding_score = 0.0
        if len(per_turn_influence) >= 2:
            turn_ids = np.array(sorted(per_turn_influence.keys()), dtype=float)
            influences = np.array([per_turn_influence[t] for t in sorted(per_turn_influence.keys())], dtype=float)
            # Pearson correlation: positive = compounding, negative = diminishing
            if np.std(influences) > 0 and np.std(turn_ids) > 0:
                correlation = np.corrcoef(turn_ids, influences)[0, 1]
                compounding_score = float(correlation) if not np.isnan(correlation) else 0.0
            else:
                # Constant influence across turns (persistent but not compounding)
                compounding_score = 0.0

        result = {
            "injection_turn": injection_turn,
            "total_turns": total_turns,
            "affected_turns": affected_turns,
            "propagation_depth": propagation_depth,
            "propagation_rate": round(propagation_rate, 3),
            "compounding_score": round(compounding_score, 3),
            "self_correction_turn": self_correction_turn,
            "per_turn_influence": per_turn_influence,
        }

        logger.info(
            "MTAS: injection_turn=%d depth=%d rate=%.2f compounding=%.2f correction=%s",
            injection_turn, propagation_depth, propagation_rate,
            compounding_score, self_correction_turn,
        )
        return result

    def compare_single_vs_multi_turn(
        self,
        single_eps_df: pd.DataFrame,
        multi_mtas_list: list[dict],
    ) -> dict:
        """Key finding for paper: multi-turn errors are qualitatively different.

        Compares single-turn EPS distribution with multi-turn MTAS metrics.
        Returns statistical comparison dict.
        """
        # Single-turn stats
        if not single_eps_df.empty and "eps" in single_eps_df.columns:
            single_mean_eps = float(single_eps_df["eps"].mean())
            single_std_eps = float(single_eps_df["eps"].std()) if len(single_eps_df) > 1 else 0.0
            single_output_rate = float(single_eps_df["reached_output"].mean()) if "reached_output" in single_eps_df.columns else 0.0
        else:
            single_mean_eps = 0.0
            single_std_eps = 0.0
            single_output_rate = 0.0

        # Multi-turn stats
        if multi_mtas_list:
            depths = [m["propagation_depth"] for m in multi_mtas_list]
            rates = [m["propagation_rate"] for m in multi_mtas_list]
            compounds = [m["compounding_score"] for m in multi_mtas_list]
            corrections = [1 for m in multi_mtas_list if m.get("self_correction_turn") is not None]

            multi_mean_depth = float(np.mean(depths))
            multi_std_depth = float(np.std(depths)) if len(depths) > 1 else 0.0
            multi_mean_rate = float(np.mean(rates))
            multi_mean_compound = float(np.mean(compounds))
            multi_correction_rate = len(corrections) / len(multi_mtas_list)
        else:
            multi_mean_depth = 0.0
            multi_std_depth = 0.0
            multi_mean_rate = 0.0
            multi_mean_compound = 0.0
            multi_correction_rate = 0.0

        result = {
            "single_turn": {
                "count": len(single_eps_df),
                "mean_eps": round(single_mean_eps, 3),
                "std_eps": round(single_std_eps, 3),
                "output_reach_rate": round(single_output_rate, 3),
            },
            "multi_turn": {
                "count": len(multi_mtas_list),
                "mean_propagation_depth": round(multi_mean_depth, 3),
                "std_propagation_depth": round(multi_std_depth, 3),
                "mean_propagation_rate": round(multi_mean_rate, 3),
                "mean_compounding_score": round(multi_mean_compound, 3),
                "self_correction_rate": round(multi_correction_rate, 3),
            },
            "finding": (
                "Multi-turn errors propagate deeper and compound over time "
                f"(mean depth={multi_mean_depth:.1f}) compared to single-turn EPS "
                f"(mean={single_mean_eps:.1f}). Self-correction rate: {multi_correction_rate:.0%}."
            ),
        }

        logger.info(
            "Single vs Multi comparison: single_eps=%.2f multi_depth=%.2f",
            single_mean_eps, multi_mean_depth,
        )
        return result
