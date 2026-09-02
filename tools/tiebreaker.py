"""
tools/tiebreaker.py

Gemini tiebreaker for the hybrid inference mode.

When the base backend (Ollama/HF) fails to reach unanimity after VOTING_MAX_ROUNDS,
these functions call Gemini sequentially to break the tie.

Protocol (per stage — classify or extract):
  1. Call Gemini once (blind, same prompt as regular agents).
  2. If the result matches the Ollama plurality → use it ("matched_majority").
  3. If not → call Gemini (GEMINI_TIEBREAKER_CALLS - 1) more times and check unanimity
     across ALL calls.
  4. If unanimous → use that value ("tiebreaker_unanimous").
  5. Otherwise → low_confidence; return None.

Both functions return:
    (final_value_or_None, tiebreaker_log: dict, total_in_tokens: int, total_out_tokens: int)
"""
import logging
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)


def tiebreak_string(
    majority: str,
    gemini_fn: Callable,
    n_calls: Optional[int] = None,
) -> tuple:
    """
    Tiebreak a single string classification using Gemini.

    Args:
        majority:   The Ollama plurality value (already normalised).
        gemini_fn:  callable() → (normalised_value: str, input_tokens: int, output_tokens: int)
                    Uses the same prompt as the regular agents; Gemini answers blind.
        n_calls:    Total Gemini calls to make (defaults to config.GEMINI_TIEBREAKER_CALLS).

    Returns:
        (final_value_or_None, tiebreaker_log, total_input_tokens, total_output_tokens)
    """
    if n_calls is None:
        n_calls = config.GEMINI_TIEBREAKER_CALLS

    total_in, total_out = 0, 0
    calls_log = []

    # ── Call 1 ──────────────────────────────────────────────────────────────
    try:
        value_1, in_tok, out_tok = gemini_fn()
    except Exception as exc:
        logger.warning(f"[Tiebreaker] Gemini call 1 failed: {exc}")
        value_1, in_tok, out_tok = "", 0, 0
    total_in += in_tok
    total_out += out_tok
    matched = value_1 == majority
    calls_log.append({
        "call_num": 1,
        "response": value_1,
        "matched_majority": matched,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    })
    logger.info(f"[Tiebreaker] call 1: value={value_1!r}  matched_majority={matched}")

    if matched:
        return value_1, {
            "invoked": True,
            "ollama_plurality": majority,
            "calls": calls_log,
            "outcome": "matched_majority",
            "final_value": value_1,
        }, total_in, total_out

    # ── Calls 2..n_calls ────────────────────────────────────────────────────
    all_values = [value_1]
    for i in range(2, n_calls + 1):
        try:
            value_i, in_tok, out_tok = gemini_fn()
        except Exception as exc:
            logger.warning(f"[Tiebreaker] Gemini call {i} failed: {exc}")
            value_i, in_tok, out_tok = "", 0, 0
        total_in += in_tok
        total_out += out_tok
        all_values.append(value_i)
        calls_log.append({
            "call_num": i,
            "response": value_i,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
        logger.info(f"[Tiebreaker] call {i}: value={value_i!r}")

    # ── Evaluate unanimity across all n_calls ───────────────────────────────
    unanimous = len(set(all_values)) == 1
    outcome = "tiebreaker_unanimous" if unanimous else "low_confidence"
    final_value = all_values[0] if unanimous else None

    logger.info(f"[Tiebreaker] outcome={outcome!r}  final_value={final_value!r}")
    return final_value, {
        "invoked": True,
        "ollama_plurality": majority,
        "calls": calls_log,
        "outcome": outcome,
        "final_value": final_value,
    }, total_in, total_out


def tiebreak_fields(
    majority_fields: dict,
    gemini_fn: Callable,
    voted_field_names: list,
    n_calls: Optional[int] = None,
) -> tuple:
    """
    Tiebreak a set of voted extraction fields using Gemini.

    "Match" is defined as ALL voted fields matching simultaneously.
    "Unanimity" (for extra calls) is defined the same way across all n_calls.

    Args:
        majority_fields:    Dict of Ollama plurality values (already normalised),
                            keyed by voted field name.
        gemini_fn:          callable() → (fields_dict: dict, input_tokens: int, output_tokens: int)
                            fields_dict must contain normalised values for all voted_field_names.
                            Non-voted fields (e.g. details) may also be present and are taken
                            from call 1's response.
        voted_field_names:  List of field names to compare (voted fields only).
        n_calls:            Total Gemini calls (defaults to config.GEMINI_TIEBREAKER_CALLS).

    Returns:
        (final_fields_or_None, tiebreaker_log, total_input_tokens, total_output_tokens)
        final_fields contains all fields from call 1 (non-voted included) when successful.
    """
    if n_calls is None:
        n_calls = config.GEMINI_TIEBREAKER_CALLS

    total_in, total_out = 0, 0
    calls_log = []
    normalised_majority = {f: majority_fields.get(f) for f in voted_field_names}

    def _voted_subset(fields: dict) -> dict:
        return {f: fields.get(f) for f in voted_field_names}

    def _all_match(a: dict, b: dict) -> bool:
        return all(str(a.get(f)) == str(b.get(f)) for f in voted_field_names)

    # ── Call 1 ──────────────────────────────────────────────────────────────
    first_full_fields: dict = {}
    try:
        first_full_fields, in_tok, out_tok = gemini_fn()
    except Exception as exc:
        logger.warning(f"[Tiebreaker] Gemini fields call 1 failed: {exc}")
        in_tok, out_tok = 0, 0
    total_in += in_tok
    total_out += out_tok
    voted_1 = _voted_subset(first_full_fields)
    matched = _all_match(voted_1, normalised_majority)
    calls_log.append({
        "call_num": 1,
        "response": voted_1,
        "matched_majority": matched,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    })
    logger.info(f"[Tiebreaker] fields call 1: voted={voted_1}  matched_majority={matched}")

    if matched:
        non_voted = {k: v for k, v in first_full_fields.items() if k not in voted_field_names}
        final_fields = {**non_voted, **voted_1}
        return final_fields, {
            "invoked": True,
            "ollama_plurality": normalised_majority,
            "calls": calls_log,
            "outcome": "matched_majority",
            "final_value": voted_1,
        }, total_in, total_out

    # ── Calls 2..n_calls ────────────────────────────────────────────────────
    all_voted = [voted_1]
    for i in range(2, n_calls + 1):
        raw_i: dict = {}
        try:
            raw_i, in_tok, out_tok = gemini_fn()
        except Exception as exc:
            logger.warning(f"[Tiebreaker] Gemini fields call {i} failed: {exc}")
            in_tok, out_tok = 0, 0
        total_in += in_tok
        total_out += out_tok
        voted_i = _voted_subset(raw_i)
        all_voted.append(voted_i)
        calls_log.append({
            "call_num": i,
            "response": voted_i,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
        logger.info(f"[Tiebreaker] fields call {i}: voted={voted_i}")

    # ── Evaluate unanimity across all n_calls ───────────────────────────────
    unanimous = all(_all_match(all_voted[0], v) for v in all_voted[1:])
    if unanimous:
        non_voted = {k: v for k, v in first_full_fields.items() if k not in voted_field_names}
        final_fields = {**non_voted, **all_voted[0]}
        outcome = "tiebreaker_unanimous"
        logger.info(f"[Tiebreaker] fields outcome={outcome!r}  final_value={all_voted[0]}")
        return final_fields, {
            "invoked": True,
            "ollama_plurality": normalised_majority,
            "calls": calls_log,
            "outcome": outcome,
            "final_value": all_voted[0],
        }, total_in, total_out

    outcome = "low_confidence"
    logger.info(f"[Tiebreaker] fields outcome={outcome!r}")
    return None, {
        "invoked": True,
        "ollama_plurality": normalised_majority,
        "calls": calls_log,
        "outcome": outcome,
        "final_value": None,
    }, total_in, total_out
