import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)


def _run_parallel(fn: Callable, n: int) -> list:
    """Run fn() n times in parallel. Failed calls return None."""
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(fn) for _ in range(n)]
        results = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.warning(f"[voting] parallel call failed: {e}")
                results.append(None)
        return results


def vote_for_string(
    fn: Callable,
    normalize: Callable[[str], str],
    n: int = None,
    max_rounds: int = None,
    gemini_confirm_fn: Optional[Callable] = None,
) -> tuple:
    """
    Run fn() n times in parallel each round. fn() returns (raw_text, in_tok, out_tok).
    normalize(raw_text) maps raw to a canonical string.
    All n must agree for consensus. After max_rounds, take plurality across all rounds.

    gemini_confirm_fn: optional callable() → (normalized_str, in_tok, out_tok).
    When provided, a unanimous Ollama round also requires Gemini confirmation before
    accepting. If Gemini disagrees, the round is treated as non-unanimous and the loop
    continues. Only called on unanimous rounds; skipped otherwise.

    Returns (consensus, rounds_log, total_input_tokens, total_output_tokens,
             gemini_input_tokens, gemini_output_tokens).
    """
    if n is None:
        n = config.VOTING_N
    if max_rounds is None:
        max_rounds = config.VOTING_MAX_ROUNDS

    total_in, total_out = 0, 0
    gemini_in, gemini_out = 0, 0
    rounds_log = []
    final_value = "Unknown"
    all_normalized_values: list = []

    for round_num in range(1, max_rounds + 1):
        logger.info(f"[Voting] ── Round {round_num}/{max_rounds} ──")
        results = _run_parallel(fn, n)
        round_responses = []
        normalized_values = []

        for r in results:
            raw, in_tok, out_tok = ("", 0, 0) if r is None else r
            total_in += in_tok
            total_out += out_tok
            norm = normalize(raw)
            round_responses.append({"raw": raw[:200], "normalized": norm})
            normalized_values.append(norm)

        all_normalized_values.extend(normalized_values)

        counter = Counter(normalized_values)
        plurality = counter.most_common(1)[0][0]
        unanimous = len(counter) == 1

        round_entry = {"round": round_num, "responses": round_responses, "unanimous": unanimous}
        if not unanimous:
            round_entry["plurality"] = plurality

        if unanimous and gemini_confirm_fn is not None:
            try:
                g_val, g_in_tok, g_out_tok = gemini_confirm_fn()
            except Exception as exc:
                logger.warning(f"[Voting] Gemini validator call failed: {exc}")
                g_val, g_in_tok, g_out_tok = "", 0, 0
            gemini_in += g_in_tok
            gemini_out += g_out_tok
            matched = g_val == plurality
            round_entry["gemini_validator"] = {
                "called": True,
                "response": g_val,
                "matched": matched,
            }
            if matched:
                logger.info(
                    f"[Voting] Gemini validator confirmed unanimous round {round_num}: {g_val!r}"
                )
            else:
                logger.info(
                    f"[Voting] Gemini validator rejected unanimous round {round_num}: "
                    f"ollama={plurality!r}  gemini={g_val!r}"
                )
                unanimous = False
                round_entry["unanimous"] = False
                round_entry["plurality"] = plurality

        rounds_log.append(round_entry)

        if unanimous:
            final_value = plurality
            break

    # If consensus was never reached, use accumulated plurality across all rounds
    if not rounds_log[-1]["unanimous"]:
        final_value = Counter(all_normalized_values).most_common(1)[0][0]

    return final_value, rounds_log, total_in, total_out, gemini_in, gemini_out


def vote_for_fields(
    fn: Callable,
    fields_to_vote: dict,
    n: int = None,
    max_rounds: int = None,
    gemini_confirm_fn: Optional[Callable] = None,
) -> tuple:
    """
    Run fn() n times in parallel each round. fn() returns (fields_dict, in_tok, out_tok).

    fields_to_vote maps field_name → normalizer callable:
      - Date fields:       normalizer = _try_parse_date  (→ YYYY-MM-DD or None)
      - Categorical fields (cert_type): normalizer = _normalize_cert_type

    Fields NOT in fields_to_vote are non-voted; taken from the first successful parse.
    All n agents must agree on ALL voted fields simultaneously for consensus.
    After max_rounds without unanimity, take plurality across all rounds per field.

    gemini_confirm_fn: optional callable() → (fields_dict_with_normalised_voted, in_tok, out_tok).
    When provided, a unanimous Ollama round also requires Gemini confirmation (all voted
    fields match) before accepting. If any field disagrees, the round is treated as
    non-unanimous and the loop continues.

    Returns (final_fields, rounds_log, total_input_tokens, total_output_tokens,
             gemini_input_tokens, gemini_output_tokens).
    """
    if n is None:
        n = config.VOTING_N
    if max_rounds is None:
        max_rounds = config.VOTING_MAX_ROUNDS

    voted_field_names = list(fields_to_vote.keys())
    total_in, total_out = 0, 0
    gemini_in, gemini_out = 0, 0
    rounds_log = []
    non_voted_fields: Optional[dict] = None
    final_voted: dict = {f: None for f in voted_field_names}
    accumulated_field_values: dict = {f: [] for f in voted_field_names}

    for round_num in range(1, max_rounds + 1):
        logger.info(f"[Voting] ── Round {round_num}/{max_rounds} ──")
        results = _run_parallel(fn, n)
        round_responses = []
        per_field_values: dict = {f: [] for f in voted_field_names}

        for r in results:
            fields, in_tok, out_tok = ({}, 0, 0) if r is None else r
            total_in += in_tok
            total_out += out_tok

            # Capture first valid non-voted fields (e.g., details)
            if non_voted_fields is None and fields:
                non_voted_fields = {k: v for k, v in fields.items() if k not in voted_field_names}

            normalized = {}
            for f, normalizer in fields_to_vote.items():
                raw_val = fields.get(f)
                if raw_val and isinstance(raw_val, str):
                    norm = normalizer(raw_val)
                else:
                    norm = raw_val  # None passes through
                normalized[f] = norm
                per_field_values[f].append(norm)

            round_responses.append({"fields": fields, "normalized": normalized})

        # Accumulate values across rounds for final plurality calculation
        for f in voted_field_names:
            accumulated_field_values[f].extend(per_field_values[f])

        # Unanimity: all voted fields must agree simultaneously
        unanimous = all(
            len(set(str(v) for v in per_field_values[f])) == 1
            for f in voted_field_names
        )

        plurality_voted = {}
        for f in voted_field_names:
            top = Counter(str(v) for v in per_field_values[f]).most_common(1)[0][0]
            plurality_voted[f] = None if top == "None" else top

        round_entry = {"round": round_num, "responses": round_responses, "unanimous": unanimous}
        if not unanimous:
            round_entry["plurality"] = plurality_voted

        if unanimous and gemini_confirm_fn is not None and voted_field_names:
            try:
                g_fields, g_in_tok, g_out_tok = gemini_confirm_fn()
            except Exception as exc:
                logger.warning(f"[Voting] Gemini validator call failed: {exc}")
                g_fields, g_in_tok, g_out_tok = {}, 0, 0
            gemini_in += g_in_tok
            gemini_out += g_out_tok
            # Match: all voted fields must agree between Ollama consensus and Gemini
            matched = all(
                str(g_fields.get(f)) == str(per_field_values[f][0])
                for f in voted_field_names
            )
            round_entry["gemini_validator"] = {
                "called": True,
                "response": {f: g_fields.get(f) for f in voted_field_names},
                "matched": matched,
            }
            if matched:
                logger.info(
                    f"[Voting] Gemini validator confirmed unanimous round {round_num}: "
                    f"{{{', '.join(f + '=' + repr(per_field_values[f][0]) for f in voted_field_names)}}}"
                )
            else:
                ollama_vals = {f: per_field_values[f][0] for f in voted_field_names}
                gemini_vals = {f: g_fields.get(f) for f in voted_field_names}
                logger.info(
                    f"[Voting] Gemini validator rejected unanimous round {round_num}: "
                    f"ollama={ollama_vals}  gemini={gemini_vals}"
                )
                unanimous = False
                round_entry["unanimous"] = False
                round_entry["plurality"] = plurality_voted

        rounds_log.append(round_entry)

        if unanimous:
            # Use actual (non-stringified) normalized values from this round
            final_voted = {f: per_field_values[f][0] for f in voted_field_names}
            break

    # If consensus was never reached, use accumulated plurality across all rounds
    if not rounds_log[-1]["unanimous"]:
        for f in voted_field_names:
            top = Counter(str(v) for v in accumulated_field_values[f]).most_common(1)[0][0]
            final_voted[f] = None if top == "None" else top

    final_fields = {**(non_voted_fields or {}), **final_voted}
    return final_fields, rounds_log, total_in, total_out, gemini_in, gemini_out


