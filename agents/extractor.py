import json
import logging
import re
from typing import List, Optional

import config
from tools.date_utils import _try_parse_date
from tools.voting import vote_for_fields

from tools import hf_inference, ollama_inference, gemini_inference

_QUERY_FN_BY_BACKEND = {
    "huggingface": hf_inference.query_vlm,
    "gemini":      gemini_inference.query_vlm,
    "ollama":      ollama_inference.query_vlm,
}

logger = logging.getLogger(__name__)

# Matches floor-notation tower labels: G+XXXIII, B+G+XXXII, B+G+XXXIV, etc.
# Handles spaces around '+' and any combination of B/G prefixes + Roman numerals.
_FLOOR_NOTATION_RE = re.compile(
    r'\b(?:B\s*\+\s*)?G\s*\+\s*[IVXLCDM]+\b',
    re.IGNORECASE,
)
# Standard Roman numerals never repeat the same character 4+ times consecutively.
_INVALID_ROMAN_RE = re.compile(r'([IVXLCDM])\1{3,}', re.IGNORECASE)


def _is_valid_floor_notation(notation: str) -> bool:
    """Return False if the Roman numeral part contains 4+ consecutive identical chars."""
    return not _INVALID_ROMAN_RE.search(notation)


def _expand_tower_details(details: list) -> list:
    """
    If any detail item contains multiple floor-notation patterns grouped together
    (e.g. "Tower G+XXXIII" alone when the doc had three towers, or a grouped string),
    expand them into individual "Tower X" entries.

    Scans every string in details for floor-notation sub-patterns and replaces
    grouped/partial items with fully split individual entries.
    """
    if not isinstance(details, list):
        return details

    seen: list = []
    for item in details:
        if not isinstance(item, str):
            seen.append(item)
            continue
        notations = _FLOOR_NOTATION_RE.findall(item)
        if notations:
            for n in notations:
                if not _is_valid_floor_notation(n):
                    logger.warning(f"[Extractor] Dropping invalid floor notation: {n!r}")
                    continue
                label = "Tower " + re.sub(r'\s+', '', n).upper()
                if label not in seen:
                    seen.append(label)
        else:
            if item not in seen:
                seen.append(item)
    return seen


BACKSTORY = (
    "You are a precise data extraction specialist for real estate legal documents. "
    "You extract only what is explicitly present in the document."
)

_FIELDS_BY_TYPE = {
    "registration certificate": """\
Extract the following fields and return ONLY a flat JSON object with these exact keys:
  - registration_date
  - promised_completion_date
      * STRICT RULE: Look for phrases like "ending with", "ending on", "valid till", "validity of registration", or "completion date".
      * The END date of the registration validity period IS the promised completion date — NOT the start/commencement date.
      * Example: "valid for a period of X years commencing from DD/MM/YYYY and ending with DD/MM/YYYY" — use the SECOND (ending) date only.
      * WARNING: Do NOT confuse the commencement/start date with the ending date. The promised_completion_date is always the LATER date in the validity period.

Extraction Rules:
- Date Format: Extract dates EXACTLY as they appear in the document (e.g., "29/11/2022" or "29-11-2022"). Do NOT reformat or convert.
- Language: Handle multi-lingual text and ALWAYS normalize/translate values to English.
- Nulls: If a value is missing, use null.
- Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example output:
{{"registration_date": "29/11/2022", "promised_completion_date": "30/06/2025"}}""",

    "extension certificate": """\
Extract the following fields and return ONLY a flat JSON object with these exact keys:
  - extension_approval_date
      * STRICT RULE: Look for "Date of Approval" or "Dated:".
      * WARNING: DO NOT confuse the Extension Serial Number (e.g., "Extension No. 07") or revision number (e.g., "2020/07") with the month/date.
  - revised_promised_completion_date
      * STRICT RULE: Look for phrases like "ending on", "ending with", "valid till", "extended till", or "Validity of Registration".
      * The end date of the extended validity period IS the revised_promised_completion_date.
      * Example: "extended by X months ending on DD/MM/YYYY" or "validity...ending on DD/MM/YYYY" — that date is the revised_promised_completion_date.

Extraction Rules:
- Date Format: Extract dates EXACTLY as they appear in the document (e.g., "15/12/2022" or "15-12-2022"). Do NOT reformat or convert.
- Language: Handle multi-lingual text and ALWAYS normalize/translate values to English.
- Nulls: If a value is missing, use null.
- Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example output:
{{"extension_approval_date": "10/04/2023", "revised_promised_completion_date": "31/12/2026"}}""",

    # Date-only prompts for OC/CC voting (no details — keeps JSON simple and prevents truncation)
    "occupancy certificate": """\
Extract ONLY the following field and return a flat JSON object with this exact key:
  - oc_cc_issue_date: the date THIS certificate/document was issued.
      * STRICT RULE: Extract the date from the CERTIFICATE HEADER — the date printed at the
        top of the letter next to the memo number, reference number, or a label such as
        "Date:", "Dated:", "Dt:", "Dt.", "Date of Issue:", "OC/CC DATE", "CC/OC DATE",
        "तारीख:" / "दिनांक:" (Marathi/Hindi), "তারিখ:" (Bengali), "તારીખ:" (Gujarati).
      * For Form 4 / Architect's Certificates of Completion: use the date of the Form 4
        letter itself (shown at the top of the document), NOT the dates of OC numbers
        listed in any reference table within the document.
      * If the PDF contains multiple bundled documents (e.g. a cover letter or corrigendum
        on page 1 followed by the actual certificate on a later page), extract the date
        from the page that is the actual certificate.
      * Do NOT extract: dates from sanction orders, building permission references, prior
        approval dates, OC/CC reference tables, or any date belonging to a cited document
        rather than this certificate itself.
      * Multilingual numerals: Gujarati/Devanagari digits should be read as standard Arabic
        numerals (e.g. "૧૩/૦૩/૨૦૨૩" = "13/03/2023").

Extraction Rules:
- Date Format: Extract the date EXACTLY as it appears in the document (e.g., "05/04/2024" or "05-04-2024"). Do NOT reformat or convert.
- Nulls: If the date is missing, use null.
- Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example output:
{{"oc_cc_issue_date": "05/04/2024"}}""",

    "completion certificate": """\
Extract ONLY the following field and return a flat JSON object with this exact key:
  - oc_cc_issue_date: the date THIS certificate/document was issued.
      * STRICT RULE: Extract the date from the CERTIFICATE HEADER — the date printed at the
        top of the letter next to the memo number, reference number, or a label such as
        "Date:", "Dated:", "Dt:", "Dt.", "Date of Issue:", "OC/CC DATE", "CC/OC DATE",
        "तारीख:" / "दिनांक:" (Marathi/Hindi), "তারিখ:" (Bengali), "તારીખ:" (Gujarati).
      * For Form 4 / Architect's Certificates of Completion: use the date of the Form 4
        letter itself (shown at the top of the document), NOT the dates of OC numbers
        listed in any reference table within the document.
      * If the PDF contains multiple bundled documents (e.g. a cover letter or corrigendum
        on page 1 followed by the actual certificate on a later page), extract the date
        from the page that is the actual certificate.
      * Do NOT extract: dates from sanction orders, building permission references, prior
        approval dates, OC/CC reference tables, or any date belonging to a cited document
        rather than this certificate itself.
      * Multilingual numerals: Gujarati/Devanagari digits should be read as standard Arabic
        numerals (e.g. "૧૩/૦૩/૨૦૨૩" = "13/03/2023").

Extraction Rules:
- Date Format: Extract the date EXACTLY as it appears in the document (e.g., "15/05/2024" or "15-05-2024"). Do NOT reformat or convert.
- Nulls: If the date is missing, use null.
- Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example output:
{{"oc_cc_issue_date": "15/05/2024"}}""",
}

# Separate full prompts used for the single details-extraction call for OC/CC
_DETAILS_FIELDS_BY_TYPE = {
    "occupancy certificate": """\
Extract ONLY the following field and return a flat JSON object with this exact key:
  - details: a string describing the building scope covered by this certificate.

      FOLLOW THESE PRIORITIES IN ORDER. Stop at the first priority that yields a result:

      * PRIORITY 1 — Building Particulars section: Look for a labeled inline line or section
        starting with "Building Particulars", "Building Parameters", "Particulars of Building",
        "Description of Building", or similar. Extract the FULL TEXT of that line/section
        verbatim as a single string. Include ALL identifiers: plot numbers, RS/LR numbers,
        premises numbers, holding numbers, mouza names, ward numbers, sanction/regularization
        references, rule numbers, approval numbers, and dates exactly as written.
        Example 1: "Regularized U/R 26 (2a) & (2b) of KMC Building Rule 2009, Vide S.l no. 056/BLDG/I/23-24, dt. 18/10/2022"
        Example 2: "Plot- RS-184 : 969,360, Premises Number - 199, Holding No - 199, Mouza - MAHINAGAR, Ward-22"

      * PRIORITY 2 — Named building components: If no Building Particulars section exists,
        extract block, tower, or wing labels from the body text or tables.
        - Explicitly certified in body text (e.g., "Block 11", "Block 12", "Tower A")
        - Named buildings in a "Building Details" table (e.g., "HIG A1", "MIG 1", "LIG")
        - Plot/RS/LR identifiers (e.g., "RS-452") — do NOT include sub-plot numbers
        - Floor-notation labels: prefix with "Tower" (e.g., "Tower G+XXXIII", "Tower B+G+XXXII")
        Return as a comma-separated string. Split grouped notations into separate items.

      * If NEITHER a Building Particulars section NOR named components exist, return "".

      DO NOT extract:
      * The main certification/body paragraph (text beginning with "I hereby certify",
        "With reference to your notice", "On the basis of the same", or similar letter prose)
      * Signature blocks, sender/recipient addresses, or subject lines
      * TRANSLATE all text to English.

Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example outputs:
{{"details": "Regularized U/R 26 (2a) & (2b) of KMC Building Rule 2009, Vide S.l no. 056/BLDG/I/23-24, dt. 18/10/2022"}}
{{"details": "Block 11, Block 12, Tower A"}}
{{"details": "Plot- RS-184 : 969,360, Premises Number - 199, Holding No - 199, Mouza - MAHINAGAR, Ward-22"}}""",

    "completion certificate": """\
Extract ONLY the following field and return a flat JSON object with this exact key:
  - details: a string describing the building scope covered by this certificate.

      FOLLOW THESE PRIORITIES IN ORDER. Stop at the first priority that yields a result:

      * PRIORITY 1 — Building Particulars section: Look for a labeled inline line or section
        starting with "Building Particulars", "Building Parameters", "Particulars of Building",
        "Description of Building", or similar. Extract the FULL TEXT of that line/section
        verbatim as a single string. Include ALL identifiers: plot numbers, RS/LR numbers,
        premises numbers, holding numbers, mouza names, ward numbers, sanction/regularization
        references, rule numbers, approval numbers, and dates exactly as written.
        Example 1: "Regularized U/R 26 (2a) & (2b) of KMC Building Rule 2009, Vide S.l no. 056/BLDG/I/23-24, dt. 18/10/2022"
        Example 2: "Plot- RS-184 : 969,360, Premises Number - 199, Holding No - 199, Mouza - MAHINAGAR, Ward-22"

      * PRIORITY 2 — Named building components: If no Building Particulars section exists,
        extract block, tower, or wing labels from the body text or tables.
        - Explicitly certified in body text (e.g., "Block 11", "Block 12", "Tower A")
        - Named buildings in a "Building Details" table (e.g., "HIG A1", "MIG 1", "LIG")
        - Plot/RS/LR identifiers (e.g., "RS-452") — do NOT include sub-plot numbers
        - Floor-notation labels: prefix with "Tower" (e.g., "Tower G+XXXIII", "Tower B+G+XXXII")
        Return as a comma-separated string. Split grouped notations into separate items.

      * If NEITHER a Building Particulars section NOR named components exist, return "".

      DO NOT extract:
      * The main certification/body paragraph (text beginning with "I hereby certify",
        "With reference to your notice", "On the basis of the same", or similar letter prose)
      * Signature blocks, sender/recipient addresses, or subject lines
      * TRANSLATE all text to English.

Format: Return ONLY raw JSON. No markdown backticks, no explanatory text.

Example outputs:
{{"details": "Regularized U/R 26 (2a) & (2b) of KMC Building Rule 2009, Vide S.l no. 056/BLDG/I/23-24, dt. 18/10/2022"}}
{{"details": "Block 11, Block 12, Tower A"}}
{{"details": "Plot- RS-184 : 969,360, Premises Number - 199, Holding No - 199, Mouza - MAHINAGAR, Ward-22"}}""",
}

_PROMPT_TEMPLATE = """{backstory}

Document type: {document_type}

{fields_instructions}
"""

_OC_CC_TYPES = {"occupancy certificate", "completion certificate"}

# Voted fields per document type: field_name → normalizer callable
# Fields NOT listed here (e.g., details) are taken from the first successful parse.
_VOTED_FIELDS_BY_TYPE: dict = {
    "registration certificate": {
        "registration_date":        _try_parse_date,
        "promised_completion_date": _try_parse_date,
    },
    "extension certificate": {
        "extension_approval_date":           _try_parse_date,
        "revised_promised_completion_date":  _try_parse_date,
    },
    # OC/CC: only vote on the date — cert_type is derived, details extracted separately
    "occupancy certificate": {
        "oc_cc_issue_date": _try_parse_date,
    },
    "completion certificate": {
        "oc_cc_issue_date": _try_parse_date,
    },
}


def _parse_json_response(raw: str, context: str = "[Extractor]") -> dict:
    """Parse a JSON response, handling double-brace {{...}} variants."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except Exception:
        # Retry: strip outer double-braces ({{...}} → {...})
        if clean.startswith("{{") and clean.endswith("}}"):
            clean = clean[1:-1]
        try:
            parsed = json.loads(clean)
        except Exception:
            logger.warning(f"{context} Could not parse JSON: {raw[:200]}")
            return {}
    if not isinstance(parsed, dict):
        logger.warning(f"{context} Expected JSON object, got {type(parsed).__name__}: {raw[:200]}")
        return {}
    return parsed


def extract_once(
    images: List,
    doc_type: str,
    hf_token: Optional[str],
    query_fn=None,
    model_override: Optional[str] = None,
) -> tuple:
    """Single VLM extraction call. Returns (fields_dict, input_tokens, output_tokens).

    query_fn and model_override are used by the hybrid tiebreaker to route the call
    through Gemini while reusing the same prompt logic.
    """
    if query_fn is None:
        _cat = "oc_cc" if doc_type in _OC_CC_TYPES else "rc_ec"
        query_fn = _QUERY_FN_BY_BACKEND[config.get_backend(_cat)]

    fields_instructions = _FIELDS_BY_TYPE.get(doc_type, "")
    if not fields_instructions:
        logger.warning(f"[Extractor] Unknown document_type={doc_type!r}, no field instructions")

    prompt = _PROMPT_TEMPLATE.format(
        backstory=BACKSTORY,
        document_type=doc_type,
        fields_instructions=fields_instructions,
    )

    if model_override:
        model = model_override
    else:
        _cat = "oc_cc" if doc_type in _OC_CC_TYPES else "rc_ec"
        model = (
            config.GEMINI_EXTRACTOR_MODEL
            if config.get_backend(_cat) == "gemini"
            else config.EXTRACTOR_MODEL
        )
    raw, input_tokens, output_tokens = query_fn(
        prompt=prompt,
        images=images,
        model=model,
        token=hf_token,
    )

    fields = _parse_json_response(raw)

    if "details" in fields and isinstance(fields.get("details"), list):
        fields["details"] = _expand_tower_details(fields["details"])

    logger.debug(f"[Extractor] extracted keys={list(fields.keys())}")
    return fields, input_tokens, output_tokens


def _extract_details_once(
    images: List,
    doc_type: str,
    hf_token: Optional[str],
    query_fn=None,
    model_override: Optional[str] = None,
) -> tuple:
    """Single VLM call to extract only the details field for OC/CC. Returns (fields_dict, in_tok, out_tok)."""
    fields_instructions = _DETAILS_FIELDS_BY_TYPE.get(doc_type, "")
    if not fields_instructions:
        return {}, 0, 0

    prompt = _PROMPT_TEMPLATE.format(
        backstory=BACKSTORY,
        document_type=doc_type,
        fields_instructions=fields_instructions,
    )

    if query_fn is None:
        query_fn = _QUERY_FN_BY_BACKEND[config.get_backend("oc_cc")]
    model = model_override or (
        config.GEMINI_EXTRACTOR_MODEL
        if config.get_backend("oc_cc") == "gemini"
        else config.EXTRACTOR_MODEL
    )
    raw, input_tokens, output_tokens = query_fn(
        prompt=prompt,
        images=images,
        model=model,
        token=hf_token,
    )

    fields = _parse_json_response(raw, context="[Extractor/details]")

    if "details" in fields and isinstance(fields.get("details"), list):
        fields["details"] = _expand_tower_details(fields["details"])

    return fields, input_tokens, output_tokens


def extract_with_vote(state: dict) -> dict:
    """Run N parallel extractors and vote for consensus on key fields. Updates WorkflowState."""
    doc_type = state.get("document_type", "").strip().lower()
    fields_to_vote = _VOTED_FIELDS_BY_TYPE.get(doc_type, {})
    voted_field_names = list(fields_to_vote.keys())
    images = state["current_images"]
    hf_token = state.get("hf_token")

    category = state.get("document_category", "rc_ec")
    backend  = config.get_backend(category)
    query_fn = _QUERY_FN_BY_BACKEND[backend]
    model    = config.GEMINI_EXTRACTOR_MODEL if backend == "gemini" else config.EXTRACTOR_MODEL

    # Force-load all pixel data into memory before spawning threads.
    # PIL images lazily read from BytesIO buffers; concurrent threads race on the read pointer.
    for img in images:
        img.load()
    # Each worker gets its own copy so threads don't share mutable PIL state.
    fn = lambda: extract_once(
        [img.copy() for img in images], doc_type, hf_token,
        query_fn=query_fn, model_override=model,
    )

    # Gemini confirm function — reused by both the round validator and the tiebreaker.
    gemini_confirm_fn = None
    if config.GEMINI_ROUND_VALIDATOR and backend != "gemini":
        def gemini_confirm_fn():
            fields, g_in, g_out = extract_once(
                [img.copy() for img in images], doc_type, hf_token,
                query_fn=gemini_inference.query_vlm,
                model_override=config.GEMINI_EXTRACTOR_MODEL,
            )
            normalised = {
                f: normaliser(fields.get(f))
                if fields.get(f) and isinstance(fields.get(f), str) else fields.get(f)
                for f, normaliser in fields_to_vote.items()
            }
            return {**fields, **normalised}, g_in, g_out
        logger.info("[Extractor] Gemini round validator enabled")
    else:
        logger.debug(
            f"[Extractor] Gemini round validator skipped "
            f"(GEMINI_ROUND_VALIDATOR={config.GEMINI_ROUND_VALIDATOR}, "
            f"backend={backend!r})"
        )

    final_fields, rounds_log, in_tok, out_tok, g_val_in, g_val_out = vote_for_fields(
        fn=fn,
        fields_to_vote=fields_to_vote,
        gemini_confirm_fn=gemini_confirm_fn,
    )

    logger.info(
        f"[Extractor] doc_type={doc_type!r}, "
        f"rounds={len(rounds_log)}, "
        f"unanimous={rounds_log[-1]['unanimous']}, "
        f"voted_fields={voted_field_names}"
    )

    audit_entry = {
        "stage": "extract",
        "rounds_taken": len(rounds_log),
        "consensus_reached": rounds_log[-1]["unanimous"],
        "voted_fields": voted_field_names,
        "final_voted": {f: final_fields.get(f) for f in fields_to_vote},
        "rounds": rounds_log,
    }

    gemini_in = state.get("gemini_input_tokens", 0) + g_val_in
    gemini_out = state.get("gemini_output_tokens", 0) + g_val_out
    low_confidence = False

    # ── Hybrid tiebreaker ────────────────────────────────────────────────────
    if config.INFERENCE_MODE == "hybrid" and backend != "gemini" and not rounds_log[-1]["unanimous"]:
        from tools.tiebreaker import tiebreak_fields

        if gemini_confirm_fn is None:
            def gemini_confirm_fn():
                fields, g_in, g_out = extract_once(
                    [img.copy() for img in images],
                    doc_type, hf_token,
                    query_fn=gemini_inference.query_vlm,
                    model_override=config.GEMINI_EXTRACTOR_MODEL,
                )
                normalised = {
                    f: normaliser(fields.get(f))
                    if fields.get(f) and isinstance(fields.get(f), str) else fields.get(f)
                    for f, normaliser in fields_to_vote.items()
                }
                return {**fields, **normalised}, g_in, g_out

        # majority_fields: the per-field plurality values already computed by vote_for_fields
        majority_fields = {f: final_fields.get(f) for f in voted_field_names}

        tb_fields, tb_log, g_in, g_out = tiebreak_fields(
            majority_fields=majority_fields,
            gemini_fn=gemini_confirm_fn,
            voted_field_names=voted_field_names,
            n_calls=config.GEMINI_TIEBREAKER_CALLS,
        )
        gemini_in += g_in
        gemini_out += g_out
        audit_entry["tiebreaker"] = tb_log

        if tb_fields is None:
            # Tiebreaker could not reach confidence — null out all fields
            final_fields = {}
            low_confidence = True
            logger.warning(f"[Extractor] low_confidence — tiebreaker exhausted for doc_type={doc_type!r}")
        else:
            final_fields = tb_fields
            logger.info(f"[Extractor] tiebreaker outcome={tb_log['outcome']!r}")

    # ── OC/CC post-processing (only when not low_confidence) ─────────────────
    if not low_confidence:
        # For OC/CC: use the classifier output exactly as the cert_type
        if doc_type in _OC_CC_TYPES:
            final_fields["cert_type"] = state.get("document_type", "")

        # For OC/CC: single VLM call for details after date consensus is reached
        if doc_type in _DETAILS_FIELDS_BY_TYPE:
            details_fields, det_in, det_out = _extract_details_once(
                [img.copy() for img in images], doc_type, hf_token,
                query_fn=query_fn, model_override=model,
            )
            final_fields["details"] = details_fields.get("details", "")
            in_tok += det_in
            out_tok += det_out

    audit_entry["final_voted"] = {f: final_fields.get(f) for f in fields_to_vote}

    return {
        "extracted_fields": final_fields,
        "extraction_rounds": len(rounds_log),
        "extraction_unanimous": rounds_log[-1]["unanimous"],
        "input_tokens": state.get("input_tokens", 0) + in_tok,
        "output_tokens": state.get("output_tokens", 0) + out_tok,
        "gemini_input_tokens": gemini_in,
        "gemini_output_tokens": gemini_out,
        "low_confidence": low_confidence,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
