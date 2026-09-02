import json
import logging
from typing import List, Optional

import config
from tools.voting import vote_for_string

from tools import hf_inference, ollama_inference, gemini_inference

_QUERY_FN_BY_BACKEND = {
    "huggingface": hf_inference.query_vlm,
    "gemini":      gemini_inference.query_vlm,
    "ollama":      ollama_inference.query_vlm,
}

logger = logging.getLogger(__name__)

BACKSTORY = (
    "You are a document analyst specialising in Indian RERA regulatory filings. "
    "Your only job is to identify what type of document you are looking at."
)

_RC_EC_PROMPT = """{backstory}

Look at the document image(s) and return ONLY a valid JSON object with a single key "document_type".
The value MUST be exactly one of:
  "Registration Certificate"  — issued SPECIFICALLY by a state RERA authority (e.g. MahaRERA, WBRERA, HRERA), grants a RERA project registration number (format like P51700XXXXXXX), and lists real-estate project details such as promoter name, project address, and completion conditions.
  "Extension Certificate"     — issued by a RERA authority, extends the validity/completion date of an existing RERA project registration.
  "Unknown"                   — use this for ANYTHING else.

IMPORTANT — the following must be classified as "Unknown", NOT as "Registration Certificate":
- GST Registration Certificate (Form GST REG-06) — these are issued by the Government of India / tax authority, show a GSTIN number, and list business registration details.
- Company/LLP/firm incorporation or registration certificates.
- Shop-and-establishment registration, MSME, or any other government registration unrelated to RERA.
- Sanction plans, NOCs, court orders, architect certificates, bank documents, or any document that is not a RERA certificate.

A true RERA Registration Certificate will prominently display the name of the RERA authority and a RERA project registration number.

IMPORTANT DISTINCTION — Registration Certificate vs Extension Certificate:
- A Registration Certificate ALWAYS contains a validity period clause (e.g. "valid till DD-MM-YYYY" or "registration shall be valid for X years ending DD-MM-YYYY"). This is standard language on every new registration — it does NOT make the document an Extension Certificate.
- An Extension Certificate is a SEPARATE document issued after the original registration has expired or is about to expire. It will explicitly reference an earlier registration and use language like "extended by X months", "revised completion date", or "extension of registration".
- If the document is titled "Registration Certificate", "Form C", or similar — it is a Registration Certificate, regardless of any validity period mentioned.

Return ONLY raw JSON. No markdown, no explanation.
Example: {{"document_type": "Registration Certificate"}}
"""

_OC_CC_PROMPT = """{backstory}

Look at the document image(s) and return ONLY a valid JSON object with a single key "document_type".
The value MUST be exactly one of:
  "Occupancy Certificate"   — a certificate granted BY a government authority TO the developer/owner
    declaring a specific building/flat/wing is fit for OCCUPATION. Valid issuers include municipal
    corporations, panchayats, town municipalities, zilla parishads, development authorities (e.g.
    NOIDA, HUDA, AUDA), and town & country planning departments. Also includes: Building Use
    Certificate (BUC), Building Use Permission (BU Permission / B.U. Permission) issued in Gujarat;
    Occupation Certificate (Form BR-VII) issued in Haryana; and Form 4 / Architect's Certificate of
    Completion that explicitly certifies 100% construction is complete AND lists actual OC numbers
    already granted by a municipal body.
  "Completion Certificate"  — a certificate granted BY a government authority TO the developer/owner
    declaring that construction of a building/wing is COMPLETE and conforms to approved plans. Valid
    issuers are the same as above.
  "Unknown"                 — use this for ANYTHING else.

IMPORTANT — the following must always be classified as "Unknown":
- Annual reports, balance sheets, profit & loss statements, financial statements, or any accounting/audit document — even if they contain an "Auditor's Certificate" or "Independent Auditor's Report".
- GST registration certificates, tax documents, or any certificate issued by a tax authority.
- Company/LLP/firm incorporation certificates, partnership deeds, or business registration documents.
- Booking application forms, allotment letters, proforma/annexure documents, sale agreements, sale deeds, or any document that a buyer/allottee fills out or signs.
- Documents listing payment schedules, instalment plans, terms and conditions for property purchase, or general/special conditions for buyers.
- Architectural/structural drawings, sanction plans, building plan approvals, blueprints, NOCs, layout plans, or site plans.
- Court orders, legal notices, affidavits, or any judicial/quasi-judicial document.
- Bank documents, loan sanction letters, or any document issued by a financial institution.
- Chartered Accountant's Certificate, CA Certificate, Form 3 (RERA), or any financial progress / fund-withdrawal certificate issued by a CA firm — these are RERA compliance documents submitted to justify withdrawals from the designated account and are NOT issued by a municipal body.
- Engineer's Certificate (Form 2, RERA) or any certificate by a licensed engineer certifying percentage of construction cost incurred or balance cost to complete — these are fund-withdrawal compliance documents, NOT OC/CC certificates.
- Architect's percentage-of-completion certificate (Annexure-A or similar) certifying that X% of construction work is done for the purpose of demanding instalments from allottees — NOT a Form 4 completion certificate.
- Request letters or applications FROM a developer/builder TO a municipal authority asking for issuance of an OC/CC — the document must be issued BY the authority, not sent TO it.
- Society handover documents, transfer deeds, or any document transferring project management to a housing society or co-operative.
- Income Tax Return acknowledgements (ITR), PAN cards, Aadhaar cards, or any identity or tax document.
- Drain/sewer connection permits, water connection permits, or utility connection clearances — these are infrastructure permits, not OC/CC.
- Commencement notices — letters informing a municipal body that construction has started.
- Any certificate NOT issued by a government authority or licensed architect (Form 4 only).

STRICT IDENTIFICATION RULES — apply IN ORDER:

RULE 0 — OVERALL DOCUMENT PURPOSE (check before everything else):
Classify based on the PRIMARY PURPOSE of the ENTIRE document, not individual pages. Many PDFs bundle a main document with attached copies — a handover agreement that includes attached OC/CC copies is still a handover agreement. A CA certificate that references an OC number is still a CA certificate. Always ask: "What is THIS document, as a whole?" before looking at individual pages.

RULE 1 — POSITIVE EVIDENCE GATE (mandatory):
Before classifying anything as "Occupancy Certificate" or "Completion Certificate", you MUST be able to confirm ALL four of the following for the document AS A WHOLE. If even ONE is absent → return "Unknown".
  ✓ The document was ISSUED BY a government body (municipal corporation, development authority, panchayat, or town planning dept) — the issuing authority appears on the primary/first page, not just on an attached copy buried in later pages.
  ✓ It explicitly states the building/wing/project is FIT FOR OCCUPATION (OC) or that construction is COMPLETE and conforms to approved plans (CC).
  ✓ It is addressed TO a developer/builder/owner — NOT written FROM a developer/applicant or TO a housing society.
  ✓ It is signed or stamped by an authorised government official (Commissioner, Executive Engineer, Municipal Officer, etc.) as the primary signatory.

RULE 2 — REJECTION FILTERS (return "Unknown" immediately if any applies):
(a) SOCIETY / HANDOVER: The document is a handover, transfer of possession, or conveyance TO a housing society, co-operative, or residents' association — regardless of any OC/CC documents embedded or attached within it.
(b) APPLICATION / REQUEST: The document is FROM a developer/applicant directed TO a government authority:
  • Starts with "To, The Commissioner", "To, The Municipal Corporation/Authority/Council/Officer"
  • Contains "I/We hereby apply", "I/We request", "kindly issue", "please grant"
  • Heading says "APPLICATION FOR OCCUPANCY/COMPLETION CERTIFICATE" or similar
  • Contains empty fields or blank spaces for an applicant to fill in

RULE 3: "Building Use Certificate", "Building Use Permission", "B.U. Permission" are valid OC equivalents issued in Gujarat — treat them as "Occupancy Certificate".

RULE 4: A Form 4 / Architect's Certificate that certifies only a PERCENTAGE (e.g. "75% complete") is NOT a valid OC/CC — it is "Unknown". A valid Form 4 must state 100% completion AND list OC numbers already granted.

RULE 5: If the document contains financial figures, cost tables, or percentage-of-work-done tables — it is "Unknown".

RULE 6: If the document is on a CA firm's letterhead (e.g. "Chartered Accountants", "LLP", "LLPIN") — it is "Unknown", even if it mentions "Occupancy Certificate" or "Completion Certificate".

RULE 7: When in doubt, always return "Unknown".

Return ONLY raw JSON. No markdown, no explanation.
Example: {{"document_type": "Occupancy Certificate"}}
"""

_PROMPT_BY_CATEGORY = {
    "rc_ec": _RC_EC_PROMPT,
    "oc_cc": _OC_CC_PROMPT,
}

# Canonical document type values — any VLM output is normalized to one of these or "Unknown".
_CANONICAL_TYPES = {
    "registration certificate": "Registration Certificate",
    "extension certificate": "Extension Certificate",
    "occupancy certificate": "Occupancy Certificate",
    "occupation certificate": "Occupancy Certificate",
    "occupancy": "Occupancy Certificate",
    "occupation": "Occupancy Certificate",
    "oc": "Occupancy Certificate",
    # Gujarat equivalents — Building Use Certificate / Building Use Permission
    "building use certificate": "Occupancy Certificate",
    "building use permission":  "Occupancy Certificate",
    "bu permission":            "Occupancy Certificate",
    "b.u. permission":          "Occupancy Certificate",
    "completion certificate": "Completion Certificate",
    "completion": "Completion Certificate",
    "cc": "Completion Certificate",
}


# Valid output types per category — used to clamp out-of-bounds model hallucinations.
# If a model using the oc_cc prompt returns "Registration Certificate", it is clamped to Unknown.
_VALID_BY_CATEGORY: dict = {
    "oc_cc": {"Occupancy Certificate", "Completion Certificate", "Unknown"},
    "rc_ec": {"Registration Certificate", "Extension Certificate", "Unknown"},
}


def _normalize_doc_type(raw_value: str) -> str:
    """Map a raw VLM document_type string to one of the canonical values."""
    if not raw_value:
        return "Unknown"
    return _CANONICAL_TYPES.get(raw_value.strip().lower(), "Unknown")


def _hint_from_name(name: str) -> str:
    """Extract a document type hint from a filename. Returns canonical type or ''."""
    lower = name.lower().replace("_", " ")
    if "completion certificate" in lower:
        hint = "Completion Certificate"
    elif "occupancy certificate" in lower:
        hint = "Occupancy Certificate"
    elif "registration certificate" in lower:
        hint = "Registration Certificate"
    elif "extension certificate" in lower:
        hint = "Extension Certificate"
    else:
        return ""
    # Suppress the hint if the filename also signals a non-OC/CC document type.
    # A wrong hint is worse than no hint — the model can read the image itself.
    if (
        "ca certificate" in lower or "ca cert" in lower
        or "gst" in lower
        or "itr" in lower
        or "engineer" in lower
        or "drain" in lower or "sewer" in lower
        or "affidavit" in lower
        or "application" in lower
        or "handover" in lower or "hand over" in lower
        or ("occupancy certificate" in lower and "completion certificate" in lower)
        or "balance sheet" in lower
        or " fy " in lower
        or "profit" in lower
        or "financial" in lower
    ):
        return ""
    return hint


_FILENAME_HINT_SUFFIX = """
Filename context (weak signal — trust the document image above all else):
  The file is named "{name}". This may indicate the type is "{hint}"."""


def classify_once(
    images: List,
    category: str,
    hf_token: Optional[str],
    filename_hint: str = "",
    filename: str = "",
    query_fn=None,
    model_override: Optional[str] = None,
) -> tuple:
    """Single VLM classification call. Returns (normalized_doc_type, input_tokens, output_tokens).

    query_fn and model_override are used by the hybrid tiebreaker to route the call
    through Gemini while reusing the same prompt logic.
    """
    if query_fn is None:
        query_fn = _QUERY_FN_BY_BACKEND[config.get_backend(category)]

    template = _PROMPT_BY_CATEGORY.get(category, _RC_EC_PROMPT)
    prompt = template.format(backstory=BACKSTORY)
    if filename_hint:
        prompt = prompt + _FILENAME_HINT_SUFFIX.format(name=filename, hint=filename_hint)

    if model_override:
        model = model_override
    else:
        model = (
            config.GEMINI_CLASSIFIER_MODEL
            if config.get_backend(category) == "gemini"
            else config.CLASSIFIER_MODEL
        )
    raw, input_tokens, output_tokens = query_fn(
        prompt=prompt,
        images=images,
        model=model,
        token=hf_token,
    )

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        # Handle double-brace responses: {{"key": "value"}} → {"key": "value"}
        if clean.startswith("{{") and clean.endswith("}}"):
            clean = clean[1:-1]
        result = json.loads(clean)
        raw_type = result.get("document_type", "")
    except Exception:
        logger.warning(f"[Classifier] Could not parse JSON: {raw[:200]}")
        raw_type = ""

    doc_type = _normalize_doc_type(raw_type)

    # Clamp: if the model returns a type outside the valid set for this category, treat as Unknown.
    # This prevents cross-category hallucinations (e.g. oc_cc prompt returning "Registration Certificate").
    valid_set = _VALID_BY_CATEGORY.get(category, set())
    if valid_set and doc_type not in valid_set:
        logger.warning(
            f"[Classifier] Out-of-category output clamped to Unknown: "
            f"category={category!r}, raw={raw_type!r}, normalized={doc_type!r}"
        )
        doc_type = "Unknown"

    logger.debug(f"[Classifier] category={category!r}, raw={raw_type!r}, normalized={doc_type!r}")
    return doc_type, input_tokens, output_tokens


def classify_with_vote(state: dict) -> dict:
    """Run N parallel classifiers and vote for consensus. Updates WorkflowState."""
    images = state["current_images"]
    category = state.get("document_category", "rc_ec")
    hf_token = state.get("hf_token")
    filename = state.get("current_name", "")
    hint = _hint_from_name(filename)

    backend  = config.get_backend(category)
    query_fn = _QUERY_FN_BY_BACKEND[backend]
    model    = config.GEMINI_CLASSIFIER_MODEL if backend == "gemini" else config.CLASSIFIER_MODEL

    # Force-load all pixel data into memory before spawning threads.
    # PIL images lazily read from BytesIO buffers; concurrent threads race on the read pointer.
    for img in images:
        img.load()
    # Each worker gets its own copy so threads don't share mutable PIL state.
    fn = lambda: classify_once(
        [img.copy() for img in images], category, hf_token, hint, filename,
        query_fn=query_fn, model_override=model,
    )

    gemini_confirm_fn = None
    if config.GEMINI_ROUND_VALIDATOR and backend != "gemini":
        def gemini_confirm_fn():
            return classify_once(
                [img.copy() for img in images], category, hf_token, hint, filename,
                query_fn=gemini_inference.query_vlm,
                model_override=config.GEMINI_CLASSIFIER_MODEL,
            )
        logger.info("[Classifier] Gemini round validator enabled")
    else:
        logger.debug(
            f"[Classifier] Gemini round validator skipped "
            f"(GEMINI_ROUND_VALIDATOR={config.GEMINI_ROUND_VALIDATOR}, "
            f"backend={backend!r})"
        )

    consensus, rounds_log, in_tok, out_tok, g_val_in, g_val_out = vote_for_string(
        fn=fn,
        normalize=_normalize_doc_type,
        gemini_confirm_fn=gemini_confirm_fn,
    )

    logger.info(
        f"[Classifier] consensus={consensus!r}, "
        f"rounds={len(rounds_log)}, "
        f"unanimous={rounds_log[-1]['unanimous']}"
    )

    audit_entry = {
        "stage": "classify",
        "rounds_taken": len(rounds_log),
        "consensus_reached": rounds_log[-1]["unanimous"],
        "final_value": consensus,
        "rounds": rounds_log,
    }

    gemini_in = state.get("gemini_input_tokens", 0) + g_val_in
    gemini_out = state.get("gemini_output_tokens", 0) + g_val_out
    low_confidence = False

    # ── Hybrid tiebreaker ────────────────────────────────────────────────────
    if config.INFERENCE_MODE == "hybrid" and backend != "gemini" and not rounds_log[-1]["unanimous"]:
        from tools.tiebreaker import tiebreak_string

        if gemini_confirm_fn is None:
            def gemini_confirm_fn():
                return classify_once(
                    [img.copy() for img in images],
                    category, hf_token, hint, filename,
                    query_fn=gemini_inference.query_vlm,
                    model_override=config.GEMINI_CLASSIFIER_MODEL,
                )

        tb_value, tb_log, g_in, g_out = tiebreak_string(
            majority=consensus,
            gemini_fn=gemini_confirm_fn,
            n_calls=config.GEMINI_TIEBREAKER_CALLS,
        )
        gemini_in += g_in
        gemini_out += g_out
        audit_entry["tiebreaker"] = tb_log

        if tb_value is None:
            # Tiebreaker could not reach confidence — document is low_confidence
            consensus = "low_confidence"
            low_confidence = True
            logger.warning(f"[Classifier] low_confidence — tiebreaker exhausted for {filename!r}")
        else:
            consensus = tb_value
            logger.info(f"[Classifier] tiebreaker outcome={tb_log['outcome']!r}  value={consensus!r}")

    audit_entry["final_value"] = consensus

    return {
        "document_type": consensus,
        "classification_rounds": len(rounds_log),
        "classification_unanimous": rounds_log[-1]["unanimous"],
        "input_tokens": state.get("input_tokens", 0) + in_tok,
        "output_tokens": state.get("output_tokens", 0) + out_tok,
        "gemini_input_tokens": gemini_in,
        "gemini_output_tokens": gemini_out,
        "low_confidence": low_confidence,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
