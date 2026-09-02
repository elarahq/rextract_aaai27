"""Read-only demo runner.

Runs the RE-XTRACT decision pipeline over one project's filings and emits
events for the browser.

Deliberately does NOT import graph.workflow: that module builds a storage
backend at import time and writes to Databricks in two places. It also runs
CREATE TABLE IF NOT EXISTS on the read path. Nothing here writes anywhere.
The only Databricks call is the single SELECT in get_document_urls().

Deduplication (MD5, perceptual hash, filename pre-filter) is not performed.
"""
import logging
import threading
import time
from typing import Callable, Optional

import config
from agents.classifier import classify_with_vote
from agents.extractor import extract_with_vote
from db.databricks_backend import DatabricksStorageBackend
from models import ExtensionDetail, OCCCDetail, ProjectData, RegistrationDetails
from tools.date_utils import _normalize_dates
from tools.pdf_processor import convert_to_images, download_pdf

Emit = Callable[[str, dict], None]

# Mirrors graph/workflow.py:464 minus the low_confidence sentinel, which the
# graph only needs so it can still persist an audit entry.
_EXTRACTABLE = {
    "registration certificate",
    "extension certificate",
    "occupancy certificate",
    "completion certificate",
}


# Log records reach the browser through a single root handler that broadcasts
# to every active run. Filtering by thread does not work: the M votes are issued
# from a ThreadPoolExecutor inside tools.voting._run_parallel, so they carry
# worker thread ids, not the run thread's. Broadcasting is correct for one run
# at a time, which is how the demo is used; two concurrent runs would each see
# the other's lines.
_SINKS: "list[Emit]" = []
_SINKS_LOCK = threading.Lock()
_HANDLER_INSTALLED = False

# Library chatter that adds nothing to the decision trace.
_NOISE_SUBSTRINGS = (
    "AFC is enabled",
    "Direct use of automatic function calling",
)
_NOISE_LOGGERS = ("httpx", "httpcore", "urllib3", "databricks", "google_genai.models")


def _is_noise(record: logging.LogRecord) -> bool:
    if record.name.startswith(_NOISE_LOGGERS):
        return True
    try:
        message = record.getMessage()
    except Exception:
        return True
    return any(n in message for n in _NOISE_SUBSTRINGS)


class _BroadcastLogHandler(logging.Handler):
    """Forward pipeline log records to every active run's event sink."""

    def emit(self, record: logging.LogRecord) -> None:
        if _is_noise(record):
            return
        try:
            payload = {
                "level": record.levelname,
                "logger": record.name,
                "text": record.getMessage(),
            }
        except Exception:
            return
        with _SINKS_LOCK:
            sinks = list(_SINKS)
        for sink in sinks:
            try:
                sink("log", payload)
            except Exception:
                pass


def _ensure_handler() -> None:
    global _HANDLER_INSTALLED
    if _HANDLER_INSTALLED:
        return
    handler = _BroadcastLogHandler()
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    _HANDLER_INSTALLED = True


def run_project(rera_id: str, emit: Emit) -> None:
    """Run the pipeline for one project, emitting events as it goes."""
    _ensure_handler()
    with _SINKS_LOCK:
        _SINKS.append(emit)
    try:
        _run(rera_id, emit)
    except Exception as exc:  # surfaced in the browser rather than a traceback
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
    finally:
        with _SINKS_LOCK:
            if emit in _SINKS:
                _SINKS.remove(emit)


def _run(rera_id: str, emit: Emit) -> None:
    started = time.time()
    totals = {"local_in": 0, "local_out": 0, "frontier_in": 0, "frontier_out": 0}

    emit("status", {"message": f"Fetching document list for {rera_id}"})
    urls = DatabricksStorageBackend().get_document_urls(rera_id) or []
    emit("documents", {
        "documents": [
            {"name": u.get("name") or u["url"].split("/")[-1],
             "category": u["category"], "url": u["url"]}
            for u in urls
        ]
    })
    if not urls:
        emit("status", {"message": "No documents returned for this project."})
        return

    project = ProjectData(rera_id=rera_id)

    for index, entry in enumerate(urls, 1):
        name = entry.get("name") or entry["url"].split("/")[-1]
        category = entry["category"]
        emit("doc_start", {
            "index": index, "total": len(urls), "name": name,
            "category": category, "url": entry["url"],
        })

        try:
            images = convert_to_images(
                download_pdf(entry["url"]), max_pages=config.MAX_PAGES
            )
        except Exception as exc:
            emit("doc_error", {"name": name, "message": f"{type(exc).__name__}: {exc}"})
            continue

        state = {
            "current_images": images,
            "document_category": category,
            "current_name": name,
            "hf_token": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "gemini_input_tokens": 0,
            "gemini_output_tokens": 0,
            "audit_log": [],
            "low_confidence": False,
        }

        classification = classify_with_vote(state)
        state.update(classification)
        emit("classify", {
            "name": name,
            "document_type": classification["document_type"],
            "rounds": classification["classification_rounds"],
            "unanimous": classification["classification_unanimous"],
            "audit": classification["audit_log"][-1],
        })

        if classification.get("low_confidence"):
            emit("abstained", {"name": name, "stage": "classification"})
            _accumulate(totals, state)
            continue

        doc_type = classification["document_type"].strip().lower()
        if doc_type not in _EXTRACTABLE:
            emit("skipped", {
                "name": name, "document_type": classification["document_type"],
            })
            _accumulate(totals, state)
            continue

        extraction = extract_with_vote(state)
        state.update(extraction)
        emit("extract", {
            "name": name,
            "fields": extraction["extracted_fields"],
            "rounds": extraction["extraction_rounds"],
            "unanimous": extraction["extraction_unanimous"],
            "audit": extraction["audit_log"][-1],
        })
        _accumulate(totals, state)

        if extraction.get("low_confidence"):
            emit("abstained", {"name": name, "stage": "extraction"})
            continue

        _merge(project, state, name)
        emit("record", {"record": project.model_dump(mode="json")})

    local = totals["local_in"] + totals["local_out"]
    frontier = totals["frontier_in"] + totals["frontier_out"]
    emit("summary", {
        "elapsed_seconds": round(time.time() - started, 1),
        "local_tokens": local,
        "frontier_tokens": frontier,
        "local_share": round(100 * local / (local + frontier), 1) if local + frontier else None,
    })


def _accumulate(totals: dict, state: dict) -> None:
    totals["local_in"] += state.get("input_tokens", 0)
    totals["local_out"] += state.get("output_tokens", 0)
    totals["frontier_in"] += state.get("gemini_input_tokens", 0)
    totals["frontier_out"] += state.get("gemini_output_tokens", 0)


def _merge(project: ProjectData, state: dict, doc_name: str) -> None:
    """Fold one document's fields into the in-memory record.

    Mirrors graph/workflow.py:298-334. Nothing is persisted.
    """
    fields = _normalize_dates(state.get("extracted_fields", {}))
    doc_type = fields.get("document_type", state.get("document_type", "")).lower()
    voting = {
        "classification_rounds": state.get("classification_rounds"),
        "classification_unanimous": state.get("classification_unanimous"),
        "extraction_rounds": state.get("extraction_rounds"),
        "extraction_unanimous": state.get("extraction_unanimous"),
    }

    if "registration" in doc_type:
        project.registration_details = RegistrationDetails(
            doc_id=doc_name,
            registration_date=fields.get("registration_date"),
            promised_completion_date=fields.get("promised_completion_date"),
            **voting,
        )
    elif "extension" in doc_type:
        project.extension_details.append(ExtensionDetail(
            doc_id=doc_name,
            extension_approval_date=fields.get("extension_approval_date"),
            revised_promised_completion_date=fields.get("revised_promised_completion_date"),
            **voting,
        ))
    elif any(k in doc_type for k in ("oc", "cc", "occupancy", "completion")):
        project.oc_cc_documents.append(OCCCDetail(
            doc_id=doc_name,
            issue_date=fields.get("oc_cc_issue_date"),
            type=fields.get("cert_type"),
            details=fields.get("details"),
            **voting,
        ))
