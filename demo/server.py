"""FastAPI server for the RE-XTRACT demo.

Run from the repository root:
    ./.venv/bin/uvicorn demo.server:app --port 8000
"""
import json
import queue
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must precede the config import: config reads env at import time

import config

# Belt and braces. Nothing in this package calls save(), but if a future edit
# does, this stops it writing local JSON alongside Databricks.
config.LOCAL_JSON_DUMP = False

from urllib.parse import urlparse  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402

from demo.projects import PROJECTS  # noqa: E402
from demo.runner import run_project  # noqa: E402
from tools.pdf_processor import _resolve_s3_url, download_pdf  # noqa: E402

app = FastAPI(title="RE-XTRACT demo")
STATIC = Path(__file__).parent / "static"
_SENTINEL = object()

# Hosts the pipeline already fetches from. The PDF proxy refuses anything else,
# so this endpoint cannot be used to make the server fetch arbitrary URLs.
_ALLOWED_HOSTS = {
    "d38irxr1xt4zpo.cloudfront.net",
    "xbyte-rera-documents-housing.s3.ap-south-1.amazonaws.com",
    "dl.dropboxusercontent.com",
    "www.dropbox.com",
}
_pdf_cache: "dict[str, bytes]" = {}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/projects")
def projects():
    return PROJECTS


@app.get("/api/config")
def runtime_config():
    """Shown in the UI so the video records which models actually ran."""
    return {
        "local_model": config.CLASSIFIER_MODEL,
        "frontier_model": config.GEMINI_CLASSIFIER_MODEL,
        "votes_per_round": config.VOTING_N,
        "max_rounds": config.VOTING_MAX_ROUNDS,
        "frontier_calls": config.GEMINI_TIEBREAKER_CALLS,
        "rc_ec_backend": config.RC_EC_BACKEND,
        "oc_cc_backend": config.OC_CC_BACKEND,
        "round_validator": config.GEMINI_ROUND_VALIDATOR,
        "inference_mode": config.INFERENCE_MODE,
    }


@app.get("/api/pdf")
def pdf(url: str = Query(..., description="Document URL from the current run")):
    """Serve a filing so the browser can render it beside the extracted values.

    Proxied rather than linked directly because the source hosts do not allow
    framing. Restricted to the hosts the pipeline itself reads from.
    """
    host = urlparse(_resolve_s3_url(url)).netloc
    if host not in _ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail=f"host not allowed: {host}")
    if url not in _pdf_cache:
        try:
            _pdf_cache[url] = download_pdf(url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    return Response(
        content=_pdf_cache[url],
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/run")
def run(rera_id: str = Query(..., description="RERA project identifier")):
    events: "queue.Queue" = queue.Queue()

    def emit(event: str, payload: dict) -> None:
        events.put((event, payload))

    def worker() -> None:
        try:
            run_project(rera_id, emit)
        finally:
            events.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is _SENTINEL:
                yield "event: done\ndata: {}\n\n"
                return
            event, payload = item
            body = json.dumps(payload, default=str)
            yield f"event: {event}\ndata: {body}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
