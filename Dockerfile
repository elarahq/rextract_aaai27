# RE-XTRACT demo — AAAI-27 demonstration track
#
# Build (from the repository root):
#   docker build --platform linux/amd64 -t rextract-demo:1.0 .
#
# Run locally:
#   docker run --rm -p 8000:8000 --env-file .env rextract-demo:1.0
#
# The image contains NO credentials. GEMINI_API_KEY and the three DATABRICKS_*
# variables must be supplied at runtime (see deploy/k8s.yaml).

FROM python:3.11-slim-bookworm

# PYTHONUNBUFFERED matters here: the demo streams the pipeline's own log to the
# browser over SSE, and a buffered stdout also hides startup errors from kubectl logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so application edits do not invalidate this layer.
# Every dependency ships a manylinux wheel for amd64, so no compiler is needed.
# If a wheel is ever missing for your architecture, add before this line:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code. paper/, .venv/ and .env are excluded by .dockerignore.
COPY config.py models.py ./
COPY agents/ ./agents/
COPY db/ ./db/
COPY tools/ ./tools/
COPY demo/ ./demo/

# Run unprivileged. The demo writes nothing to disk, so the container is
# compatible with readOnlyRootFilesystem: true.
RUN useradd --create-home --uid 10001 app
USER 10001

EXPOSE 8000

# One worker on purpose. Each run streams from a background thread pool, and a
# single worker keeps the log trace in the video coherent. Scale with replicas
# rather than workers if you ever need to.
CMD ["uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]
