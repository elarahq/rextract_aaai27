# RE-XTRACT

Unanimity-gated document extraction with calibrated abstention, for Indian
real-estate regulatory (RERA) filings, plus the AAAI-27 demonstration-track
paper describing it.

Every routing and extraction decision passes three gates over a self-hosted
open-weight VLM:

1. **Voting** — `M=5` votes from one model must agree exactly, retried up to
   `N-1` times
2. **Verification** — a frontier model must independently reproduce the
   unanimous answer
3. **Escalation and abstention** — `K-1` further frontier calls, and if those do
   not agree the pipeline writes nothing rather than guessing

## Layout

```
demo/     web UI that runs the pipeline live and streams its decision trace
paper/    AAAI-27 demo paper (LaTeX, AAAI AuthorKit27 style)
agents/   classifier and extractor, each wrapped in the voting stack
tools/    voting, escalation, PDF rendering, date normalisation, inference backends
db/       Databricks read path
config.py all tunable parameters
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the values
```

The open-weight model is served over Ollama at the host set in
`config.py` (`OLLAMA_API_URL`). That host must be reachable from wherever you
run the demo.

## Running the demo

```bash
uvicorn demo.server:app --port 8000
```

Open `http://localhost:8000`, pick one of the six preloaded projects, and the
pipeline runs on it live. The centre panel streams the pipeline's own log: every
vote as it lands, the validator's verdict, each round transition, and the record
written or the abstention.

### The demo never writes

It reads document URLs from Databricks through `get_document_urls()` and nothing
else. It deliberately does not import `graph.workflow` from the production
system, because that module builds a storage backend at import time and writes
in two places; its read path also issues `CREATE TABLE IF NOT EXISTS`. It also
performs no MD5 or perceptual-hash deduplication, so every document in a project
is processed and shown.

## Building the paper

```bash
cd paper && latexmk -pdf main.tex
```

Produces two pages of content plus one page of references, per the AAAI-27 call.
`paper/README.md` records what is still outstanding before submission.

`paper/reference/` holds accepted AAAI-26 demo papers kept for style comparison.
They are other authors' work and are gitignored.
