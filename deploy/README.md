# RE-XTRACT demo — deployment notes

A single-container FastAPI service. A visitor picks one of six preloaded RERA
projects; the service runs the extraction pipeline live and streams its decision
trace to the browser. It backs the AAAI-27 demonstration-track submission.

## Build

```bash
docker build --platform linux/amd64 -t <registry>/rextract-demo:1.0 .
docker push <registry>/rextract-demo:1.0
```

Then set that image path in `deploy/k8s.yaml`.

## Secrets (required)

The application's only contract is that these four environment variables are set
in the container. Nothing is baked into the image, and nothing is read from a
file on disk.

| variable | purpose |
|---|---|
| `GEMINI_API_KEY` | frontier validator (gemini-2.5-flash-lite) |
| `DATABRICKS_SERVER_HOSTNAME` | source of document URLs |
| `DATABRICKS_HTTP_PATH` | " |
| `DATABRICKS_ACCESS_TOKEN` | " |

`k8s.yaml` sources them with `envFrom.secretRef`, pointing at a Secret named
`rextract-demo-secrets`. How that Secret gets populated is your platform's
business and the Deployment does not care -- External Secrets Operator
(`ExternalSecret` -> `target.name: rextract-demo-secrets`), Vault Agent
injection, or a Secrets Store CSI driver with `secretObjects` all produce the
same result. If your secret manager injects env vars directly rather than
through a Secret, delete the `envFrom` block and let it do so; the four names
above are all that matter.

Databricks access is **read-only**: the service calls `get_document_urls()` and
nothing else. It issues no writes and no DDL.

## Python version

Development venv: **3.11.9** (macOS/arm64). Image base: **`python:3.11-slim-bookworm`**
(linux/amd64). Same minor version, so every `cp311` wheel resolves identically;
the patch level floats so the image picks up 3.11 security updates. Pin to
`python:3.11.9-slim-bookworm` only if you need byte-identical parity with the
dev environment, accepting that it then freezes on known CVEs.

The architecture differs (arm64 dev, amd64 image), which is why the build uses
`--platform linux/amd64`. `requirements.lock.txt` contains no platform-specific
packages, so it is valid on both.

## Egress — the thing most likely to break

The pod must reach all four of these. Anything blocked, and a run either hangs
or aborts partway:

| destination | port | why |
|---|---|---|
| `10.10.0.86` | 11434 | self-hosted Ollama; every local vote goes here |
| `generativelanguage.googleapis.com` | 443 | Gemini validator |
| `$DATABRICKS_SERVER_HOSTNAME` | 443 | document URLs |
| `d38irxr1xt4zpo.cloudfront.net`, `xbyte-rera-documents-housing.s3.ap-south-1.amazonaws.com`, `dl.dropboxusercontent.com`, `www.dropbox.com` | 443 | the filings themselves |

The Ollama address is a private IP set in `config.py` (`OLLAMA_API_URL`); it has
been confirmed reachable from the container.

## Ingress

`/api/run` is a Server-Sent Events stream held open for the length of a pipeline
run — minutes, not seconds. Two consequences:

- **Proxy buffering must be off.** With nginx's default buffering the page shows
  nothing at all until the run finishes, which looks exactly like a hang.
- **Read timeouts must exceed the longest run.** The manifest sets 1800s.

Both are set as annotations in `k8s.yaml`. If you front this with something
other than ingress-nginx, carry the equivalent settings across.

## Resources and scaling

No persistent storage. The service writes nothing to disk, so it runs with
`readOnlyRootFilesystem: true` and an emptyDir on `/tmp`.

Work is dominated by waiting on Ollama and Gemini rather than local CPU;
requests are 500m/512Mi with limits at 2 CPU / 2Gi, the headroom being for
PyMuPDF rendering up to 10 pages at 150 DPI with 5 votes in flight.

One replica is deliberate. Each request is self-contained so more would be
correct, but a demo has one viewer at a time and a single pod keeps the log
trace coherent while recording. Note that the in-process PDF cache
(`demo/server.py`, `_pdf_cache`) is per-pod and unbounded — bounded in practice
by six fixed projects, but it does not evict.

## Verifying a deployment

```bash
kubectl -n rextract-demo port-forward svc/rextract-demo 8000:80
curl -s localhost:8000/api/config          # config constants, no external calls
curl -s localhost:8000/api/projects        # the six preloaded projects
curl -N "localhost:8000/api/run?rera_id=<id>"   # full run, streams events
```

`/api/config` is what the probes hit; it touches no external service, so a
healthy pod there does **not** prove egress works. `/api/run` is the real test.
