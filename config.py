import os

# Inference mode:
#   "pure"   — use the category backend for all calls (N agents, M rounds, voting as-is)
#   "hybrid" — use the category backend for voting; Gemini as tiebreaker when not unanimous
#   NOTE: INFERENCE_MODE="hybrid" raises ValueError at startup when all category backends are "gemini".
INFERENCE_MODE = "hybrid"

# Per-category inference backends — each independently set to "ollama", "huggingface", or "gemini"
RC_EC_BACKEND = "ollama"   # Registration Certificate / Extension Certificate
OC_CC_BACKEND = "ollama"   # Occupancy Certificate / Completion Certificate


def get_backend(category: str) -> str:
    """Return the configured inference backend for a document category."""
    return OC_CC_BACKEND if category == "oc_cc" else RC_EC_BACKEND


"""\
PICK FROM THESE 2 MODELS (Ollama / HuggingFace):
- qwen3.5:2b
- qwen3.5:9b
"""

CLASSIFIER_MODEL = "qwen3.5:2b"
EXTRACTOR_MODEL  = "qwen3.5:2b"

HF_API_URL     = "https://router.huggingface.co/v1/chat/completions"
OLLAMA_API_URL = "http://10.10.0.86:11434/api/chat"
OLLAMA_TIMEOUT = 300              # seconds; increase for larger models
OLLAMA_NUM_PREDICT = 5000         # max output tokens for classifier + extractor

# Gemini backend settings
GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY")
GEMINI_CLASSIFIER_MODEL = "gemini-2.5-flash-lite" # use gemini-2.5-flash or emini-2.5-flash-lite
GEMINI_EXTRACTOR_MODEL  = "gemini-2.5-flash-lite"
GEMINI_NUM_PREDICT      = 8192   # max output tokens
GEMINI_TIMEOUT          = 300    # seconds

# Hybrid mode: number of Gemini tiebreaker calls (including the first call).
# Call 1 checks against Ollama plurality. If it doesn't match, calls 2..N are made
# and unanimity among all N is required, otherwise the document is marked low_confidence.
GEMINI_TIEBREAKER_CALLS = 3

# Per-round Gemini validator: when True and a category's backend is not "gemini",
# unanimous base-backend results are confirmed by one Gemini call per round before accepting.
# If Gemini disagrees, the round is treated as non-unanimous and the loop continues.
# Independent of INFERENCE_MODE — both can be active simultaneously.
GEMINI_ROUND_VALIDATOR = True

MAX_PAGES         = 10
MAX_IMAGES_TO_VLM = 10

VOTING_N          = 5    # number of parallel agents per voting round
VOTING_MAX_ROUNDS = 3   # max rounds before falling back to plurality

PHASH_THRESHOLD   = 10   # max Hamming distance for perceptual hash duplicate detection

STORAGE_BACKEND   = "databricks"          # switch to "databricks" to use DatabricksStorageBackend
STORAGE_DIR       = "storage"
LOG_DIR           = "storage/logs"

# When True, Databricks backend also writes run logs locally under LOG_DIR
# (same format as JSONStorageBackend). Useful for debugging without full DB write support.
LOCAL_LOG_DUMP    = True

# When True, Databricks backend also writes ProjectData locally as storage/{rera_id}.json.
# The file is never read back — load() always goes to Databricks (or returns None).
LOCAL_JSON_DUMP   = True

# Databricks credentials — read from environment variables
DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH       = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_ACCESS_TOKEN    = os.getenv("DATABRICKS_ACCESS_TOKEN")

# Validate mode combination at import time
if INFERENCE_MODE == "hybrid" and RC_EC_BACKEND == "gemini" and OC_CC_BACKEND == "gemini":
    raise ValueError(
        "INFERENCE_MODE='hybrid' is invalid when all category backends are 'gemini'. "
        "Hybrid mode requires at least one non-Gemini category backend so that "
        "Gemini can act as a separate tiebreaker."
    )
