"""
Shared settings for the API, the ingestion pipeline and the evals.

Every value can be overridden with an environment variable (or .env).
Keep the embedding settings here only: queries and documents must be
embedded with exactly the same model and options.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ============================================================
# PATHS
# ============================================================

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NEW_STRUCTURED_DIR = DATA_DIR / "new_structured"
# Last accepted copy of the site (compared against every new scrape).
# Was data/structured_final; that folder was created owner-only by the
# old promote.py (tempfile.mkdtemp) and can become unreadable.
BASELINE_DIR = DATA_DIR / "baseline"
COMPARISON_DIR = DATA_DIR / "comparison"
CHANGES_FILE = COMPARISON_DIR / "changes.json"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", DATA_DIR / "vectorstore"))
PIPELINE_LOG_DIR = DATA_DIR / "pipeline_logs"

# Written by the pipeline after every successful vector update.
# The API watches it and reloads its in-memory index when it changes.
INDEX_VERSION_FILE = VECTORSTORE_DIR / "index_version.json"


# ============================================================
# EMBEDDINGS / VECTOR STORE
# ============================================================

COLLECTION_NAME = "consiva_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 16)


def embedding_kwargs() -> dict:
    """Keyword arguments for langchain_huggingface.HuggingFaceEmbeddings."""
    return {
        "model_name": EMBEDDING_MODEL,
        "model_kwargs": {"device": EMBEDDING_DEVICE},
        "encode_kwargs": {"normalize_embeddings": True},
    }


def use_offline_model_cache() -> None:
    """
    Skip the ~40 Hugging Face network checks per load when the model
    is already downloaded. Falls back to online mode on first use.
    """
    if os.getenv("HF_HUB_OFFLINE"):
        return

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(EMBEDDING_MODEL, "config.json")
    except Exception:
        cached = None

    if isinstance(cached, str):
        os.environ["HF_HUB_OFFLINE"] = "1"


# ============================================================
# CHUNKING
# ============================================================

# Bump this whenever chunking or cleaning output changes.
# The vector updater re-indexes every page whose vectors were built
# with a different version, even if the page content is unchanged.
CHUNKER_VERSION = "2026-10-10.3"

CHUNK_TARGET_CHARS = _int("CHUNK_TARGET_CHARS", 1200)
CHUNK_MIN_CHARS = _int("CHUNK_MIN_CHARS", 350)
CHUNK_MAX_CHARS = _int("CHUNK_MAX_CHARS", 1800)
CHUNK_OVERLAP_CHARS = _int("CHUNK_OVERLAP_CHARS", 200)


# ============================================================
# RETRIEVAL
# ============================================================

# Candidates taken from each retriever before fusion.
RETRIEVAL_CANDIDATES = _int("RETRIEVAL_CANDIDATES", 20)

# Chunks finally sent to the LLM.
TOP_K = _int("TOP_K", 8)

# At most this many chunks from the same page.
MAX_CHUNKS_PER_PAGE = _int("MAX_CHUNKS_PER_PAGE", 4)

# If the best dense cosine similarity is below this, the question is
# treated as out of scope and the LLM is not called. Measured on bge-m3:
# clearly off-topic questions score 0.33-0.36, real questions 0.50+.
# Questions in between go to the LLM, which says "not found" when the
# sources do not answer them.
MIN_RELEVANCE = _float("MIN_RELEVANCE", 0.38)

# Chunks scoring this far below the best one are dropped.
RELATIVE_RELEVANCE_MARGIN = _float("RELATIVE_RELEVANCE_MARGIN", 0.2)


# ============================================================
# LLM
# ============================================================

# OpenAI (OPEN_AI_API_KEY is accepted as an alias of the standard name).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
# gpt-4.1-mini: ~1-2 s answers, low temperature for consistent wording.
# gpt-5.4-mini scored the same on the eval (reasoning model, ~2-3 s).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Reasoning models (gpt-5.x, o-series) only: none / low / medium / high.
# Answering from retrieved text needs little reasoning; "low" keeps it fast.
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# "openai" or "gemini". Defaults to OpenAI when its key is set.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or ("openai" if OPENAI_API_KEY else "gemini")).lower()

LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 45)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 2)

# Thinking tokens for Gemini 2.5+/3.x. 0 = off: answering from retrieved
# text needs no long reasoning, and thinking made answers take ~60 s.
# Use -1 (model decides) for models that require thinking.
LLM_THINKING_BUDGET = _int("LLM_THINKING_BUDGET", 0)

# Conversation turns (user + assistant messages) used for follow-ups.
MAX_HISTORY_MESSAGES = _int("MAX_HISTORY_MESSAGES", 6)

NOT_FOUND_ANSWER = (
    "I couldn't find that information in the Consiva knowledge base. "
    "You can ask about Consiva's plans, features, policies or the DPDP Act, "
    "or contact sales@consiva.ai."
)


# ============================================================
# API
# ============================================================

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 1000)
RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", 20)


# ============================================================
# PIPELINE
# ============================================================

# Scraping continues when at most this share of pages fails;
# failed pages keep their last good raw copy.
SCRAPE_MAX_FAILURE_RATIO = _float("SCRAPE_MAX_FAILURE_RATIO", 0.1)

# Refuse to delete pages if the sitemap suddenly shrinks this much.
SITEMAP_MIN_KEEP_RATIO = _float("SITEMAP_MIN_KEEP_RATIO", 0.7)

STAGE_TIMEOUTS_SECONDS = {
    "scraping": _int("TIMEOUT_SCRAPING", 60 * 60),
    "cleaning": _int("TIMEOUT_CLEANING", 10 * 60),
    "comparison": _int("TIMEOUT_COMPARISON", 10 * 60),
    "vector_update": _int("TIMEOUT_VECTOR_UPDATE", 120 * 60),
}

# Optional Slack / Teams / generic webhook for failure alerts.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
