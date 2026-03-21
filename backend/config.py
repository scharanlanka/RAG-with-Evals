import os
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from both repo root and backend/.env when present.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _get_env(key: str, default=None):
    """Fetch environment variable and strip wrapping quotes/spaces."""
    value = os.getenv(key, default)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


def _get_endpoint(key: str):
    """Return base endpoint (scheme + netloc) if a full URL is provided."""
    raw = _get_env(key)
    if not raw:
        return raw
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw


class Config:
    # Azure OpenAI - Chat Completion
    AZURE_LLM_ENDPOINT = _get_endpoint("AZURE_LLM_ENDPOINT")
    AZURE_LLM_API_KEY = _get_env("AZURE_LLM_API_KEY")
    AZURE_LLM_DEPLOYMENT_NAME = _get_env("AZURE_LLM_DEPLOYMENT_NAME")
    AZURE_LLM_API_VERSION = _get_env("AZURE_LLM_API_VERSION", "2024-05-01-preview")

    # Azure OpenAI - Embeddings
    AZURE_EMBEDDING_ENDPOINT = _get_endpoint("AZURE_EMBEDDING_ENDPOINT")
    AZURE_EMBEDDING_API_KEY = _get_env("AZURE_EMBEDDING_API_KEY")
    AZURE_EMBEDDING_DEPLOYMENT_NAME = _get_env("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    AZURE_EMBEDDING_API_VERSION = _get_env("AZURE_EMBEDDING_API_VERSION", "2023-05-15")

    # Vector Store Configuration
    VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
    CHROMA_DISTANCE_METRIC = _get_env("CHROMA_DISTANCE_METRIC", "cosine")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback local model
    PAGE_WINDOW = 2      # pages per chunk
    PAGE_OVERLAP = 1     # overlapping pages between chunks

    # Directories
    DATA_DIR = str(BACKEND_DIR / "data" / "documents")
    VECTOR_STORE_DIR = str(BACKEND_DIR / "vectorstore")
    LOGS_DIR = str(BACKEND_DIR / "logs")
