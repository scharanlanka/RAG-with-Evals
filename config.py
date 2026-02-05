import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()


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
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback local model
    PAGE_WINDOW = 2      # pages per chunk
    PAGE_OVERLAP = 1     # overlapping pages between chunks

    # Directories
    DATA_DIR = "data/documents"
    VECTOR_STORE_DIR = "vectorstore"

    # Streamlit Configuration
    PAGE_TITLE = "RAG Q&A System"
    PAGE_ICON = "AI"

