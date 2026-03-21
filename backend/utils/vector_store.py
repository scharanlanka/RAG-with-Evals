import os
import shutil
import time
import logging
from pathlib import Path
from collections import OrderedDict
from typing import List
import chromadb
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# LangChain >= 0.2 community imports
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    from langchain.embeddings import SentenceTransformerEmbeddings

from langchain_openai import AzureOpenAIEmbeddings
from backend.config import Config

class VectorStore:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "vectorstore"):
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.embeddings = self._build_embeddings(embedding_model)
        self.vectorstore = None
        self.logger = self._setup_logger()
        self._query_embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._query_embedding_cache_max = 128
        self.collection_name = "rag_collection"
        self.collection_metadata = {
            "hnsw:space": (Config.CHROMA_DISTANCE_METRIC or "cosine").strip().lower()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Create a file logger for retrieval timing diagnostics."""
        root_dir = Path(__file__).resolve().parents[1]
        log_dir = root_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "retrieval_perf.log"

        logger = logging.getLogger("retrieval_perf")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _build_embeddings(self, embedding_model: str):
        """Return an embedding model, preferring Azure OpenAI when configured."""
        if (
            Config.AZURE_EMBEDDING_ENDPOINT
            and Config.AZURE_EMBEDDING_API_KEY
            and Config.AZURE_EMBEDDING_DEPLOYMENT_NAME
        ):
            return AzureOpenAIEmbeddings(
                model=Config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                azure_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                api_key=Config.AZURE_EMBEDDING_API_KEY,
                azure_endpoint=Config.AZURE_EMBEDDING_ENDPOINT,
                api_version=Config.AZURE_EMBEDDING_API_VERSION,
            )
        # Fallback to local sentence transformer
        return SentenceTransformerEmbeddings(model_name=embedding_model)
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata=self.collection_metadata,
        )
        self.vectorstore.persist()
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                collection_metadata=self.collection_metadata,
            )
            self._warn_if_metric_mismatch()
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False

    def _warn_if_metric_mismatch(self) -> None:
        """Log when an existing collection still uses a different distance metric."""
        if self.vectorstore is None:
            return
        try:
            metadata = getattr(self.vectorstore._collection, "metadata", None) or {}
            actual = str(metadata.get("hnsw:space", "l2")).strip().lower()
            expected = str(self.collection_metadata.get("hnsw:space", "cosine")).strip().lower()
            if actual != expected:
                self.logger.warning(
                    (
                        "Chroma collection metric mismatch | collection=%s | expected=%s | actual=%s. "
                        "Existing index keeps its original metric; clear vectorstore and re-index to migrate."
                    ),
                    self.collection_name,
                    expected,
                    actual,
                )
        except Exception:
            # Metadata shape is version-dependent; skip warning when unavailable.
            pass
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        try:
            if self.vectorstore is None:
                self.create_vectorstore(documents)
            else:
                self.vectorstore.add_documents(documents)
                self.vectorstore.persist()
        except Exception as e:
            if not self._is_sqlite_readonly_error(e):
                raise
            self._recover_after_readonly_error()
            try:
                if self.vectorstore is None:
                    self.create_vectorstore(documents)
                else:
                    self.vectorstore.add_documents(documents)
                    self.vectorstore.persist()
            except Exception as retry_error:
                if not self._is_sqlite_readonly_error(retry_error):
                    raise
                self._reset_persisted_store()
                self.create_vectorstore(documents)
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4) -> List[Document]:
        """Perform similarity search and return relevant documents."""
        if self.vectorstore is None:
            return []

        normalized_query = " ".join((query or "").split())
        cache_hit = normalized_query in self._query_embedding_cache
        t_total_start = time.perf_counter()
        t_embed = 0.0
        t_search = 0.0

        try:
            if cache_hit:
                query_embedding = self._query_embedding_cache[normalized_query]
                self._query_embedding_cache.move_to_end(normalized_query)
            else:
                t0 = time.perf_counter()
                query_embedding = self.embeddings.embed_query(normalized_query)
                t_embed = time.perf_counter() - t0
                self._query_embedding_cache[normalized_query] = query_embedding
                if len(self._query_embedding_cache) > self._query_embedding_cache_max:
                    self._query_embedding_cache.popitem(last=False)

            t0 = time.perf_counter()
            docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=k)
            t_search = time.perf_counter() - t0
            self.logger.info(
                "PERF retrieval | cache_hit=%s | embed=%.3fs | vector_search=%.3fs | total=%.3fs | k=%s | query_preview=%s",
                cache_hit,
                t_embed,
                t_search,
                time.perf_counter() - t_total_start,
                k,
                normalized_query[:160],
            )
            return docs
        except Exception:
            # Fallback path for vector store variants lacking similarity_search_by_vector.
            t0 = time.perf_counter()
            docs = self.vectorstore.similarity_search(normalized_query, k=k)
            t_search = time.perf_counter() - t0
            self.logger.info(
                "PERF retrieval_fallback | cache_hit=%s | embed=%.3fs | search=%.3fs | total=%.3fs | k=%s | query_preview=%s",
                cache_hit,
                t_embed,
                t_search,
                time.perf_counter() - t_total_start,
                k,
                normalized_query[:160],
            )
            return docs
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the vector store collection."""
        if self.vectorstore is None:
            return
        try:
            self.vectorstore.delete_collection()
        except Exception:
            # Collection may already be missing/corrupt; continue with cleanup.
            pass
        finally:
            self.vectorstore = None

    def clear_persisted_store(self) -> None:
        """Completely remove persisted Chroma data from disk."""
        self.vectorstore = None
        self._clear_chroma_system_cache()
        shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def get_indexed_filenames(self) -> List[str]:
        """Best-effort list of unique file names currently indexed."""
        if self.vectorstore is None:
            return []
        try:
            result = self.vectorstore._collection.get(include=["metadatas"])
            names = set()
            for metadata in result.get("metadatas", []) or []:
                if not isinstance(metadata, dict):
                    continue
                filename = metadata.get("filename")
                if filename:
                    names.add(str(filename))
            return sorted(names)
        except Exception:
            return []

    def _is_sqlite_readonly_error(self, error: Exception) -> bool:
        """Detect SQLite readonly/moved-db errors surfaced by Chroma."""
        message = str(error).lower()
        return (
            "readonly" in message
            or "code: 1032" in message
            or "attempt to write a readonly database" in message
        )

    def _recover_after_readonly_error(self) -> None:
        """Re-open Chroma to clear stale SQLite handles and restore writes."""
        self.vectorstore = None
        os.makedirs(self.persist_directory, exist_ok=True)
        self._clear_chroma_system_cache()
        self.load_vectorstore()

    def _reset_persisted_store(self) -> None:
        """Reset local Chroma files when DB becomes permanently readonly."""
        self.vectorstore = None
        self._clear_chroma_system_cache()
        shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def _clear_chroma_system_cache(self) -> None:
        """Best-effort clear of Chroma's shared system cache across versions."""
        try:
            from chromadb.api.client import SharedSystemClient
            clear_fn = getattr(SharedSystemClient, "clear_system_cache", None)
            if callable(clear_fn):
                clear_fn()
        except Exception:
            # Cache clear API is version-dependent; ignore when unavailable.
            pass
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        if self.vectorstore is None:
            return {"status": "No collection loaded", "count": 0}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {"status": "Collection loaded", "count": count}
        except Exception as e:

            return {"status": f"Error: {str(e)}", "count": 0}
