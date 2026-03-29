import os
import shutil
import time
import logging
import re
import math
from pathlib import Path
from collections import OrderedDict
from collections import Counter
from typing import List, Dict, Any, Tuple
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
                         k: int = 4,
                         hybrid_enabled: bool = False,
                         keyword_weight: float = 0.3,
                         semantic_weight: float = 0.7) -> List[Document]:
        """Perform similarity search and return relevant documents."""
        if self.vectorstore is None:
            return []
        if hybrid_enabled:
            return [doc for doc, _ in self.hybrid_search_with_score(
                query,
                k=k,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
            )]

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
                                   k: int = 4,
                                   hybrid_enabled: bool = False,
                                   keyword_weight: float = 0.3,
                                   semantic_weight: float = 0.7) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vectorstore is None:
            return []
        if hybrid_enabled:
            return self.hybrid_search_with_score(
                query,
                k=k,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
            )
        
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _normalize_hybrid_weights(
        self,
        keyword_weight: float,
        semantic_weight: float,
    ) -> Tuple[float, float]:
        kw = max(0.0, float(keyword_weight))
        sw = max(0.0, float(semantic_weight))
        total = kw + sw
        if total <= 0:
            return 0.3, 0.7
        return kw / total, sw / total

    def _tokenize_bm25(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _build_bm25_scores(self, query: str, docs: List[str]) -> List[float]:
        """Compute BM25 scores for all docs for a query."""
        if not docs:
            return []
        query_terms = self._tokenize_bm25(query)
        if not query_terms:
            return [0.0 for _ in docs]

        tokenized_docs = [self._tokenize_bm25(d) for d in docs]
        doc_lens = [len(tokens) for tokens in tokenized_docs]
        avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
        k1 = 1.5
        b = 0.75

        term_doc_freq: Dict[str, int] = {}
        for term in set(query_terms):
            df = 0
            for tokens in tokenized_docs:
                if term in tokens:
                    df += 1
            term_doc_freq[term] = df

        n_docs = len(tokenized_docs)
        scores: List[float] = []
        for idx, tokens in enumerate(tokenized_docs):
            tf = Counter(tokens)
            dl = doc_lens[idx]
            score = 0.0
            for term in query_terms:
                term_tf = tf.get(term, 0)
                if term_tf == 0:
                    continue
                df = term_doc_freq.get(term, 0)
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = term_tf + k1 * (1.0 - b + b * (dl / max(avg_dl, 1e-9)))
                score += idf * ((term_tf * (k1 + 1.0)) / max(denom, 1e-9))
            scores.append(score)
        return scores

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[tuple]:
        """Hybrid retrieval using BM25 keyword score + semantic similarity score."""
        if self.vectorstore is None:
            return []

        t_total_start = time.perf_counter()
        normalized_query = " ".join((query or "").split())
        keyword_weight, semantic_weight = self._normalize_hybrid_weights(
            keyword_weight,
            semantic_weight,
        )

        try:
            collection = self.vectorstore._collection
            all_rows = collection.get(include=["documents", "metadatas"])
            ids = list(all_rows.get("ids", []) or [])
            texts = list(all_rows.get("documents", []) or [])
            metadatas = list(all_rows.get("metadatas", []) or [])
            if not ids or not texts:
                return []

            bm25_scores = self._build_bm25_scores(normalized_query, texts)
            max_bm25 = max(bm25_scores) if bm25_scores else 0.0
            bm25_norm = [
                (score / max_bm25) if max_bm25 > 0 else 0.0 for score in bm25_scores
            ]

            # Semantic pool > k so hybrid can reshuffle based on BM25.
            semantic_pool = min(max(k * 6, 20), len(ids))
            semantic_rows = self.vectorstore.similarity_search_with_score(
                normalized_query,
                k=semantic_pool,
            )
            semantic_similarity_by_key: Dict[Tuple[str, str, str], float] = {}
            for doc, score in semantic_rows:
                filename = str(doc.metadata.get("filename", ""))
                page_range = str(doc.metadata.get("page_range", ""))
                content = str(doc.page_content)
                # Chroma score is distance-like for cosine/l2 configs; convert to similarity.
                sim = max(0.0, min(1.0, 1.0 - float(score)))
                semantic_similarity_by_key[(filename, page_range, content)] = sim

            fused: List[Tuple[Document, float, float, float]] = []
            for idx, text in enumerate(texts):
                metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
                filename = str(metadata.get("filename", ""))
                page_range = str(metadata.get("page_range", ""))
                sem = semantic_similarity_by_key.get((filename, page_range, str(text)), 0.0)
                kw = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
                hybrid_similarity = (semantic_weight * sem) + (keyword_weight * kw)
                doc = Document(page_content=str(text or ""), metadata=dict(metadata))
                # Preserve UI assumption that score is "distance-like" and lower is better.
                hybrid_distance = max(0.0, min(1.0, 1.0 - hybrid_similarity))
                fused.append((doc, hybrid_distance, sem, kw))

            fused.sort(key=lambda row: row[1])  # lowest distance (highest fused similarity) first
            result = [(doc, score) for doc, score, _, _ in fused[:k]]
            self.logger.info(
                (
                    "PERF retrieval_hybrid | bm25_weight=%.3f | semantic_weight=%.3f | "
                    "docs=%s | total=%.3fs | k=%s | query_preview=%s"
                ),
                keyword_weight,
                semantic_weight,
                len(texts),
                time.perf_counter() - t_total_start,
                k,
                normalized_query[:160],
            )
            return result
        except Exception:
            self.logger.exception(
                "Hybrid retrieval failed, falling back to semantic search | query_preview=%s",
                normalized_query[:160],
            )
            return self.vectorstore.similarity_search_with_score(normalized_query, k=k)
    
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
