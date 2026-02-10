import os
import shutil
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
from config import Config

class VectorStore:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "vectorstore"):
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.embeddings = self._build_embeddings(embedding_model)
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

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
            collection_name="rag_collection"
        )
        self.vectorstore.persist()
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
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
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the vector store collection."""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None

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
