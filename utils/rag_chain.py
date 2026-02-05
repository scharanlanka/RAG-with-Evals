from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from openai import AzureOpenAI
from utils.vector_store import VectorStore
from config import Config


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = self._setup_client()
        self.logger = self._setup_logger()

    def _setup_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI chat completion client."""
        if not all(
            [
                Config.AZURE_LLM_ENDPOINT,
                Config.AZURE_LLM_API_KEY,
                Config.AZURE_LLM_DEPLOYMENT_NAME,
            ]
        ):
            raise ValueError(
                "Azure LLM configuration is incomplete. Please set endpoint, key, and deployment name."
            )
        return AzureOpenAI(
            api_key=Config.AZURE_LLM_API_KEY,
            api_version=Config.AZURE_LLM_API_VERSION,
            azure_endpoint=Config.AZURE_LLM_ENDPOINT,
        )

    def _setup_logger(self) -> logging.Logger:
        """Create a file logger for LLM calls (workspace root/logs)."""
        root_dir = Path(__file__).resolve().parents[1]
        log_dir = root_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "llm_calls.log"

        logger = logging.getLogger("llm_calls")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Avoid duplicate handlers across Streamlit reruns
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the query."""
        return self.vector_store.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown source")
            page_range = doc.metadata.get("page_range", "")
            suffix = f" [{page_range}]" if page_range else ""
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} (Source: {source}{suffix}):\n{content}")

        return "\n\n".join(context_parts)
    
    def _reformulate_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Use Azure OpenAI to reformulate the query with recent chat history."""
        history = chat_history or []
        last_turns = [m for m in history if m.get("role") in {"user", "assistant"}]
        last_turns = last_turns[-6:]  # limit to last 6 turns

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query reformulation assistant. Given recent conversation turns, rewrite the latest "
                    "user question into a clear, standalone question that captures prior context. Keep it concise."
                ),
            }
        ]

        for m in last_turns:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append(
            {
                "role": "system",
                "content": (
                    "Return only the reformulated question text without any prefixes, explanations, or citations."
                ),
            }
        )

        try:
            resp = self.client.chat.completions.create(
                model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.3,
            )
            reformulated = resp.choices[0].message.content.strip()
            self.logger.info(
                "Query reformulated | original=%s | reformulated=%s",
                query.replace("\n", " ")[:200],
                reformulated[:200],
            )
            return reformulated or query
        except Exception as e:
            self.logger.error("Reformulation failed, falling back to original query | error=%s", e)
            return query

    def generate_answer(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 4) -> Dict[str, Any]:
        """Generate answer using RAG pipeline."""
        try:
            reformulated_query = self._reformulate_query(query, chat_history)

            # Step 1: Retrieve relevant documents
            documents = self.retrieve_documents(reformulated_query, k=k)
            
            if not documents:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": ""
                }
            
            # Step 2: Format context
            context = self.format_context(documents)
            
            # Step 3: Build messages for Azure OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that answers questions using ONLY the provided context. "
                        "If the answer is not in the context, say you don't know."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {reformulated_query}\n\n"
                        "Answer concisely and cite which source snippets you used."
                    ),
                },
            ]

            # Step 4: Get response from Azure OpenAI
            self.logger.info(
                "LLM request | endpoint=%s | deployment=%s | k=%s | query_preview=%s",
                Config.AZURE_LLM_ENDPOINT,
                Config.AZURE_LLM_DEPLOYMENT_NAME,
                k,
                query.replace("\n", " ")[:200],
            )
            try:
                response = self.client.chat.completions.create(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.2,
                )
                answer = response.choices[0].message.content.strip()

                # Log call metadata (without sensitive data)
                self.logger.info(
                    "LLM response | endpoint=%s | deployment=%s | tokens_prompt=%s | tokens_completion=%s",
                    Config.AZURE_LLM_ENDPOINT,
                    Config.AZURE_LLM_DEPLOYMENT_NAME,
                    getattr(response.usage, "prompt_tokens", "n/a"),
                    getattr(response.usage, "completion_tokens", "n/a"),
                )
            except Exception as call_err:
                self.logger.error(
                    "LLM error | endpoint=%s | deployment=%s | error=%s",
                    Config.AZURE_LLM_ENDPOINT,
                    Config.AZURE_LLM_DEPLOYMENT_NAME,
                    call_err,
                )
                raise
            
            # Step 5: Extract sources with page ranges
            sources = []
            for doc in documents:
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page_range": doc.metadata.get("page_range", ""),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "query": query,
                "reformulated_query": reformulated_query,
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query,
                "reformulated_query": None,
            }
    
    def chat_with_context(self, query: str, chat_history: List[Dict] = None, k: int = 4) -> Dict[str, Any]:
        """Enhanced chat function with conversation history."""
        # For now, we'll implement basic RAG. 
        # You can extend this to include chat history in the prompt
        return self.generate_answer(query, k=k)
