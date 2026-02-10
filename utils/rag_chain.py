from typing import List, Dict, Any, Optional
import logging
import os
import time
from pathlib import Path
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from openai import AzureOpenAI
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    ChatCompletionsClient = None
    SystemMessage = None
    UserMessage = None
    AssistantMessage = None
    AzureKeyCredential = None
from utils.vector_store import VectorStore
from config import Config


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        endpoint = (Config.AZURE_LLM_ENDPOINT or "").lower()
        self.use_foundry_models = "services.ai.azure.com" in endpoint
        self.client_type = "azure_openai"
        self.client = self._setup_client()
        self.logger = self._setup_logger()

    def _setup_client(self):
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
        if self.use_foundry_models:
            if not all([ChatCompletionsClient, AzureKeyCredential]):
                raise ImportError(
                    "azure-ai-inference and azure-core are required for Foundry model streaming."
                )
            self.client_type = "azure_inference"
            endpoint = f"{Config.AZURE_LLM_ENDPOINT.rstrip('/')}/models"
            return ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(Config.AZURE_LLM_API_KEY),
            )
        return AzureOpenAI(
            api_key=Config.AZURE_LLM_API_KEY,
            api_version=Config.AZURE_LLM_API_VERSION,
            azure_endpoint=Config.AZURE_LLM_ENDPOINT,
        )

    def _to_inference_message(self, role: str, content: str):
        """Convert chat role/content into Azure AI Inference message objects."""
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            return AssistantMessage(content=content)
        return UserMessage(content=content)

    def _extract_message_text(self, message: Any) -> str:
        """Extract plain text from completion message payloads."""
        if message is None:
            return ""
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                    continue
                if isinstance(item, dict):
                    candidate = item.get("text")
                    if candidate:
                        parts.append(str(candidate))
            return "".join(parts)
        return str(content or "")

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
            if self.client_type == "azure_inference":
                inference_messages = [
                    self._to_inference_message(m["role"], m["content"]) for m in messages
                ]
                resp = self.client.complete(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=inference_messages,
                    temperature=0.3,
                )
                reformulated = self._extract_message_text(resp.choices[0].message).strip()
            else:
                resp = self.client.chat.completions.create(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.3,
                )
                reformulated = self._extract_message_text(resp.choices[0].message).strip()
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
        t_total_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            reformulated_query = self._reformulate_query(query, chat_history)
            t_reformulate = time.perf_counter() - t0

            # Step 1: Retrieve relevant documents
            t0 = time.perf_counter()
            documents = self.retrieve_documents(reformulated_query, k=k)
            t_retrieve = time.perf_counter() - t0
            
            if not documents:
                self.logger.info(
                    "PERF no_docs | reformulate=%.3fs | retrieve=%.3fs | total=%.3fs",
                    t_reformulate,
                    t_retrieve,
                    time.perf_counter() - t_total_start,
                )
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": ""
                }
            
            # Step 2: Format context
            t0 = time.perf_counter()
            context = self.format_context(documents)
            t_context = time.perf_counter() - t0
            
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
                t0 = time.perf_counter()
                if self.client_type == "azure_inference":
                    inference_messages = [
                        self._to_inference_message(m["role"], m["content"]) for m in messages
                    ]
                    response = self.client.complete(
                        model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                        messages=inference_messages,
                        temperature=0.2,
                    )
                    answer = self._extract_message_text(response.choices[0].message).strip()
                else:
                    response = self.client.chat.completions.create(
                        model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                        messages=messages,
                        temperature=0.2,
                    )
                    answer = self._extract_message_text(response.choices[0].message).strip()
                t_llm = time.perf_counter() - t0

                # Log call metadata (without sensitive data)
                self.logger.info(
                    "LLM response | endpoint=%s | deployment=%s | tokens_prompt=%s | tokens_completion=%s",
                    Config.AZURE_LLM_ENDPOINT,
                    Config.AZURE_LLM_DEPLOYMENT_NAME,
                    getattr(response.usage, "prompt_tokens", "n/a"),
                    getattr(response.usage, "completion_tokens", "n/a"),
                )
                self.logger.info(
                    "PERF sync_answer | reformulate=%.3fs | retrieve=%.3fs | context=%.3fs | llm=%.3fs | total=%.3fs",
                    t_reformulate,
                    t_retrieve,
                    t_context,
                    t_llm,
                    time.perf_counter() - t_total_start,
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

    def _extract_delta_text(self, delta: Any) -> str:
        """Extract plain text from streaming delta payloads."""
        if delta is None:
            return ""

        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                    continue
                if isinstance(item, dict):
                    candidate = item.get("text")
                    if candidate:
                        parts.append(str(candidate))
            return "".join(parts)
        return ""

    def generate_answer_stream(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 4
    ) -> Dict[str, Any]:
        """Generate answer with token streaming."""
        t_total_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            reformulated_query = self._reformulate_query(query, chat_history)
            t_reformulate = time.perf_counter() - t0

            t0 = time.perf_counter()
            documents = self.retrieve_documents(reformulated_query, k=k)
            t_retrieve = time.perf_counter() - t0
            if not documents:
                self.logger.info(
                    "PERF stream_no_docs | reformulate=%.3fs | retrieve=%.3fs | total=%.3fs",
                    t_reformulate,
                    t_retrieve,
                    time.perf_counter() - t_total_start,
                )
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": "",
                    "query": query,
                    "reformulated_query": reformulated_query,
                }

            t0 = time.perf_counter()
            context = self.format_context(documents)
            t_context = time.perf_counter() - t0
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

            self.logger.info(
                "LLM stream request | endpoint=%s | deployment=%s | k=%s | query_preview=%s",
                Config.AZURE_LLM_ENDPOINT,
                Config.AZURE_LLM_DEPLOYMENT_NAME,
                k,
                query.replace("\n", " ")[:200],
            )

            t0 = time.perf_counter()
            response_stream = self.client.chat.completions.create(
                model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.2,
                stream=True,
            ) if self.client_type != "azure_inference" else self.client.complete(
                model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                messages=[self._to_inference_message(m["role"], m["content"]) for m in messages],
                temperature=0.2,
                stream=True,
            )
            t_stream_open = time.perf_counter() - t0

            sources = []
            for doc in documents:
                sources.append(
                    {
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "page_range": doc.metadata.get("page_range", ""),
                        "content_preview": doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content,
                    }
                )

            def answer_stream():
                t_stream_start = time.perf_counter()
                first_token_s = None
                chunk_count = 0
                char_count = 0
                for chunk in response_stream:
                    chunk_count += 1
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    text = self._extract_delta_text(delta)
                    if not text and self.client_type == "azure_inference":
                        text = getattr(delta, "reasoning_content", "") or ""
                    if text:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - t_stream_start
                        char_count += len(text)
                        yield text
                    if self.client_type == "azure_inference" and getattr(chunk, "usage", None):
                        self.logger.info(
                            "LLM stream usage | endpoint=%s | deployment=%s | usage=%s",
                            Config.AZURE_LLM_ENDPOINT,
                            Config.AZURE_LLM_DEPLOYMENT_NAME,
                            chunk.usage,
                        )
                self.logger.info(
                    (
                        "PERF stream_answer | reformulate=%.3fs | retrieve=%.3fs | context=%.3fs | "
                        "stream_open=%.3fs | first_token=%.3fs | stream_total=%.3fs | chunks=%s | chars=%s | total=%.3fs"
                    ),
                    t_reformulate,
                    t_retrieve,
                    t_context,
                    t_stream_open,
                    first_token_s if first_token_s is not None else -1.0,
                    time.perf_counter() - t_stream_start,
                    chunk_count,
                    char_count,
                    time.perf_counter() - t_total_start,
                )

            return {
                "answer_stream": answer_stream(),
                "sources": sources,
                "context": context,
                "query": query,
                "reformulated_query": reformulated_query,
                "perf": {
                    "reformulating_query_s": t_reformulate,
                    "vector_retrieval_s": t_retrieve,
                },
            }
        except Exception as e:
            self.logger.error(
                "LLM stream error | endpoint=%s | deployment=%s | error=%s",
                Config.AZURE_LLM_ENDPOINT,
                Config.AZURE_LLM_DEPLOYMENT_NAME,
                e,
            )
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
