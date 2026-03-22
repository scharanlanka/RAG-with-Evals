from typing import List, Dict, Any, Optional
import logging
import time
import re
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

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = None
    HAS_LANGGRAPH = False

from backend.utils.vector_store import VectorStore
from backend.utils.Agents import AgentState, build_agent_runnables
from backend.config import Config


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        endpoint = (Config.AZURE_LLM_ENDPOINT or "").lower()
        self.use_foundry_models = "services.ai.azure.com" in endpoint
        self.client_type = "azure_openai"
        self.client = self._setup_client()
        self.logger = self._setup_logger()
        self.agents = build_agent_runnables(
            reformulate_query=self._reformulate_query,
            generate_adversarial_queries=self._generate_adversarial_queries,
            retrieve_documents=self.retrieve_documents,
            retrieve_documents_with_scores=self.retrieve_documents_with_scores,
            format_context=self.format_context,
            build_stream_response=self._build_stream_response,
            inspect_retrieval=self._inspect_retrieval,
        )
        self.answer_graph = self._build_answer_graph()
        self.inspector_graph = self._build_inspector_graph()

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

    def _setup_logger(self) -> logging.Logger:
        """Create a file logger for LLM calls (workspace root/logs)."""
        root_dir = Path(__file__).resolve().parents[1]
        log_dir = root_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "llm_calls.log"

        logger = logging.getLogger("llm_calls")
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

    def _build_answer_graph(self):
        """Build LangGraph for Query Agent -> Main Answering Agent."""
        if not HAS_LANGGRAPH:
            return None
        graph = StateGraph(AgentState)
        graph.add_node("query_agent", self.agents.query_agent)
        graph.add_node("answer_agent", self.agents.answer_agent)
        graph.set_entry_point("query_agent")
        graph.add_edge("query_agent", "answer_agent")
        graph.add_edge("answer_agent", END)
        return graph.compile()

    def _build_inspector_graph(self):
        """Build LangGraph for Retrieval Inspector Agent."""
        if not HAS_LANGGRAPH:
            return None
        graph = StateGraph(AgentState)
        graph.add_node("retrieval_inspector_agent", self.agents.retrieval_inspector_agent)
        graph.set_entry_point("retrieval_inspector_agent")
        graph.add_edge("retrieval_inspector_agent", END)
        return graph.compile()

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3] + "..."

    def _needs_reformulation(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        history = chat_history or []
        has_assistant_turn = any(m.get("role") == "assistant" for m in history)
        if not has_assistant_turn:
            return False
        normalized = (query or "").strip().lower()
        if not normalized:
            return False
        context_tokens = (
            "it", "they", "them", "that", "those", "these", "this", "he", "she", "there",
            "earlier", "previous", "above", "same", "again", "also", "too", "former", "latter",
        )
        context_starters = (
            "and ", "so ", "then ", "what about", "how about", "can you elaborate",
            "explain more", "tell me more",
        )
        if any(normalized.startswith(prefix) for prefix in context_starters):
            return True
        return any(f" {token} " in f" {normalized} " for token in context_tokens)

    def _reformulate_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Query Agent core reformulation behavior."""
        if not self._needs_reformulation(query, chat_history):
            return query

        history = chat_history or []
        last_turns = [m for m in history if m.get("role") in {"user", "assistant"}][-4:]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query reformulation assistant. Rewrite the latest user question into a clear "
                    "standalone question using recent conversation context."
                ),
            }
        ]
        for m in last_turns:
            role = m.get("role", "user")
            max_chars = 240 if role == "assistant" else 180
            content = self._truncate_text(str(m.get("content", "")), max_chars)
            if content:
                messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "system",
                "content": "Return only the reformulated question text.",
            }
        )

        try:
            if self.client_type == "azure_inference":
                resp = self.client.complete(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=[self._to_inference_message(m["role"], m["content"]) for m in messages],
                    temperature=0.2,
                    max_tokens=48,
                )
                reformulated = self._extract_message_text(resp.choices[0].message).strip()
            else:
                resp = self.client.chat.completions.create(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=48,
                )
                reformulated = self._extract_message_text(resp.choices[0].message).strip()
            self.logger.info(
                "Query agent reformulated | original=%s | reformulated=%s",
                query.replace("\n", " ")[:200],
                reformulated[:200],
            )
            return reformulated or query
        except Exception as e:
            self.logger.error("Query agent reformulation failed, using original query | error=%s", e)
            return query

    def _generate_adversarial_queries(self, reformulated_query: str) -> List[str]:
        """Generate deterministic edge/adversarial variants for inspection workflows."""
        q = reformulated_query.strip()
        if not q:
            return []
        lower = q.rstrip("?")
        return [
            f"{lower}? Include contradictions if present.",
            f"What evidence disproves: {lower}?",
            f"{lower}? Answer only if exact wording exists.",
        ]

    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_documents_with_scores(self, query: str, k: int = 4) -> List[Any]:
        """Retrieve ranked chunks with raw similarity scores when available."""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception:
            return []

    def format_context(self, documents: List[Document]) -> str:
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

    def _build_stream_response(self, messages: List[Dict[str, str]]):
        """Open the model stream used by the main answering agent."""
        self.logger.info(
            "LLM stream request | endpoint=%s | deployment=%s",
            Config.AZURE_LLM_ENDPOINT,
            Config.AZURE_LLM_DEPLOYMENT_NAME,
        )
        return self.client.chat.completions.create(
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

    def _inspect_retrieval(self, state: AgentState) -> str:
        rankings = state.get("retrieval_rankings", [])
        lines = []
        for item in rankings:
            score = item.get("score")
            score_txt = f"{score:.6f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(
                f"{item.get('rank', '?')}. score={score_txt} | {item.get('filename', 'Unknown')} "
                f"{item.get('page_range', '')} :: {self._truncate_text(item.get('content_preview', ''), 180)}"
            )
        chunk_block = "\n".join(lines)
        answer_text = self._truncate_text(state.get("answer", ""), 1200)

        prompt = (
            "You are the Retrieval Inspector Agent.\n"
            "Given the final answer and retrieved chunk ranking/scores, provide:\n"
            "1) Ranking quality assessment\n"
            "2) Recall gaps / missing evidence\n"
            "3) Potential chunking issues\n"
            "4) Whether the answer is well-supported by top-ranked chunks\n"
            "5) Recommended next retrieval tweak\n\n"
            f"Original query: {state.get('query', '')}\n"
            f"Rewritten query: {state.get('reformulated_query', '')}\n"
            f"Adversarial checks: {state.get('adversarial_queries', [])}\n"
            f"Final answer:\n{answer_text}\n\n"
            "Note: score is raw vectorstore score (backend-dependent).\n"
            f"Retrieved chunks in rank order:\n{chunk_block}\n"
        )

        messages = [
            {"role": "system", "content": "Respond in concise markdown with short bullet points."},
            {"role": "user", "content": prompt},
        ]

        deterministic_report = self._build_groundedness_report(state)
        try:
            if self.client_type == "azure_inference":
                response = self.client.complete(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=[self._to_inference_message(m["role"], m["content"]) for m in messages],
                    temperature=0.1,
                    max_tokens=260,
                )
                report = self._extract_message_text(response.choices[0].message).strip()
            else:
                response = self.client.chat.completions.create(
                    model=Config.AZURE_LLM_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=260,
                )
                report = self._extract_message_text(response.choices[0].message).strip()
            if report:
                return f"{report}\n\n---\n\n{deterministic_report}"
            return deterministic_report
        except Exception as e:
            self.logger.error("Retrieval inspector failed | error=%s", e)
            return f"{deterministic_report}\n\n_Inspector model error: {e}_"

    def _build_groundedness_report(self, state: AgentState) -> str:
        """Deterministic fallback: groundedness/relevance from answer-chunk lexical overlap."""
        answer = state.get("answer", "") or ""
        rankings = state.get("retrieval_rankings", []) or []
        if not answer.strip():
            return "### Groundedness\n- No answer text available for groundedness check."
        if not rankings:
            return "### Groundedness\n- No retrieved chunks available for groundedness check."

        answer_tokens = self._normalize_tokens(answer)
        if not answer_tokens:
            return "### Groundedness\n- Answer has too little lexical content to evaluate."

        rows: List[Dict[str, Any]] = []
        union_overlap = set()
        weighted_overlap_sum = 0.0
        weight_sum = 0.0

        for item in rankings:
            rank = int(item.get("rank", 1) or 1)
            chunk_text = item.get("chunk_text") or item.get("content_preview") or ""
            chunk_tokens = self._normalize_tokens(chunk_text)
            overlap = answer_tokens.intersection(chunk_tokens)
            overlap_ratio = len(overlap) / max(len(answer_tokens), 1)
            union_overlap.update(overlap)
            weight = 1.0 / rank
            weighted_overlap_sum += overlap_ratio * weight
            weight_sum += weight
            rows.append(
                {
                    "rank": rank,
                    "filename": item.get("filename", "Unknown"),
                    "page_range": item.get("page_range", ""),
                    "score": item.get("score"),
                    "overlap_ratio": overlap_ratio,
                }
            )

        coverage = len(union_overlap) / max(len(answer_tokens), 1)
        weighted_support = weighted_overlap_sum / max(weight_sum, 1e-9)
        rows_sorted = sorted(rows, key=lambda r: r["overlap_ratio"], reverse=True)
        best = rows_sorted[0] if rows_sorted else None

        if coverage >= 0.45 and weighted_support >= 0.22:
            verdict = "Strongly grounded"
        elif coverage >= 0.25 and weighted_support >= 0.12:
            verdict = "Partially grounded"
        else:
            verdict = "Weakly grounded"

        details = []
        for row in sorted(rows, key=lambda r: r["rank"])[:6]:
            score = row["score"]
            score_txt = f"{score:.6f}" if isinstance(score, (int, float)) else "n/a"
            page_range = f" ({row['page_range']})" if row.get("page_range") else ""
            details.append(
                f"- Rank {row['rank']}: overlap={row['overlap_ratio']:.1%} | score={score_txt} | "
                f"{row['filename']}{page_range}"
            )

        best_txt = "N/A"
        if best:
            best_page = f" ({best['page_range']})" if best.get("page_range") else ""
            best_txt = f"Rank {best['rank']} - {best['filename']}{best_page} ({best['overlap_ratio']:.1%} overlap)"

        top_rank_support_txt = f"{weighted_support:.1%}"
        if verdict in {"Partially grounded", "Weakly grounded"}:
            top_rank_support_txt = "N/A"
            best_txt = "N/A"

        recommendation = (
            "Increase `k` or adjust chunking/page window for better recall."
            if verdict != "Strongly grounded"
            else "Current retrieval quality is acceptable; keep ranking configuration."
        )

        return (
            "### Groundedness\n"
            f"- Verdict: **{verdict}**\n"
            f"- Answer token coverage by retrieved chunks: **{coverage:.1%}**\n"
            f"- Weighted top-rank support: **{top_rank_support_txt}**\n"
            f"- Best supporting chunk: **{best_txt}**\n"
            f"- Recommendation: {recommendation}\n\n"
            "### Chunk Relevance\n"
            + "\n".join(details)
        )

    def _normalize_tokens(self, text: str) -> set:
        """Normalize text into a lightweight keyword set for overlap scoring."""
        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
            "is", "are", "was", "were", "be", "this", "that", "it", "as", "from", "you", "your",
            "can", "could", "should", "would", "about", "into", "only", "not", "if", "then",
        }
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        return {tok for tok in tokens if tok not in stop_words}

    def run_retrieval_inspector(
        self,
        *,
        query: str,
        reformulated_query: str,
        adversarial_queries: List[str],
        documents: List[Document],
        retrieval_rankings: List[Dict[str, Any]],
        answer: str,
    ) -> AgentState:
        """Run inspector after answer is available so it can verify grounding quality."""
        return self._run_inspector_graph(
            {
                "query": query,
                "reformulated_query": reformulated_query,
                "adversarial_queries": adversarial_queries,
                "documents": documents,
                "retrieval_rankings": retrieval_rankings,
                "answer": answer,
            }
        )

    def _run_answer_graph(self, state: AgentState) -> AgentState:
        if self.answer_graph is None:
            step1 = self.agents.query_agent_node(state)
            merged = dict(state)
            merged.update(step1)
            step2 = self.agents.answer_agent_node(merged)
            merged.update(step2)
            return merged
        return self.answer_graph.invoke(state)

    def _run_inspector_graph(self, state: AgentState) -> AgentState:
        if self.inspector_graph is None:
            merged = dict(state)
            merged.update(self.agents.retrieval_inspector_agent_node(state))
            return merged
        return self.inspector_graph.invoke(state)

    def generate_answer_stream(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 4
    ) -> Dict[str, Any]:
        """Agentic flow: Query Agent -> Main Answering Agent (+ async Retrieval Inspector Agent)."""
        t_total_start = time.perf_counter()
        try:
            state = self._run_answer_graph(
                {
                    "query": query,
                    "chat_history": chat_history or [],
                    "k": k,
                    "perf": {},
                }
            )
            perf = dict(state.get("perf", {}))
            reformulated_query = state.get("reformulated_query", query)
            adversarial_queries = state.get("adversarial_queries", [])
            documents = state.get("documents", [])
            sources = state.get("sources", [])
            context = state.get("context", "")

            if state.get("answer") is not None:
                self.logger.info(
                    "PERF stream_no_docs | query_agent=%.3fs | retrieve=%.3fs | total=%.3fs",
                    perf.get("query_agent_s", 0.0),
                    perf.get("vector_retrieval_s", 0.0),
                    time.perf_counter() - t_total_start,
                )
                return {
                    "answer": state.get("answer", ""),
                    "sources": [],
                    "context": "",
                    "query": query,
                    "reformulated_query": reformulated_query,
                    "adversarial_queries": adversarial_queries,
                    "retrieval_inspector_report": "No retrieved chunks available for inspection.",
                    "perf": {
                        "query_agent_s": perf.get("query_agent_s", 0.0),
                        "vector_retrieval_s": perf.get("vector_retrieval_s", 0.0),
                    },
                }

            response_stream = state.get("answer_stream_raw")
            if response_stream is None:
                return {
                    "answer": "An error occurred while generating the answer.",
                    "sources": [],
                    "context": "",
                    "query": query,
                    "reformulated_query": reformulated_query,
                    "adversarial_queries": adversarial_queries,
                }

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
                self.logger.info(
                    (
                        "PERF stream_answer | query_agent=%.3fs | retrieve=%.3fs | context=%.3fs | "
                        "stream_open=%.3fs | first_token=%.3fs | stream_total=%.3fs | chunks=%s | chars=%s | total=%.3fs"
                    ),
                    perf.get("query_agent_s", 0.0),
                    perf.get("vector_retrieval_s", 0.0),
                    perf.get("context_s", 0.0),
                    perf.get("stream_open_s", 0.0),
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
                "adversarial_queries": adversarial_queries,
                "documents": documents,
                "retrieval_rankings": state.get("retrieval_rankings", []),
                "perf": {
                    "query_agent_s": perf.get("query_agent_s", 0.0),
                    "vector_retrieval_s": perf.get("vector_retrieval_s", 0.0),
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

    def generate_answer(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 4) -> Dict[str, Any]:
        """Non-stream helper that consumes the stream output for compatibility."""
        result = self.generate_answer_stream(query, chat_history=chat_history, k=k)
        if result.get("answer") is not None:
            return result
        answer_text = "".join(token for token in result.get("answer_stream", []))
        inspector_report = ""
        inspector_state = self.run_retrieval_inspector(
            query=query,
            reformulated_query=result.get("reformulated_query", query),
            adversarial_queries=result.get("adversarial_queries", []),
            documents=result.get("documents", []),
            retrieval_rankings=result.get("retrieval_rankings", []),
            answer=answer_text,
        )
        inspector_report = inspector_state.get("retrieval_inspector_report", "")
        return {
            "answer": answer_text,
            "sources": result.get("sources", []),
            "context": result.get("context", ""),
            "query": query,
            "reformulated_query": result.get("reformulated_query"),
            "adversarial_queries": result.get("adversarial_queries", []),
            "retrieval_rankings": inspector_state.get("retrieval_rankings", result.get("retrieval_rankings", [])),
            "retrieval_inspector_report": inspector_report,
        }

    def chat_with_context(self, query: str, chat_history: List[Dict] = None, k: int = 4) -> Dict[str, Any]:
        return self.generate_answer(query, chat_history=chat_history, k=k)
