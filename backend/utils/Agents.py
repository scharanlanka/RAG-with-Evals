from typing import List, Dict, Any, Callable, TypedDict
from dataclasses import dataclass
import time

from langchain_core.runnables import RunnableLambda

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class AgentState(TypedDict, total=False):
    query: str
    chat_history: List[Dict[str, str]]
    k: int
    reformulated_query: str
    adversarial_queries: List[str]
    documents: List[Document]
    context: str
    sources: List[Dict[str, Any]]
    answer_stream_raw: Any
    answer: str
    retrieval_rankings: List[Dict[str, Any]]
    perf: Dict[str, float]
    retrieval_inspector_report: str


@dataclass
class AgentRunnables:
    query_agent_node: Callable[[AgentState], AgentState]
    answer_agent_node: Callable[[AgentState], AgentState]
    retrieval_inspector_agent_node: Callable[[AgentState], AgentState]
    query_agent: RunnableLambda
    answer_agent: RunnableLambda
    retrieval_inspector_agent: RunnableLambda


def build_agent_runnables(
    *,
    reformulate_query: Callable[[str, List[Dict[str, str]]], str],
    generate_adversarial_queries: Callable[[str], List[str]],
    retrieve_documents: Callable[[str, int], List[Document]],
    retrieve_documents_with_scores: Callable[[str, int], List[Any]],
    format_context: Callable[[List[Document]], str],
    build_stream_response: Callable[[List[Dict[str, str]]], Any],
    inspect_retrieval: Callable[[AgentState], str],
) -> AgentRunnables:
    """Create explicit LangGraph node functions and RunnableLambda wrappers."""

    def query_agent_node(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        reformulated = reformulate_query(query, chat_history)
        adversarial = generate_adversarial_queries(reformulated)
        perf = dict(state.get("perf", {}))
        perf["query_agent_s"] = time.perf_counter() - t0
        return {
            "reformulated_query": reformulated,
            "adversarial_queries": adversarial,
            "perf": perf,
        }

    def answer_agent_node(state: AgentState) -> AgentState:
        rewritten_query = state.get("reformulated_query") or state.get("query", "")
        k = state.get("k", 4)
        perf = dict(state.get("perf", {}))

        t0 = time.perf_counter()
        ranked_docs = retrieve_documents_with_scores(rewritten_query, k)
        documents: List[Document] = [doc for doc, _ in ranked_docs] if ranked_docs else retrieve_documents(rewritten_query, k)
        perf["vector_retrieval_s"] = time.perf_counter() - t0

        if not documents:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context": "",
                "documents": [],
                "perf": perf,
            }

        t0 = time.perf_counter()
        context = format_context(documents)
        perf["context_s"] = time.perf_counter() - t0

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
                    f"Question: {rewritten_query}\n\n"
                    "Answer concisely and cite which source snippets you used."
                ),
            },
        ]

        t0 = time.perf_counter()
        response_stream = build_stream_response(messages)
        perf["stream_open_s"] = time.perf_counter() - t0

        sources: List[Dict[str, Any]] = []
        retrieval_rankings: List[Dict[str, Any]] = []
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
        if ranked_docs:
            for rank, (doc, score) in enumerate(ranked_docs, 1):
                retrieval_rankings.append(
                    {
                        "rank": rank,
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "page_range": doc.metadata.get("page_range", ""),
                        "score": float(score) if score is not None else None,
                        "chunk_text": doc.page_content,
                        "content_preview": doc.page_content[:220] + "..."
                        if len(doc.page_content) > 220
                        else doc.page_content,
                    }
                )
        else:
            for rank, doc in enumerate(documents, 1):
                retrieval_rankings.append(
                    {
                        "rank": rank,
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "page_range": doc.metadata.get("page_range", ""),
                        "score": None,
                        "chunk_text": doc.page_content,
                        "content_preview": doc.page_content[:220] + "..."
                        if len(doc.page_content) > 220
                        else doc.page_content,
                    }
                )

        return {
            "answer_stream_raw": response_stream,
            "sources": sources,
            "context": context,
            "documents": documents,
            "retrieval_rankings": retrieval_rankings,
            "perf": perf,
        }

    def retrieval_inspector_agent_node(state: AgentState) -> AgentState:
        docs = state.get("documents", [])
        if not docs:
            return {
                "retrieval_inspector_report": "No retrieved chunks available for inspection.",
                "retrieval_rankings": [],
            }
        report = inspect_retrieval(state)
        return {
            "retrieval_inspector_report": report,
            "retrieval_rankings": state.get("retrieval_rankings", []),
        }

    return AgentRunnables(
        query_agent_node=query_agent_node,
        answer_agent_node=answer_agent_node,
        retrieval_inspector_agent_node=retrieval_inspector_agent_node,
        query_agent=RunnableLambda(query_agent_node),
        answer_agent=RunnableLambda(answer_agent_node),
        retrieval_inspector_agent=RunnableLambda(retrieval_inspector_agent_node),
    )
