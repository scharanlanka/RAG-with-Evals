import streamlit as st
import os
import shutil
import time
import logging
from pathlib import Path
from typing import List, Dict
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain
from config import Config

# Configure Streamlit page
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

# ---------- Theme-aware Custom CSS ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        color: var(--text-color);
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .subheader {
        text-align: center;
        font-size: 16px;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 16px;
    }
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128,128,128,0.2);
    }
    .chat-bubble {
        padding: 12px 14px;
        border-radius: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(128,128,128,0.15);
    }
    .user-bubble { background: rgba(0, 122, 255, 0.1); }
    .assistant-bubble { background: rgba(0, 200, 83, 0.08); }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role","content","sources","context"}
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _setup_app_logger() -> logging.Logger:
    """Create a file logger for app performance timing."""
    root_dir = Path(__file__).resolve().parent
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "app_perf.log"

    logger = logging.getLogger("app_perf")
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


APP_LOGGER = _setup_app_logger()


def initialize_components():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            persist_directory=Config.VECTOR_STORE_DIR
        )
        st.session_state.vectorstore.load_vectorstore()
    if (
        st.session_state.rag_chain is None
        or not hasattr(st.session_state.rag_chain, "generate_answer_stream")
    ):
        st.session_state.rag_chain = RAGChain(st.session_state.vectorstore)
    st.session_state.documents_processed = has_persisted_knowledge_base()


def has_persisted_documents() -> bool:
    """Return True when local uploaded documents exist on disk."""
    if not os.path.isdir(Config.DATA_DIR):
        return False
    return any(
        not name.startswith(".")
        for name in os.listdir(Config.DATA_DIR)
    )


def has_persisted_embeddings() -> bool:
    """Return True when persisted embeddings exist in Chroma."""
    if st.session_state.vectorstore is None:
        return False
    info = st.session_state.vectorstore.get_collection_info()
    return info.get("count", 0) > 0


def has_persisted_knowledge_base() -> bool:
    """Documents and embeddings are considered ready only when both exist."""
    return has_persisted_documents() and has_persisted_embeddings()


def main():
    st.markdown('<h1 class="main-title">Chat over your documents</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload files on the left, chat on the right — powered by Azure OpenAI.</p>', unsafe_allow_html=True)

    # Validate Azure credentials
    required_settings = {
        "AZURE_LLM_ENDPOINT": Config.AZURE_LLM_ENDPOINT,
        "AZURE_LLM_API_KEY": Config.AZURE_LLM_API_KEY,
        "AZURE_LLM_DEPLOYMENT_NAME": Config.AZURE_LLM_DEPLOYMENT_NAME,
        "AZURE_EMBEDDING_ENDPOINT": Config.AZURE_EMBEDDING_ENDPOINT,
        "AZURE_EMBEDDING_API_KEY": Config.AZURE_EMBEDDING_API_KEY,
        "AZURE_EMBEDDING_DEPLOYMENT_NAME": Config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
    }
    missing = [name for name, value in required_settings.items() if not value]
    if missing:
        st.error(f"Missing Azure configuration: {', '.join(missing)}")
        st.info("Set values in .env or Streamlit secrets, then reload.")
        return

    initialize_components()

    # Sidebar: documents and settings
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="10MB per file limit"
        )
        max_size = 10 * 1024 * 1024
        if uploaded_files:
            uploaded_files = [f for f in uploaded_files if f.size <= max_size]
            if st.button("Process & index", use_container_width=True):
                process_documents(uploaded_files)

        if st.session_state.vectorstore:
            info = st.session_state.vectorstore.get_collection_info()
            st.caption(f"Collection: {info['status']} | Items: {info['count']}")

        st.divider()
        st.subheader("Chat options")
        st.session_state["show_sources"] = st.checkbox("Show sources", value=True)
        st.session_state["show_context"] = st.checkbox("Show context", value=False)

        st.divider()
        if st.button("Clear documents", use_container_width=True):
            clear_knowledge_base()

    # Main chat area
    render_chat_area()


def render_chat_area():
    """Chat UI with conversation-style bubbles."""
    normalize_history()
    chat_placeholder = st.container()
    with chat_placeholder:
        if not st.session_state.chat_history:
            st.info("No messages yet. Upload documents, then ask a question below.")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("role") == "assistant":
                        if st.session_state.get("show_sources") and msg.get("sources"):
                            st.caption("Sources")
                            for i, src in enumerate(msg["sources"], 1):
                                label = f"{i}. {src.get('filename','')}"
                                if src.get("page_range"):
                                    label += f" ({src['page_range']})"
                                st.write(label)
                        if st.session_state.get("show_context") and msg.get("context"):
                            st.caption("Retrieved context")
                            st.text_area("Context", msg["context"], height=160)

    user_input = st.chat_input("Ask about your documents")
    if user_input:
        user_text = user_input.strip()
        if user_text:
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            st.session_state.pending_question = user_text
            st.rerun()

    pending_question = st.session_state.get("pending_question")
    if pending_question:
        handle_user_message(pending_question)


def normalize_history():
    """Ensure chat history items have role/content; upgrade legacy entries."""
    history = st.session_state.get("chat_history", [])
    if not history:
        return
    converted = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            converted.append(item)
            continue
        if isinstance(item, dict):
            # Legacy format: {"question":..., "answer":..., "sources":...}
            question = item.get("question")
            answer = item.get("answer")
            sources = item.get("sources", [])
            context = item.get("context", "")
            if question:
                converted.append({"role": "user", "content": question})
            if answer is not None:
                converted.append({"role": "assistant", "content": answer, "sources": sources, "context": context})
    st.session_state.chat_history = converted


def handle_user_message(user_text: str):
    """Run the RAG pipeline and stream results into chat UI."""
    t_total_start = time.perf_counter()
    if not st.session_state.documents_processed:
        assistant_reply = "Please upload and process documents before asking questions."
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.pending_question = None
        st.rerun()

    if not user_text:
        return

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")
        try:
            t0 = time.perf_counter()
            result = st.session_state.rag_chain.generate_answer_stream(
                user_text,
                chat_history=st.session_state.chat_history,
            )
            t_prepare = time.perf_counter() - t0
            sources = result.get("sources", [])
            context = result.get("context", "")
            perf = result.get("perf", {}) if isinstance(result, dict) else {}
            if result.get("answer_stream") is not None:
                thinking_placeholder.empty()
                stream_start = time.perf_counter()
                first_token_s = None
                last_token_s = None

                def ui_stream_wrapper():
                    nonlocal first_token_s, last_token_s
                    for token in result["answer_stream"]:
                        now = time.perf_counter()
                        if first_token_s is None:
                            first_token_s = now - stream_start
                        last_token_s = now - stream_start
                        yield token

                answer = st.write_stream(ui_stream_wrapper())
                if not isinstance(answer, str):
                    answer = str(answer)
                APP_LOGGER.info(
                    (
                        "PERF ui_stream | REFORMULATING_QUERY=%.3fs | VECTOR_RETRIEVAL=%.3fs | "
                        "FIRST_TOKEN=%.3fs | LAST_TOKEN=%.3fs"
                    ),
                    perf.get("reformulating_query_s", -1.0),
                    perf.get("vector_retrieval_s", -1.0),
                    first_token_s if first_token_s is not None else -1.0,
                    last_token_s if last_token_s is not None else -1.0,
                )
            else:
                answer = result.get("answer", "")
                thinking_placeholder.empty()
                st.markdown(answer)
                APP_LOGGER.info(
                    "PERF ui_non_stream | prepare=%.3fs | total=%.3fs | chars=%s | sources=%s",
                    t_prepare,
                    time.perf_counter() - t_total_start,
                    len(answer),
                    len(sources),
                )
        except Exception as e:
            thinking_placeholder.empty()
            answer = f"Error generating answer: {e}"
            sources = []
            context = ""
            st.markdown(answer)
            APP_LOGGER.exception("PERF ui_error | total=%.3fs", time.perf_counter() - t_total_start)

        if st.session_state.get("show_sources") and sources:
            st.caption("Sources")
            for i, src in enumerate(sources, 1):
                label = f"{i}. {src.get('filename','')}"
                if src.get("page_range"):
                    label += f" ({src['page_range']})"
                st.write(label)
        if st.session_state.get("show_context") and context:
            st.caption("Retrieved context")
            st.text_area("Context", context, height=160)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "context": context,
        }
    )
    st.session_state.pending_question = None
    st.rerun()


def process_documents(uploaded_files):
    """Process uploaded documents and add to vector store."""
    t_total_start = time.perf_counter()
    try:
        with st.spinner("Indexing documents..."):
            t0 = time.perf_counter()
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            t_save_files = time.perf_counter() - t0

            t0 = time.perf_counter()
            processor = DocumentProcessor(
                window_pages=getattr(Config, "PAGE_WINDOW", 2),
                overlap_pages=getattr(Config, "PAGE_OVERLAP", 1),
            )
            documents = processor.process_multiple_documents(file_paths)
            t_process_docs = time.perf_counter() - t0
            if documents:
                t0 = time.perf_counter()
                st.session_state.vectorstore.add_documents(documents)
                t_index = time.perf_counter() - t0
                st.session_state.documents_processed = True
                st.success("Documents processed. Ask away!")
                APP_LOGGER.info(
                    "PERF ingest | files=%s | chunks=%s | save_files=%.3fs | process_docs=%.3fs | index=%.3fs | total=%.3fs",
                    len(file_paths),
                    len(documents),
                    t_save_files,
                    t_process_docs,
                    t_index,
                    time.perf_counter() - t_total_start,
                )
            else:
                st.session_state.documents_processed = False
                st.error("No documents were successfully processed.")
                APP_LOGGER.info(
                    "PERF ingest_no_chunks | files=%s | save_files=%.3fs | process_docs=%.3fs | total=%.3fs",
                    len(file_paths),
                    t_save_files,
                    t_process_docs,
                    time.perf_counter() - t_total_start,
                )
    except Exception as e:
        st.session_state.documents_processed = False
        st.error(f"Error processing documents: {str(e)}")
        APP_LOGGER.exception("PERF ingest_error | total=%.3fs", time.perf_counter() - t_total_start)


def clear_knowledge_base():
    """Clear local documents, embeddings, and chat history."""
    if st.session_state.vectorstore:
        st.session_state.vectorstore.delete_collection()
    shutil.rmtree(Config.DATA_DIR, ignore_errors=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.documents_processed = False
    st.session_state.chat_history = []
    st.success("Documents, embeddings, and chat cleared.")
    st.rerun()


if __name__ == "__main__":
    main()
