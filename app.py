import streamlit as st
import os
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


def initialize_components():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            persist_directory=Config.VECTOR_STORE_DIR
        )
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = RAGChain(st.session_state.vectorstore)


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
        if st.button("Clear knowledge base", use_container_width=True):
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
        handle_user_message(user_input.strip())


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
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    if not st.session_state.documents_processed:
        assistant_reply = "Please upload and process documents before asking questions."
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.rerun()

    if not user_text:
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain.generate_answer(
                    user_text,
                    chat_history=st.session_state.chat_history,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                context = result.get("context", "")
            except Exception as e:
                answer = f"Error generating answer: {e}"
                sources = []
                context = ""

        st.markdown(answer)
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
    st.rerun()


def process_documents(uploaded_files):
    """Process uploaded documents and add to vector store."""
    try:
        with st.spinner("Indexing documents..."):
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            processor = DocumentProcessor(
                window_pages=getattr(Config, "PAGE_WINDOW", 2),
                overlap_pages=getattr(Config, "PAGE_OVERLAP", 1),
            )
            documents = processor.process_multiple_documents(file_paths)
            if documents:
                st.session_state.vectorstore.add_documents(documents)
                st.session_state.documents_processed = True
                st.success("Documents processed. Ask away!")
            else:
                st.session_state.documents_processed = False
                st.error("No documents were successfully processed.")
    except Exception as e:
        st.session_state.documents_processed = False
        st.error(f"Error processing documents: {str(e)}")


def clear_knowledge_base():
    """Clear the vector database and chat."""
    if st.session_state.vectorstore:
        st.session_state.vectorstore.delete_collection()
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.documents_processed = False
    st.session_state.chat_history = []
    st.success("Knowledge base and chat cleared.")
    st.rerun()


if __name__ == "__main__":
    main()

