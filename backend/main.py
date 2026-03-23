import os
import shutil
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from backend.config import Config
from backend.utils.document_processor import DocumentProcessor
from backend.utils.rag_chain import RAGChain
from backend.utils.vector_store import VectorStore


app = FastAPI(title="Agentic RAG Backend", version="0.1.0")

default_origins = [
    "http://localhost:3002",
    "http://127.0.0.1:3002",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
env_origins = os.getenv("CORS_ORIGINS", "")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()] or default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_vectorstore: Optional[VectorStore] = None
_rag_chain: Optional[RAGChain] = None

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context: str
    query: str
    reformulated_query: Optional[str] = None
    adversarial_queries: List[str] = Field(default_factory=list)
    retrieval_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_inspector_report: str = ""


def _sse_event(data: Dict[str, Any], event: Optional[str] = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def _ensure_dirs() -> None:
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)


def _get_services() -> tuple[VectorStore, RAGChain]:
    global _vectorstore, _rag_chain
    _ensure_dirs()
    if _vectorstore is None:
        _vectorstore = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            persist_directory=Config.VECTOR_STORE_DIR,
        )
        _vectorstore.load_vectorstore()
    if _rag_chain is None:
        _rag_chain = RAGChain(_vectorstore)
    return _vectorstore, _rag_chain


def _list_documents() -> List[Dict[str, Any]]:
    if not os.path.isdir(Config.DATA_DIR):
        return []
    files = []
    for name in sorted(os.listdir(Config.DATA_DIR)):
        path = os.path.join(Config.DATA_DIR, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        files.append(
            {"name": name, "path": path, "size_bytes": os.path.getsize(path)}
        )
    return files


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/documents")
def list_documents() -> Dict[str, Any]:
    vectorstore, _ = _get_services()
    return {
        "files": _list_documents(),
        "collection": vectorstore.get_collection_info(),
        "indexed_filenames": vectorstore.get_indexed_filenames(),
    }


@app.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunking_mode: str = Form("page"),
    page_window: int = Form(Config.PAGE_WINDOW),
    page_overlap: int = Form(Config.PAGE_OVERLAP),
    split_chunk_size: int = Form(900),
    split_chunk_overlap: int = Form(120),
) -> Dict[str, Any]:
    vectorstore, _ = _get_services()
    mode = (chunking_mode or "page").strip().lower()
    valid_modes = {"page", "recursive_char", "recursive_token"}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chunking_mode. Use one of: page, recursive_char, recursive_token."
            ),
        )
    if page_window < 1:
        raise HTTPException(status_code=400, detail="page_window must be >= 1.")
    if page_overlap < 0:
        raise HTTPException(status_code=400, detail="page_overlap must be >= 0.")
    if split_chunk_size < 50:
        raise HTTPException(status_code=400, detail="split_chunk_size must be >= 50.")
    if split_chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="split_chunk_overlap must be >= 0.")
    if split_chunk_overlap >= split_chunk_size:
        raise HTTPException(
            status_code=400,
            detail="split_chunk_overlap must be smaller than split_chunk_size.",
        )

    saved_paths: List[str] = []
    for upload in files:
        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {upload.filename}. Allowed: pdf, docx, txt.",
            )
        data = await upload.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} exceeds 10MB limit.",
            )
        safe_name = os.path.basename(upload.filename or "")
        file_path = os.path.join(Config.DATA_DIR, safe_name)
        with open(file_path, "wb") as f:
            f.write(data)
        saved_paths.append(file_path)

    processor = DocumentProcessor(
        chunking_mode=mode,
        window_pages=page_window,
        overlap_pages=page_overlap,
        split_chunk_size=split_chunk_size,
        split_chunk_overlap=split_chunk_overlap,
    )
    documents = processor.process_multiple_documents(saved_paths)
    if not documents:
        raise HTTPException(status_code=400, detail="No valid text extracted from uploaded files.")
    vectorstore.add_documents(documents)
    return {
        "message": "Documents indexed successfully.",
        "files_saved": len(saved_paths),
        "chunks_indexed": len(documents),
        "chunking_mode": mode,
        "page_window": processor.window_pages if mode == "page" else None,
        "page_overlap": processor.overlap_pages if mode == "page" else None,
        "split_chunk_size": split_chunk_size if mode != "page" else None,
        "split_chunk_overlap": split_chunk_overlap if mode != "page" else None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    vectorstore, rag_chain = _get_services()
    collection = vectorstore.get_collection_info()
    if collection.get("count", 0) == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload documents first.")

    result = rag_chain.generate_answer(
        request.query,
        chat_history=request.chat_history,
        k=request.k,
    )
    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        context=result.get("context", ""),
        query=result.get("query", request.query),
        reformulated_query=result.get("reformulated_query"),
        adversarial_queries=result.get("adversarial_queries", []),
        retrieval_rankings=result.get("retrieval_rankings", []),
        retrieval_inspector_report=result.get("retrieval_inspector_report", ""),
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    vectorstore, rag_chain = _get_services()
    collection = vectorstore.get_collection_info()
    if collection.get("count", 0) == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload documents first.")

    result = rag_chain.generate_answer_stream(
        request.query,
        chat_history=request.chat_history,
        k=request.k,
    )

    if result.get("answer") is not None:
        message = result.get("answer", "")

        def immediate_error_stream():
            yield _sse_event({"query": request.query}, event="start")
            yield _sse_event({"token": message}, event="token")
            yield _sse_event(
                {
                    "answer": message,
                    "sources": result.get("sources", []),
                    "context": result.get("context", ""),
                    "query": result.get("query", request.query),
                    "reformulated_query": result.get("reformulated_query"),
                    "adversarial_queries": result.get("adversarial_queries", []),
                    "retrieval_rankings": result.get("retrieval_rankings", []),
                    "retrieval_inspector_report": result.get("retrieval_inspector_report", ""),
                },
                event="done",
            )

        return StreamingResponse(immediate_error_stream(), media_type="text/event-stream")

    def token_stream():
        try:
            yield _sse_event({"query": request.query}, event="start")
            answer_parts: List[str] = []
            for token in result.get("answer_stream", []):
                answer_parts.append(token)
                yield _sse_event({"token": token}, event="token")

            answer_text = "".join(answer_parts)
            inspector_state = rag_chain.run_retrieval_inspector(
                query=request.query,
                reformulated_query=result.get("reformulated_query", request.query),
                adversarial_queries=result.get("adversarial_queries", []),
                documents=result.get("documents", []),
                retrieval_rankings=result.get("retrieval_rankings", []),
                answer=answer_text,
            )
            yield _sse_event(
                {
                    "answer": answer_text,
                    "sources": result.get("sources", []),
                    "context": result.get("context", ""),
                    "query": result.get("query", request.query),
                    "reformulated_query": result.get("reformulated_query"),
                    "adversarial_queries": result.get("adversarial_queries", []),
                    "retrieval_rankings": inspector_state.get(
                        "retrieval_rankings", result.get("retrieval_rankings", [])
                    ),
                    "retrieval_inspector_report": inspector_state.get("retrieval_inspector_report", ""),
                },
                event="done",
            )
        except Exception as exc:
            yield _sse_event({"error": str(exc)}, event="error")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(token_stream(), media_type="text/event-stream", headers=headers)


@app.delete("/knowledge-base")
def clear_knowledge_base() -> Dict[str, str]:
    global _vectorstore, _rag_chain
    _ensure_dirs()
    if _vectorstore is not None:
        _vectorstore.clear_persisted_store()
    else:
        shutil.rmtree(Config.VECTOR_STORE_DIR, ignore_errors=True)
        os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)

    shutil.rmtree(Config.DATA_DIR, ignore_errors=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    _vectorstore = None
    _rag_chain = None
    return {"message": "Knowledge base cleared."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8002, reload=True)
