# Backend - Agentic RAG API

FastAPI backend powering ingestion, retrieval, multi-agent answering, and retrieval diagnostics.

## Features

- Document ingestion for `pdf`, `docx`, `txt`
- Configurable chunking at upload time:
  - `page` (window + overlap)
  - `recursive_char`
  - `recursive_token`
- Persistent vector retrieval with ChromaDB
- Multi-agent workflow:
  - Query reformulation
  - Answer generation (streaming)
  - Retrieval inspection and groundedness report
- SSE endpoint for token streaming

## Stack

- Python 3.12+
- FastAPI
- LangChain + LangGraph
- Azure OpenAI
- ChromaDB

## Project Structure

```text
backend/
  main.py                 # FastAPI app and routes
  config.py               # env/config loading
  utils/
    document_processor.py # file parsing + chunking
    vector_store.py       # Chroma operations
    rag_chain.py          # retrieval + generation pipeline
    Agents.py             # agent node definitions
  data/documents/         # uploaded documents
  vectorstore/            # persisted vector DB
```

## Prerequisites

- Azure OpenAI credentials
- Python environment with dependencies installed

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternative (if using `uv` and `pyproject.toml`):

```bash
uv sync
```

## Environment Variables

Create `.env` in repo root or backend directory:

```env
AZURE_LLM_ENDPOINT=https://<your-llm-endpoint>
AZURE_LLM_API_KEY=<your-llm-key>
AZURE_LLM_DEPLOYMENT_NAME=<your-llm-deployment>
AZURE_LLM_API_VERSION=2024-05-01-preview

AZURE_EMBEDDING_ENDPOINT=https://<your-embedding-endpoint>
AZURE_EMBEDDING_API_KEY=<your-embedding-key>
AZURE_EMBEDDING_DEPLOYMENT_NAME=<your-embedding-deployment>
AZURE_EMBEDDING_API_VERSION=2023-05-15

CHROMA_DISTANCE_METRIC=cosine
```

## Run

From repository root:

```bash
uvicorn backend.main:app --reload --port 8002
```

Health check:

```bash
curl http://localhost:8002/health
```

## API Endpoints

- `GET /health`
- `GET /documents`
- `POST /documents/upload` (multipart)
- `POST /chat`
- `POST /chat/stream` (SSE)
- `DELETE /knowledge-base`

## Upload Chunking Parameters

`POST /documents/upload` accepts:

- `files` (one or more)
- `chunking_mode`: `page` | `recursive_char` | `recursive_token`
- `page_window`, `page_overlap` (for `page`)
- `split_chunk_size`, `split_chunk_overlap` (for recursive modes)

## Operational Notes

- If you change chunking strategy or distance metric, clear and re-index the knowledge base for consistent retrieval behavior.
- Persisted vector data is stored in `backend/vectorstore`.
