# RAG with Retrieval Evals

A production-style Retrieval-Augmented Generation (RAG) system focused on **answer quality, grounding, and retrieval transparency**.

This project combines a FastAPI backend, a Next.js frontend, and a multi-agent RAG workflow to show not only answers, but also the retrieval evidence and diagnostics behind them.

## Highlights

- Multi-agent orchestration for query rewriting, answering, and retrieval inspection
- Streaming chat experience with source-backed responses
- Groundedness metrics: token coverage, support, best supporting chunk
- Retrieval diagnostics: cosine similarity, overlap, and relevance labels
- Upload-time chunking control:
  - Page chunking (window + overlap)
  - Recursive character splitter
  - Token splitter
- Persistent vector store using ChromaDB

## Architecture

```text
Frontend (Next.js)
  -> Upload docs, select chunking strategy, ask questions
  -> Streams answers + diagnostics

Backend (FastAPI)
  -> /documents/upload (ingestion + chunking)
  -> /chat/stream (SSE)
  -> LangGraph agents + Chroma retrieval

Storage
  -> backend/data/documents
  -> backend/vectorstore
```

## Tech Stack

- Python, FastAPI, LangGraph, LangChain
- Azure OpenAI (LLM + embeddings)
- ChromaDB
- Next.js, React, TypeScript

## Monorepo Layout

```text
RAG-with-Evals/
  backend/    # API, ingestion, chunking, retrieval, agents
  frontend/   # Chat UI, retrieval inspector, upload UX
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/scharanlanka/RAG-with-Evals.git
cd RAG-with-Evals
```

### 2. Configure environment

Create a `.env` at the repo root (or `backend/.env`):

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

### 3. Run backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8002
```

### 4. Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3002` and expects backend at `http://localhost:8002`.

## Engineering Focus

This repository demonstrates practical GenAI engineering across:

- RAG architecture and retrieval quality design
- Agent-based orchestration and streaming systems
- Groundedness evaluation and explainability UX
- End-to-end full-stack implementation (API + UI + observability)

## Additional Docs

- Backend setup and API details: [`backend/README.md`](backend/README.md)
- Frontend setup and UI details: [`frontend/README.md`](frontend/README.md)
