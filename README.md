# Agentic RAG with Retrieval Inspector

Streamlit-based Retrieval-Augmented Generation (RAG) app with persistent ChromaDB storage, Azure OpenAI models, and LangGraph agent nodes.

It supports document upload/indexing, streaming answers, and an inspector workflow that evaluates chunk recall/ranking and answer groundedness.

## Features

- Multi-format ingestion: PDF, DOCX, TXT
- Persistent knowledge base:
  - Uploaded files are stored in `data/documents/`
  - Vector index is stored in `vectorstore/`
- Agentic query flow (LangGraph):
  - Query Agent node (query rewriting + adversarial variants)
  - Main Answering Agent node (retrieval + response streaming)
  - Retrieval Inspector Agent node (chunk ranking/recall + groundedness report)
- Source attribution and retrieved-context display
- Chunk recall and ranking dropdown in UI, including raw similarity scores
- Retrieval diagnostics logs and performance logs
- Clear Documents action removes both local files and vectorstore data

## Current Agentic Flow

1. User asks a question.
2. Query Agent reformulates (when needed) and generates adversarial checks.
3. Main Answering Agent retrieves ranked chunks and streams answer tokens.
4. Retrieval Inspector Agent evaluates:
   - chunk ranking quality
   - recall gaps
   - support of final answer by retrieved chunks
   - groundedness verdict
5. UI displays:
   - answer
   - sources
   - `Chunk Recall & Ranking` dropdown
   - `Retrieval Inspector Agent` dropdown

## Tech Stack

- Python 3.12+
- Streamlit
- LangChain + LangGraph
- ChromaDB
- Azure OpenAI / Azure AI Inference
- PyPDF2, python-docx

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Add Azure environment variables.
4. Run Streamlit.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open: `http://localhost:8501`

## Configuration

Set these in `.env`:

```env
AZURE_LLM_ENDPOINT=https://<your-llm-endpoint>
AZURE_LLM_API_KEY=<your-llm-key>
AZURE_LLM_DEPLOYMENT_NAME=<your-llm-deployment>
AZURE_LLM_API_VERSION=2024-05-01-preview

AZURE_EMBEDDING_ENDPOINT=https://<your-embedding-endpoint>
AZURE_EMBEDDING_API_KEY=<your-embedding-key>
AZURE_EMBEDDING_DEPLOYMENT_NAME=<your-embedding-deployment>
AZURE_EMBEDDING_API_VERSION=2023-05-15
```

Optional behavior defaults are in `config.py`:

- `PAGE_WINDOW`, `PAGE_OVERLAP`
- `DATA_DIR`, `VECTOR_STORE_DIR`
- fallback embedding model: `all-MiniLM-L6-v2`

## Project Structure

```text
RAG-with-Evals/
├── app.py
├── config.py
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── documents/
├── vectorstore/
├── logs/
│   ├── app_perf.log
│   ├── llm_calls.log
│   └── retrieval_perf.log
└── utils/
    ├── Agents.py
    ├── document_processor.py
    ├── rag_chain.py
    ├── vector_store.py
    └── __init__.py
```

## Notes on Retrieval Scores

- `Chunk Recall & Ranking` shows raw similarity scores from vector retrieval.
- Score interpretation can vary by backend/metric.
- Inspector report combines model analysis with deterministic groundedness checks.

## Logs

- `logs/app_perf.log`: UI timing (`QUERY_AGENT`, retrieval, token timings)
- `logs/llm_calls.log`: LLM call-level telemetry
- `logs/retrieval_perf.log`: retrieval breakdown (embedding vs vector search, cache hit)

## Data Management

- Uploaded files persist across restarts in `data/documents/`.
- Chroma vectors persist across restarts in `vectorstore/`.
- `Clear documents` in UI deletes both persisted files and vector index.

## Dependencies

Core dependencies are listed in:

- `requirements.txt`
- `pyproject.toml`

Includes: `langgraph`, `langchain`, `langchain-openai`, `chromadb`, `streamlit`, `azure-ai-inference`, `openai`, and parsing libraries.
