# Agentic RAG with Retrieval Inspector

Enterprise-style multi-agent RAG system built with **LangGraph, Azure OpenAI, and ChromaDB**, featuring groundedness verification, retrieval diagnostics, and full observability into LLM outputs.

This project focuses on **GenAI reliability engineering** — ensuring answers are supported by retrieved context and providing transparency into retrieval quality.

---

## Key Capabilities

**Multi-Agent Workflow (LangGraph)**

* Query Agent: query rewriting and robustness checks
* Answering Agent: semantic retrieval and streaming responses
* Retrieval Inspector Agent: evaluates groundedness, recall, and ranking quality

**Groundedness and Retrieval Evaluation**

* Token-level answer support analysis
* Chunk ranking and similarity score inspection
* Best supporting chunk identification
* Groundedness verdict and retrieval recommendations

**Persistent Knowledge Base**

* Multi-format ingestion: PDF, DOCX, TXT
* Persistent ChromaDB vector store
* Incremental indexing and retrieval

**Observability and Diagnostics**

* Retrieval performance logs
* LLM telemetry tracking
* Retrieval recall and ranking analysis

---

## Architecture

```
User Query
   ↓
Query Agent (rewrite & validate)
   ↓
Vector Retrieval (ChromaDB)
   ↓
Answering Agent (generate response)
   ↓
Retrieval Inspector Agent (groundedness & diagnostics)
```

---

## Tech Stack

* Python
* LangGraph, LangChain
* Azure OpenAI
* ChromaDB
* FastAPI (backend API)
* Next.js (frontend - planned)

---

## Example Output

Provides:

* Answer with source attribution
* Groundedness verdict (e.g., Strongly grounded)
* Token coverage and ranking analysis
* Retrieval diagnostics and recommendations

---

## Setup and Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/scharanlanka/RAG-with-Evals.git
cd RAG-with-Evals
```

---

### 2. Create and Activate Virtual Environment

**Mac/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

Install all required packages:

```bash
pip install -r backend/requirements.txt
```

This installs core libraries including:

* LangGraph
* LangChain
* Azure OpenAI SDK
* ChromaDB
* Streamlit
* Document parsers

---

### 4. Configure Azure OpenAI Credentials

Create a `.env` file in the project root:

```env
AZURE_LLM_ENDPOINT=https://your-llm-endpoint.openai.azure.com/
AZURE_LLM_API_KEY=your_llm_api_key
AZURE_LLM_DEPLOYMENT_NAME=your_llm_deployment
AZURE_LLM_API_VERSION=2024-05-01-preview

AZURE_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
AZURE_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment
AZURE_EMBEDDING_API_VERSION=2023-05-15
CHROMA_DISTANCE_METRIC=cosine
```

These credentials enable:

* LLM inference
* Embedding generation
* Retrieval pipeline

---

### 5. Run the Backend API

Start the FastAPI server from the repo root:

```bash
uvicorn backend.main:app --reload --port 8002
```

---

### 6. API Endpoints

Use:

* `GET /health`
* `GET /documents`
* `POST /documents/upload`
* `POST /chat`
* `POST /chat/stream` (SSE)
* `DELETE /knowledge-base`

---

### 7. Current Scope

Backend APIs are ready. Frontend implementation will be added in `frontend/` next.

---

### 8. Optional: Clear Indexed Data

Use the **"Clear Documents"** button in the UI to remove:

* Uploaded files
* Vector database entries

This resets the knowledge base.

---

### Troubleshooting

If environment variables are not detected:

```bash
pip install python-dotenv
```

Ensure `.env` is in the project root directory.

If you change `CHROMA_DISTANCE_METRIC` (for example from default `l2` to `cosine`), clear the existing `backend/vectorstore` data and re-index documents so Chroma rebuilds the collection with the new metric.


---

## Why This Project Matters

Demonstrates production-relevant GenAI engineering skills:

* Agentic AI orchestration
* RAG system architecture
* Retrieval reliability and explainability
* Enterprise GenAI design patterns

---

GitHub: https://github.com/scharanlanka
LinkedIn: https://linkedin.com/in/saicharanlanka
