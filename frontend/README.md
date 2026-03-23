# Frontend - Retrieval Inspector UI

This Next.js app is the user-facing interface for the RAG system. It supports document upload, chunking strategy selection, streaming chat, and retrieval diagnostics.

## Features

- Streaming answer rendering via `POST /chat/stream`
- Retrieval Inspector with:
  - Coverage, support, and groundedness verdict
  - Retrieval table (similarity, overlap, relevance)
  - Sources tab and groundedness tab
- Upload flow with chunking strategy selection before indexing
- Theme toggle and knowledge base controls

## Stack

- Next.js 15
- React 19
- TypeScript

## Prerequisites

- Node.js 18+
- Running backend API (`http://localhost:8002` by default)

## Setup

```bash
cd frontend
npm install
```

## Run (Dev)

```bash
npm run dev
```

App runs on `http://localhost:3002`.

## Environment

Set API base URL if needed:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8002
```

You can place this in `.env.local`.

## Build

```bash
npm run build
npm run start
```

## Key Files

- `app/page.tsx` - main chat UI, upload modal, inspector tabs
- `app/globals.css` - visual design and component styling
- `next.config.ts` - Next.js configuration

## UX Notes

- Page chunking mode displays page ranges in Sources.
- Recursive character/token modes do not display page ranges.
