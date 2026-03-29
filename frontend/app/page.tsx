"use client";

import Image from "next/image";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type Role = "user" | "assistant";
type InspectorTab = "retrieval" | "groundedness" | "sources";

type SourceItem = {
  filename?: string;
  page_range?: string;
  content_preview?: string;
};

type RetrievalRanking = {
  rank?: number;
  filename?: string;
  page_range?: string;
  score?: number | null;
  content_preview?: string;
};

type GenerationMetrics = {
  output_tokens_estimated?: number;
  output_chars?: number;
};

type Message = {
  id: string;
  role: Role;
  content: string;
  sources?: SourceItem[];
  retrievalRankings?: RetrievalRanking[];
  retrievalInspectorReport?: string;
  generationMetrics?: GenerationMetrics;
};

type Theme = "light" | "dark";
type ChunkingMode = "page" | "recursive_char" | "recursive_token";
type UploadChunkingOptions = {
  chunkingMode: ChunkingMode;
  pageWindow: number;
  pageOverlap: number;
  splitChunkSize: number;
  splitChunkOverlap: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8002";
const METRIC_DEFINITIONS = [
  {
    term: "Coverage",
    definition: "Percent of answer tokens found across all retrieved chunks.",
  },
  {
    term: "Support",
    definition: "Weighted evidence strength, giving more importance to higher-ranked chunks.",
  },
  {
    term: "Cos Similarity",
    definition: "Vector similarity between query and chunk embedding (higher means semantically closer).",
  },
  {
    term: "Overlap",
    definition: "Lexical overlap between answer text and a retrieved chunk.",
  },
  {
    term: "Relevance",
    definition: "Bucketed label (High/Medium/Low) derived from similarity thresholds.",
  },
  {
    term: "Token Coverage",
    definition: "Same coverage metric shown in the groundedness panel.",
  },
  {
    term: "Top-Rank Support",
    definition: "Same support metric shown in the groundedness panel.",
  },
  {
    term: "Best Supporting Chunk",
    definition: "Highest-overlap chunk that best supports the answer.",
  },
  {
    term: "Output Tokens (est.)",
    definition: "Estimated number of LLM tokens generated in the final assistant answer.",
  },
];

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeTokens(text: string) {
  const stop = new Set([
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "into",
    "about",
    "only",
  ]);
  return new Set(
    (text.toLowerCase().match(/[a-z0-9]{3,}/g) || []).filter((t) => !stop.has(t)),
  );
}

function chunkOverlap(answer: string, chunkText: string) {
  const a = normalizeTokens(answer);
  const b = normalizeTokens(chunkText);
  if (a.size === 0) return 0;
  let overlap = 0;
  for (const tok of a) {
    if (b.has(tok)) overlap += 1;
  }
  return overlap / a.size;
}

function parseGroundedness(report: string) {
  const coverage = report.match(/coverage[^\n]*\*\*([0-9]+(?:\.[0-9]+)?)%\*\*/i)?.[1];
  const support = report.match(/weighted top-rank support[^\n]*\*\*([0-9]+(?:\.[0-9]+)?)%\*\*/i)?.[1];
  const verdict = report.match(/verdict:\s*\*\*([^*]+)\*\*/i)?.[1];
  const bestRank = report.match(/best supporting chunk:\s*\*\*Rank\s+([0-9]+)/i)?.[1];
  return {
    coverage: coverage ? Number(coverage) : null,
    support: support ? Number(support) : null,
    verdict: verdict || "Unknown",
    bestRank: bestRank ? Number(bestRank) : null,
  };
}

function formatPercentMetric(value: number | null) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "N/A";
}

function renderInlineMarkdown(text: string, keyPrefix: string) {
  const tokenRegex = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let tokenIndex = 0;
  let match = tokenRegex.exec(text);

  while (match) {
    const start = match.index;
    const full = match[0];
    if (start > lastIndex) {
      nodes.push(text.slice(lastIndex, start));
    }
    if (full.startsWith("**") && full.endsWith("**")) {
      nodes.push(
        <strong key={`${keyPrefix}-b-${tokenIndex}`}>{full.slice(2, -2)}</strong>,
      );
    } else if (full.startsWith("`") && full.endsWith("`")) {
      nodes.push(<code key={`${keyPrefix}-c-${tokenIndex}`}>{full.slice(1, -1)}</code>);
    } else if (full.startsWith("*") && full.endsWith("*")) {
      nodes.push(<em key={`${keyPrefix}-i-${tokenIndex}`}>{full.slice(1, -1)}</em>);
    } else {
      nodes.push(full);
    }
    lastIndex = start + full.length;
    tokenIndex += 1;
    match = tokenRegex.exec(text);
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function renderMarkdown(text: string) {
  const content = (text || "").replace(/\r\n/g, "\n");
  if (!content.trim()) return null;

  const lines = content.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let blockIndex = 0;
  const orderedPattern = /^\s*\d+\.\s+(.*)$/;
  const unorderedPattern = /^\s*[-*+]\s+(.*)$/;

  const isListLine = (line: string) => orderedPattern.test(line) || unorderedPattern.test(line);

  while (i < lines.length) {
    const raw = lines[i];
    if (!raw.trim()) {
      i += 1;
      continue;
    }

    const orderedMatch = raw.match(orderedPattern);
    if (orderedMatch) {
      const items: string[] = [orderedMatch[1].trim()];
      i += 1;
      while (i < lines.length) {
        const next = lines[i];
        const nextOrdered = next.match(orderedPattern);
        if (nextOrdered) {
          items.push(nextOrdered[1].trim());
          i += 1;
          continue;
        }
        if (!next.trim()) break;
        if (!isListLine(next) && items.length > 0) {
          items[items.length - 1] = `${items[items.length - 1]} ${next.trim()}`;
          i += 1;
          continue;
        }
        break;
      }
      blocks.push(
        <ol key={`md-ol-${blockIndex}`}>
          {items.map((item, idx) => (
            <li key={`md-ol-item-${blockIndex}-${idx}`}>
              {renderInlineMarkdown(item, `md-ol-inline-${blockIndex}-${idx}`)}
            </li>
          ))}
        </ol>,
      );
      blockIndex += 1;
      continue;
    }

    const unorderedMatch = raw.match(unorderedPattern);
    if (unorderedMatch) {
      const items: string[] = [unorderedMatch[1].trim()];
      i += 1;
      while (i < lines.length) {
        const next = lines[i];
        const nextUnordered = next.match(unorderedPattern);
        if (nextUnordered) {
          items.push(nextUnordered[1].trim());
          i += 1;
          continue;
        }
        if (!next.trim()) break;
        if (!isListLine(next) && items.length > 0) {
          items[items.length - 1] = `${items[items.length - 1]} ${next.trim()}`;
          i += 1;
          continue;
        }
        break;
      }
      blocks.push(
        <ul key={`md-ul-${blockIndex}`}>
          {items.map((item, idx) => (
            <li key={`md-ul-item-${blockIndex}-${idx}`}>
              {renderInlineMarkdown(item, `md-ul-inline-${blockIndex}-${idx}`)}
            </li>
          ))}
        </ul>,
      );
      blockIndex += 1;
      continue;
    }

    const paragraph: string[] = [raw.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() && !isListLine(lines[i])) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    const paraText = paragraph.join(" ");
    blocks.push(
      <p key={`md-p-${blockIndex}`}>
        {renderInlineMarkdown(paraText, `md-p-inline-${blockIndex}`)}
      </p>,
    );
    blockIndex += 1;
  }

  return blocks;
}

export default function Page() {
  const [theme, setTheme] = useState<Theme>("light");
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [kbCount, setKbCount] = useState(0);
  const [kbDocuments, setKbDocuments] = useState(0);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);
  const [ingestStatus, setIngestStatus] = useState("");
  const [error, setError] = useState("");
  const [activeTabs, setActiveTabs] = useState<Record<string, InspectorTab>>({});
  const [expandedInspectors, setExpandedInspectors] = useState<Record<string, boolean>>({});
  const [showMetricDefinitions, setShowMetricDefinitions] = useState(false);
  const [hybridEnabled, setHybridEnabled] = useState(false);
  const [keywordWeight, setKeywordWeight] = useState(0.3);
  const [semanticWeight, setSemanticWeight] = useState(0.7);
  const [showChunkingModal, setShowChunkingModal] = useState(false);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[]>([]);
  const [chunkingMode, setChunkingMode] = useState<ChunkingMode>("page");
  const [pageWindow, setPageWindow] = useState(1);
  const [pageOverlap, setPageOverlap] = useState(0);
  const [splitChunkSize, setSplitChunkSize] = useState(900);
  const [splitChunkOverlap, setSplitChunkOverlap] = useState(120);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const assistantCount = useMemo(
    () => messages.filter((m) => m.role === "assistant").length,
    [messages],
  );

  const latestAssistant = useMemo(() => {
    return [...messages]
      .reverse()
      .find((m) => m.role === "assistant" && (m.retrievalRankings?.length || m.sources?.length));
  }, [messages]);

  const chunksRetrieved = latestAssistant?.retrievalRankings?.length || 0;

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
      return;
    }
    document.documentElement.setAttribute("data-theme", "light");
  }, []);

  useEffect(() => {
    void refreshKnowledgeBaseInfo();
  }, []);

  async function refreshKnowledgeBaseInfo() {
    try {
      const res = await fetch(`${API_BASE_URL}/documents`);
      if (!res.ok) return;
      const data = (await res.json()) as {
        collection?: { count?: number };
        files?: { name: string }[];
      };
      setKbCount(data.collection?.count ?? 0);
      setKbDocuments(data.files?.length ?? 0);
      setIndexedFiles((data.files || []).map((f) => f.name));
    } catch {
      // no-op
    }
  }

  function applyTheme(nextTheme: Theme) {
    setTheme(nextTheme);
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", nextTheme);
    }
    if (typeof window !== "undefined") {
      window.localStorage.setItem("theme", nextTheme);
    }
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim() || isStreaming) return;
    if (kbCount <= 0) {
      setError("Upload and index at least one document before chatting.");
      return;
    }

    setError("");
    if (hybridEnabled) {
      if (
        !Number.isFinite(keywordWeight) ||
        !Number.isFinite(semanticWeight) ||
        keywordWeight < 0 ||
        semanticWeight < 0
      ) {
        setError("Hybrid weights must be valid non-negative numbers.");
        return;
      }
      const total = keywordWeight + semanticWeight;
      if (Math.abs(total - 1) > 0.001) {
        setError("Hybrid weights must sum to 1.00 (example: 0.30 and 0.70).");
        return;
      }
    }
    setIsStreaming(true);

    const userMsg: Message = {
      id: makeId(),
      role: "user",
      content: query.trim(),
    };
    const assistantMsgId = makeId();

    const nextMessages = [
      ...messages,
      userMsg,
      { id: assistantMsgId, role: "assistant" as const, content: "" },
    ];
    setMessages(nextMessages);
    setQuery("");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMsg.content,
          chat_history: messages.map((m) => ({ role: m.role, content: m.content })),
          k: 4,
          hybrid_enabled: hybridEnabled,
          keyword_weight: keywordWeight,
          semantic_weight: semanticWeight,
        }),
      });

      if (!response.ok || !response.body) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const processEvent = (block: string) => {
        const lines = block.split("\n").filter(Boolean);
        let eventName = "message";
        let data = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data += line.slice(5).trim();
          }
        }

        if (!data) return;
        let parsed: any;
        try {
          parsed = JSON.parse(data);
        } catch {
          return;
        }

        if (eventName === "token" && parsed.token) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: m.content + parsed.token } : m,
            ),
          );
        }

        if (eventName === "error" && parsed.error) {
          setError(parsed.error);
        }

        if (eventName === "done" && typeof parsed.answer === "string") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: parsed.answer,
                    sources: Array.isArray(parsed.sources) ? parsed.sources : [],
                    retrievalRankings: Array.isArray(parsed.retrieval_rankings)
                      ? parsed.retrieval_rankings
                      : [],
                    retrievalInspectorReport:
                      typeof parsed.retrieval_inspector_report === "string"
                        ? parsed.retrieval_inspector_report
                        : "",
                    generationMetrics:
                      parsed.generation_metrics &&
                      typeof parsed.generation_metrics === "object"
                        ? {
                            output_tokens_estimated:
                              typeof parsed.generation_metrics.output_tokens_estimated === "number"
                                ? parsed.generation_metrics.output_tokens_estimated
                                : undefined,
                            output_chars:
                              typeof parsed.generation_metrics.output_chars === "number"
                                ? parsed.generation_metrics.output_chars
                                : undefined,
                          }
                        : undefined,
                  }
                : m,
            ),
          );
          setActiveTabs((prev) => ({ ...prev, [assistantMsgId]: "retrieval" }));
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");

        while (boundary !== -1) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          processEvent(block);
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: m.content || "Could not stream response from backend." }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  }

  async function uploadDocuments(files: File[], options: UploadChunkingOptions) {
    if (files.length === 0 || isUploading) return;
    setError("");
    setIngestStatus("Indexing documents...");
    setIsUploading(true);

    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }
      formData.append("chunking_mode", options.chunkingMode);
      formData.append("page_window", String(options.pageWindow));
      formData.append("page_overlap", String(options.pageOverlap));
      formData.append("split_chunk_size", String(options.splitChunkSize));
      formData.append("split_chunk_overlap", String(options.splitChunkOverlap));

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || `Upload failed with status ${response.status}`);
      }

      const chunks = payload?.chunks_indexed ?? 0;
      const modeText =
        options.chunkingMode === "page"
          ? `page chunking (window=${options.pageWindow}, overlap=${options.pageOverlap})`
          : options.chunkingMode === "recursive_char"
            ? `recursive character splitter (size=${options.splitChunkSize}, overlap=${options.splitChunkOverlap})`
            : `token splitter (size=${options.splitChunkSize}, overlap=${options.splitChunkOverlap})`;
      setIngestStatus(`Indexed successfully. ${chunks} chunks ready using ${modeText}.`);
      await refreshKnowledgeBaseInfo();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setIngestStatus("");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function openChunkingPrompt(files: File[]) {
    if (files.length === 0 || isUploading) return;
    setError("");
    setPendingUploadFiles(files);
    setShowChunkingModal(true);
  }

  async function confirmChunkingAndUpload() {
    if (pendingUploadFiles.length === 0 || isUploading) return;

    const safePageWindow = Math.max(1, Math.floor(pageWindow || 1));
    const safePageOverlap = Math.max(0, Math.floor(pageOverlap || 0));
    const safeSplitChunkSize = Math.max(50, Math.floor(splitChunkSize || 900));
    const safeSplitChunkOverlap = Math.max(0, Math.floor(splitChunkOverlap || 0));
    if (safePageOverlap >= safePageWindow) {
      setError("Page overlap must be smaller than page window.");
      return;
    }
    if (safeSplitChunkOverlap >= safeSplitChunkSize) {
      setError("Split chunk overlap must be smaller than split chunk size.");
      return;
    }

    setShowChunkingModal(false);
    const files = pendingUploadFiles;
    setPendingUploadFiles([]);
    await uploadDocuments(files, {
      chunkingMode,
      pageWindow: safePageWindow,
      pageOverlap: safePageOverlap,
      splitChunkSize: safeSplitChunkSize,
      splitChunkOverlap: safeSplitChunkOverlap,
    });
  }

  async function clearKnowledgeBase() {
    if (isClearing || isUploading || isStreaming) return;
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        "Delete all indexed documents and vector data? This cannot be undone.",
      );
      if (!confirmed) return;
    }

    setError("");
    setIngestStatus("Clearing knowledge base...");
    setIsClearing(true);

    try {
      const response = await fetch(`${API_BASE_URL}/knowledge-base`, {
        method: "DELETE",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || `Delete failed with status ${response.status}`);
      }

      setMessages([]);
      setKbCount(0);
      setKbDocuments(0);
      setIndexedFiles([]);
      setIngestStatus("Knowledge base deleted.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Delete failed";
      setError(message);
      setIngestStatus("");
    } finally {
      setIsClearing(false);
    }
  }

  return (
    <main className="design-shell">
      <aside className="design-sidebar">
        <div className="logo-row">
          <div className="logo-mark">R</div>
          <h1>RAG Chat</h1>
        </div>

        <button
          type="button"
          className="new-conversation"
          onClick={() => {
            setMessages([]);
            setError("");
          }}
        >
          + New conversation
        </button>

        <section className="side-card">
          <p className="side-title">Knowledge Base</p>
          <div className="stat-row">
            <span className="dot" />
            <span>Status</span>
            <strong>{kbCount > 0 ? "Indexed" : "Empty"}</strong>
          </div>
          <div className="stat-row">
            <span>Chunks</span>
            <strong>{kbCount}</strong>
          </div>
          <div className="stat-row">
            <span>Documents</span>
            <strong>{kbDocuments}</strong>
          </div>
          <div className="stat-row">
            <span>Responses</span>
            <strong>{assistantCount}</strong>
          </div>
        </section>

        <section className="side-card muted">
          <p className="side-title">Search Mode</p>
          <label className="chunking-option">
            <input
              type="checkbox"
              checked={hybridEnabled}
              onChange={(e) => setHybridEnabled(e.target.checked)}
              disabled={isStreaming}
            />
            <span>Enable Hybrid Retrieval (BM25 + Semantic)</span>
          </label>
          {hybridEnabled ? (
            <div className="chunking-grid">
              <label>
                Keyword (BM25)
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={keywordWeight}
                  onChange={(e) => setKeywordWeight(Number(e.target.value))}
                  disabled={isStreaming}
                />
              </label>
              <label>
                Semantic
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={semanticWeight}
                  onChange={(e) => setSemanticWeight(Number(e.target.value))}
                  disabled={isStreaming}
                />
              </label>
              <p className="chunking-note">
                Weights must sum to 1.00 (for example: 0.30 BM25 and 0.70 semantic).
              </p>
            </div>
          ) : (
            <p className="side-empty">Semantic retrieval only</p>
          )}
        </section>

        <section className="side-card muted">
          <p className="side-title">Available Files</p>
          {indexedFiles.length === 0 ? (
            <p className="side-empty">No files indexed yet</p>
          ) : (
            indexedFiles.map((file) => (
              <p className="recent-item" key={file}>
                {file}
              </p>
            ))
          )}
        </section>
      </aside>

      <section className="design-main">
        <header className="conversation-topbar">
          <div className="conversation-title-wrap">
            <h2>Conversation</h2>
            <span className="retrieved-badge">{chunksRetrieved} chunks retrieved</span>
          </div>

          <div className="top-actions">
            <input
              ref={fileInputRef}
              type="file"
              hidden
              multiple
              accept=".pdf,.docx,.txt"
              onChange={(e) => openChunkingPrompt(Array.from(e.target.files || []))}
            />
            <button
              type="button"
              className="top-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isClearing}
            >
              {isUploading ? "Uploading..." : "Upload"}
            </button>
            <button
              type="button"
              className="top-btn"
              onClick={() => applyTheme(theme === "light" ? "dark" : "light")}
            >
              Settings
            </button>
            <button
              type="button"
              className="danger-btn"
              onClick={clearKnowledgeBase}
              disabled={isClearing || isUploading || kbCount <= 0}
            >
              {isClearing ? "Deleting..." : "Delete KB"}
            </button>
            <button
              type="button"
              className="help-btn"
              aria-label="Open metric definitions"
              aria-expanded={showMetricDefinitions}
              onClick={() => setShowMetricDefinitions((prev) => !prev)}
            >
              ?
            </button>
          </div>
        </header>

        {showMetricDefinitions && (
          <div
            className="metric-help-backdrop"
            role="presentation"
            onClick={() => setShowMetricDefinitions(false)}
          >
            <section
              className="metric-help-card"
              role="dialog"
              aria-modal="true"
              aria-label="Retrieval metric definitions"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="metric-help-head">
                <h3>Metric Definitions</h3>
                <button
                  type="button"
                  className="metric-help-close"
                  onClick={() => setShowMetricDefinitions(false)}
                  aria-label="Close metric definitions"
                >
                  ×
                </button>
              </div>
              <div className="metric-help-list">
                {METRIC_DEFINITIONS.map((item) => (
                  <p key={item.term}>
                    <strong>{item.term}:</strong> {item.definition}
                  </p>
                ))}
              </div>
            </section>
          </div>
        )}

        {showChunkingModal && (
          <div
            className="chunking-backdrop"
            role="presentation"
            onClick={() => {
              setShowChunkingModal(false);
              setPendingUploadFiles([]);
              if (fileInputRef.current) fileInputRef.current.value = "";
            }}
          >
            <section
              className="chunking-card"
              role="dialog"
              aria-modal="true"
              aria-label="Choose chunking strategy"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="chunking-head">
                <h3>Choose Chunking Strategy</h3>
                <button
                  type="button"
                  className="chunking-close"
                  onClick={() => {
                    setShowChunkingModal(false);
                    setPendingUploadFiles([]);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                  aria-label="Close chunking selection"
                >
                  ×
                </button>
              </div>
              <p className="chunking-subtitle">
                {pendingUploadFiles.length} file{pendingUploadFiles.length === 1 ? "" : "s"} selected.
              </p>

              <div className="chunking-options">
                <label className="chunking-option">
                  <input
                    type="radio"
                    name="chunking-mode"
                    checked={chunkingMode === "page"}
                    onChange={() => setChunkingMode("page")}
                  />
                  <span>Page Chunking (shows page ranges in Sources)</span>
                </label>
                <label className="chunking-option">
                  <input
                    type="radio"
                    name="chunking-mode"
                    checked={chunkingMode === "recursive_char"}
                    onChange={() => setChunkingMode("recursive_char")}
                  />
                  <span>Recursive Character Splitter</span>
                </label>
                <label className="chunking-option">
                  <input
                    type="radio"
                    name="chunking-mode"
                    checked={chunkingMode === "recursive_token"}
                    onChange={() => setChunkingMode("recursive_token")}
                  />
                  <span>Token Text Splitter</span>
                </label>
              </div>

              {chunkingMode === "page" ? (
                <div className="chunking-grid">
                  <label>
                    Page Window
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={pageWindow}
                      onChange={(e) => setPageWindow(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Page Overlap
                    <input
                      type="number"
                      min={0}
                      step={1}
                      value={pageOverlap}
                      onChange={(e) => setPageOverlap(Number(e.target.value))}
                    />
                  </label>
                  <p className="chunking-note">
                    Examples: 1/0 = page-by-page, 2/1 = 1-2, 2-3, 3-4, 3/2 = sliding 3-page chunks.
                  </p>
                </div>
              ) : (
                <div className="chunking-grid">
                  <label>
                    Chunk Size
                    <input
                      type="number"
                      min={50}
                      step={10}
                      value={splitChunkSize}
                      onChange={(e) => setSplitChunkSize(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Chunk Overlap
                    <input
                      type="number"
                      min={0}
                      step={10}
                      value={splitChunkOverlap}
                      onChange={(e) => setSplitChunkOverlap(Number(e.target.value))}
                    />
                  </label>
                  <p className="chunking-note">
                    For recursive splitters, page ranges are hidden in Sources.
                  </p>
                </div>
              )}

              <div className="chunking-actions">
                <button
                  type="button"
                  className="top-btn"
                  onClick={() => {
                    setShowChunkingModal(false);
                    setPendingUploadFiles([]);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                >
                  Cancel
                </button>
                <button type="button" className="top-btn" onClick={() => void confirmChunkingAndUpload()}>
                  Index documents
                </button>
              </div>
            </section>
          </div>
        )}

        <div className="conversation-body" role="log" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">Upload documents, then ask your first question.</div>
          ) : (
            messages.map((m) => {
              const tab = activeTabs[m.id] || "retrieval";
              const isExpanded = expandedInspectors[m.id] || false;
              const rankings = m.retrievalRankings || [];
              const sources = m.sources || [];
              const inspector = parseGroundedness(m.retrievalInspectorReport || "");
              const outputTokens = m.generationMetrics?.output_tokens_estimated;
              const outputChars = m.generationMetrics?.output_chars;
              const sourcesUnavailable =
                /^partially grounded$/i.test(inspector.verdict) ||
                /^weakly grounded$/i.test(inspector.verdict);
              const displaySupport = sourcesUnavailable ? null : inspector.support;
              const displayBestRank = sourcesUnavailable ? null : inspector.bestRank;
              const uniqueSources = new Set(
                (sources.length > 0 ? sources : rankings)
                  .map((s) => s.filename || "Unknown")
                  .filter(Boolean),
              ).size;
              const sparkValues = rankings
                .slice(0, 10)
                .map((row) =>
                  typeof row.score === "number" ? Math.max(0, Math.min(1, 1 - row.score)) : 0.12,
                );

              return (
                <article key={m.id} className={`conversation-item ${m.role}`}>
                  {m.role === "user" ? (
                    <div className="user-pill">{m.content}</div>
                  ) : (
                    <>
                      <div className="answer-card">
                        <div className="answer-markdown">
                          {renderMarkdown(m.content || (isStreaming ? "..." : ""))}
                        </div>
                        {sources.length > 0 ? (
                          <p className="source-summary">
                            Summarized from {Array.from(new Set(sources.map((s) => s.filename || "Unknown"))).join(" • ")}
                          </p>
                        ) : null}
                      </div>

                      {(rankings.length > 0 || sources.length > 0 || m.retrievalInspectorReport) && (
                        <div className="inspector-shell">
                          <div className="inspector-summary">
                            <div className="inspector-icon" aria-hidden>
                              +
                            </div>
                            <div className="inspector-intro">
                              <p className="inspector-title">
                                Retrieval Inspector
                                <span className="inspector-verdict">{inspector.verdict}</span>
                              </p>
                              <p className="inspector-subtitle">
                                {rankings.length} chunks retrieved · {uniqueSources} unique sources ·{" "}
                                {inspector.coverage?.toFixed(1) ?? "0.0"}% coverage
                                {typeof outputTokens === "number"
                                  ? ` · ~${outputTokens} output tokens`
                                  : ""}
                              </p>
                            </div>

                            <div className="inspector-sparkline" aria-hidden>
                              {(sparkValues.length > 0 ? sparkValues : [0.15, 0.2, 0.25, 0.18]).map(
                                (value, idx) => (
                                  <span
                                    key={`${m.id}-spark-${idx}`}
                                    style={{ height: `${Math.max(8, Math.round(value * 56))}px` }}
                                  />
                                ),
                              )}
                            </div>

                            <div className="inspector-number">
                              <strong>{formatPercentMetric(inspector.coverage)}</strong>
                              <span>Coverage</span>
                            </div>
                            <div className="inspector-number">
                              <strong>{formatPercentMetric(displaySupport)}</strong>
                              <span>Support</span>
                            </div>

                            <button
                              type="button"
                              className="inspector-toggle"
                              onClick={() =>
                                setExpandedInspectors((prev) => ({ ...prev, [m.id]: !isExpanded }))
                              }
                            >
                              {isExpanded ? "Hide details" : "Details"}
                            </button>
                          </div>

                          {isExpanded && (
                            <div className="inspector-card">
                              <div className="tabs">
                                <button
                                  className={tab === "retrieval" ? "active" : ""}
                                  type="button"
                                  onClick={() =>
                                    setActiveTabs((prev) => ({ ...prev, [m.id]: "retrieval" }))
                                  }
                                >
                                  Retrieval
                                </button>
                                <button
                                  className={tab === "groundedness" ? "active" : ""}
                                  type="button"
                                  onClick={() =>
                                    setActiveTabs((prev) => ({ ...prev, [m.id]: "groundedness" }))
                                  }
                                >
                                  Groundedness
                                </button>
                                <button
                                  className={tab === "sources" ? "active" : ""}
                                  type="button"
                                  onClick={() =>
                                    setActiveTabs((prev) => ({ ...prev, [m.id]: "sources" }))
                                  }
                                >
                                  Sources
                                </button>
                              </div>

                              {tab === "retrieval" && (
                                <div className="tab-panel">
                                  {rankings.length === 0 ? (
                                    <p className="muted-text">No retrieval rankings available.</p>
                                  ) : (
                                    <div className="retrieval-table">
                                      <div className="retrieval-head">
                                        <span>Rank</span>
                                        <span>Source</span>
                                        <span>Cos Similarity</span>
                                        <span>Overlap</span>
                                        <span>Relevance</span>
                                      </div>
                                      {rankings.map((row, idx) => {
                                        const distance =
                                          typeof row.score === "number" ? row.score : null;
                                        const similarity =
                                          distance === null
                                            ? null
                                            : Math.max(0, Math.min(1, 1 - distance));
                                        const overlap = chunkOverlap(
                                          m.content,
                                          row.content_preview || "",
                                        );
                                        const relevance =
                                          similarity !== null && similarity >= 0.75
                                            ? "High"
                                            : similarity !== null && similarity >= 0.6
                                              ? "Medium"
                                              : "Low";

                                        return (
                                          <div className="retrieval-row" key={`${m.id}-rank-${idx}`}>
                                            <span>{row.rank ?? idx + 1}</span>
                                            <span>
                                              {row.filename || "Unknown"}
                                              {row.page_range ? ` (${row.page_range})` : ""}
                                            </span>
                                            <span>
                                              {typeof similarity === "number"
                                                ? similarity.toFixed(4)
                                                : "N/A"}
                                            </span>
                                            <span>{(overlap * 100).toFixed(1)}%</span>
                                            <span className={`pill ${relevance.toLowerCase()}`}>
                                              {relevance}
                                            </span>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>
                              )}

                              {tab === "groundedness" && (
                                <div className="tab-panel groundedness-grid">
                                  <div className="gauge">
                                    <div
                                      className="gauge-ring"
                                      style={{
                                        background: `conic-gradient(var(--ok) ${(inspector.coverage || 0) * 3.6}deg, var(--panel-soft) 0deg)`,
                                      }}
                                    >
                                      <div className="gauge-inner">
                                        {(inspector.coverage || 0).toFixed(1)}%
                                      </div>
                                    </div>
                                  </div>
                                  <div className="metric-card">
                                    <p>Token Coverage</p>
                                    <strong>{formatPercentMetric(inspector.coverage)}</strong>
                                  </div>
                                  <div className="metric-card">
                                    <p>Top-Rank Support</p>
                                    <strong>{formatPercentMetric(displaySupport)}</strong>
                                  </div>
                                  <div className="metric-card">
                                    <p>Best Supporting Chunk</p>
                                    <strong>
                                      {displayBestRank ? `Rank ${displayBestRank}` : "N/A"}
                                    </strong>
                                  </div>
                                  <div className="metric-card">
                                    <p>Output Tokens (est.)</p>
                                    <strong>
                                      {typeof outputTokens === "number" ? outputTokens : "N/A"}
                                    </strong>
                                  </div>
                                  <div className="metric-card">
                                    <p>Output Characters</p>
                                    <strong>{typeof outputChars === "number" ? outputChars : "N/A"}</strong>
                                  </div>
                                  <div className="verdict-chip">{inspector.verdict}</div>
                                </div>
                              )}

                              {tab === "sources" && (
                                <div className="tab-panel">
                                  {sourcesUnavailable ? (
                                    <p className="muted-text">Sources Not available</p>
                                  ) : sources.length === 0 ? (
                                    <p className="muted-text">No source snippets returned.</p>
                                  ) : (
                                    <div className="sources-list">
                                      {sources.map((source, idx) => (
                                        <div className="source-item" key={`${m.id}-source-${idx}`}>
                                          <p className="source-title">
                                            {source.filename || "Unknown"}
                                            {source.page_range ? ` (${source.page_range})` : ""}
                                          </p>
                                          {source.content_preview ? (
                                            <p className="source-preview">{source.content_preview}</p>
                                          ) : null}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </article>
              );
            })
          )}
        </div>

        <footer className="conversation-composer-wrap">
          <form className="conversation-composer" onSubmit={sendMessage}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about your documents..."
              disabled={isStreaming}
            />
            <button type="submit" disabled={isStreaming || !query.trim()}>
              <Image src="/send-up-arrow.svg" alt="send" width={16} height={16} />
            </button>
          </form>
          {ingestStatus ? <p className="status">{ingestStatus}</p> : null}
          {error ? <p className="error">{error}</p> : null}
        </footer>
      </section>
    </main>
  );
}
