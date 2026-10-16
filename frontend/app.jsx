import React, { useEffect, useRef, useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { ArrowUp, Check, ChevronDown, FileText, FolderOpen, LoaderCircle, Plus, RotateCcw, Search, Sparkles, Upload, X, Zap, AlertTriangle, Cpu, Globe, Activity, Layers, Database, Brain, Server } from "lucide-react";

const API = `${window.location.protocol}//${window.location.hostname || "localhost"}:5000/api`;
const starterMessages = [{ role: "assistant", text: "Your research space is ready. Add a document and I'll map the useful passages before answering.", trace: null }];
const modelChoices = [
  { provider: "ollama", model: "qwen2.5-coder:7b", label: "Local · Qwen 2.5 Coder", requiresOllama: true },
  { provider: "ollama", model: "llama3.2:3b", label: "Local · Llama 3.2", requiresOllama: true },
  { provider: "gemini", model: "gemini-3.8-flash", label: "Gemini · 3.8 Flash", requiresOllama: false },
  { provider: "gemini", model: "gemini-3.5-flash-lite", label: "Gemini · 3.5 Flash Lite", requiresOllama: false },
];

function App() {
  const [state, setState] = useState({ documents: [], documentCount: 0, chunkCount: 0, provider: "gemini", model: "gemini-3.8-flash", ready: false, ollamaOnline: false, ollamaModels: [] });
  const [messages, setMessages] = useState(starterMessages);
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [notice, setNotice] = useState({ text: "", type: "info" });
  const [theme, setTheme] = useState("dark");
  const [uploadFlow, setUploadFlow] = useState([]);
  const [searchFlow, setSearchFlow] = useState([]);
  const [showModelPanel, setShowModelPanel] = useState(false);
  const [switchingModel, setSwitchingModel] = useState(false);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const threadRef = useRef(null);
  const modelPanelRef = useRef(null);
  const statusLabel = uploading ? "Indexing document..." : state.ready ? "Ready to search" : "Waiting for a document";

  // Auto-scroll thread
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, notice]);

  // Close model panel on click outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (modelPanelRef.current && !modelPanelRef.current.contains(e.target)) {
        setShowModelPanel(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fetch initial state
  useEffect(() => {
    fetch(`${API}/state`)
      .then((r) => r.json())
      .then((data) => {
        setState(data);
        if (!data.ollamaOnline && data.provider === "ollama") {
          setNotice({ text: "Ollama is offline. Switch to Gemini or start Ollama with: ollama serve", type: "warning" });
        }
      })
      .catch(() => setNotice({ text: "API offline — start api.py to connect your workspace.", type: "error" }));
  }, []);

  // Auto-clear notices
  useEffect(() => {
    if (notice.text && notice.type === "success") {
      const t = setTimeout(() => setNotice({ text: "", type: "info" }), 4000);
      return () => clearTimeout(t);
    }
  }, [notice]);

  function animatePipeline(setter, steps) {
    setter(steps.map((step, index) => ({ ...step, active: index === 0, done: false })));
    let current = 0;
    const timer = setInterval(() => {
      current += 1;
      setter((previous) => previous.map((step, index) => ({
        ...step,
        active: index === current,
        done: index < current,
      })));
      if (current >= steps.length - 1) {
        clearInterval(timer);
        setTimeout(() => setter(steps.map((step) => ({ ...step, active: false, done: true }))), 600);
      }
    }, 700);
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setNotice({ text: `Indexing ${file.name} into your workspace...`, type: "info" });
    animatePipeline(setUploadFlow, [
      { label: "Parsing PDF", icon: "file", detail: "Extracting text and metadata" },
      { label: "Chunking text", icon: "layers", detail: "Splitting into semantic segments" },
      { label: "Embedding chunks", icon: "brain", detail: "Generating vector representations" },
      { label: "Storing vectors", icon: "database", detail: "Indexing into FAISS" },
    ]);
    const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch(`${API}/documents`, { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error);
      setState(payload.state);
      setNotice({ text: `${file.name} indexed — ${payload.document?.chunks || "?"} chunks ready to query.`, type: "success" });
    } catch (error) {
      setNotice({ text: error.message || "Could not index this document.", type: "error" });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadFlow([]), 1400);
      event.target.value = "";
    }
  }

  async function ask(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text || thinking) return;
    if (!state.ready) { setNotice({ text: "Index a document first, then ask anything inside it.", type: "warning" }); return; }
    setQuery(""); setNotice({ text: "", type: "info" });
    setMessages((current) => [...current, { role: "user", text }, { role: "assistant", loading: true }]);
    setThinking(true);
    animatePipeline(setSearchFlow, [
      { label: "Encoding query", icon: "brain", detail: "Converting to embedding vector" },
      { label: "Semantic search", icon: "search", detail: "Finding closest passages" },
      { label: "Ranking matches", icon: "layers", detail: "Scoring by relevance" },
      { label: "Grounding answer", icon: "sparkles", detail: "Generating response from context" },
    ]);
    try {
      const response = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: text }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error);
      setMessages((current) => [...current.slice(0, -1), { role: "assistant", text: payload.answer, sources: payload.sources, trace: payload.trace }]);
    } catch (error) {
      setMessages((current) => [...current.slice(0, -1), { role: "assistant", text: error.message || "The assistant could not complete that request.", error: true }]);
    } finally {
      setThinking(false);
      setTimeout(() => setSearchFlow([]), 1400);
    }
  }

  async function reset() {
    try {
      const response = await fetch(`${API}/reset`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not reset the workspace.");
      setState(payload); setMessages(starterMessages);
      setNotice({ text: "Workspace cleared.", type: "success" });
    } catch (error) {
      setNotice({ text: error.message || "API offline — could not clear the workspace.", type: "error" });
    }
  }

  async function switchModel(choice) {
    setShowModelPanel(false);
    if (choice.provider === state.provider && choice.model === state.model) return;

    setSwitchingModel(true);
    setNotice({ text: `Switching to ${choice.label}...`, type: "info" });
    try {
      const response = await fetch(`${API}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: choice.provider, model: choice.model }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not switch models.");
      setState(payload.state);
      setNotice({ text: `Switched to ${choice.label}`, type: "success" });
    } catch (error) {
      setNotice({ text: error.message || "Could not switch models.", type: "error" });
    } finally {
      setSwitchingModel(false);
    }
  }

  const activeChoice = modelChoices.find(c => c.provider === state.provider && c.model === state.model);
  const activeLabel = activeChoice ? activeChoice.label : `${state.provider === "gemini" ? "Gemini" : "Local"} · ${state.model}`;

  return <main className={`app-shell ${theme}`}>
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><Sparkles size={16} /></div>
        <span>archive</span>
        <span className="brand-tag">DOCUMENT INTELLIGENCE</span>
      </div>
      <div className="top-actions">
        <span className={`status ${state.provider === "gemini" ? "status-gemini" : state.ollamaOnline ? "status-ollama" : "status-offline"}`}>
          <i />
          {state.provider === "gemini" ? "GEMINI ACTIVE" : state.ollamaOnline ? "LOCAL ENGINE ACTIVE" : "OLLAMA OFFLINE"}
        </span>
        <button className="icon-button theme-toggle" onClick={() => setTheme((c) => c === "dark" ? "light" : "dark")} title="Toggle theme">
          {theme === "dark" ? "☀" : "☾"}
        </button>
        <button className="icon-button" onClick={reset} title="Reset workspace"><RotateCcw size={16} /></button>
        <div className="avatar">YS</div>
      </div>
    </header>

    <div className="workspace">
      {/* Sidebar */}
      <aside className="sidebar">
        <button className="new-chat" onClick={() => { setMessages(starterMessages); inputRef.current?.focus(); }}>
          <Plus size={16} /> New conversation <span>⌘ K</span>
        </button>

        <div className="side-label">YOUR LIBRARY <span>{state.documentCount}</span></div>
        <div className="document-list">
          {state.documents.length
            ? state.documents.map((document) => (
              <div className="document-item" key={document.name}>
                <div className="doc-icon"><FileText size={16} /></div>
                <div>
                  <strong>{document.name}</strong>
                  <small>{document.chunks} chunks · indexed</small>
                </div>
                <Check className="doc-check" size={14} />
              </div>
            ))
            : <div className="empty-library">
              <FolderOpen size={21} />
              <p>No documents yet</p>
              <span>Upload a PDF to start building your knowledge base.</span>
            </div>
          }
        </div>

        <div className="sidebar-bottom">
          <div className="side-label">INDEX HEALTH</div>
          <div className="health-row">
            <span><i className={uploading ? "processing" : state.ready ? "healthy" : ""} /> {statusLabel}</span>
            <b>{uploading ? "..." : `${state.chunkCount} chunks`}</b>
          </div>

          {/* Upload pipeline animation */}
          {uploadFlow.length > 0 && <PipelineCard title="Indexing pipeline" steps={uploadFlow} />}

          {uploading && (
            <div className="notice notice-info upload-progress" aria-live="polite">
              <LoaderCircle size={14} className="pulse-icon" /> Indexing uploaded PDF into the vector store…
            </div>
          )}

          <div className="storage">
            <div style={{ width: `${Math.min(100, state.chunkCount ? 18 + state.chunkCount / 5 : 4)}%` }} />
          </div>

          {/* Ollama status indicator */}
          <div className={`ollama-status ${state.ollamaOnline ? "online" : "offline"}`}>
            <Server size={13} />
            <span>{state.ollamaOnline ? "Ollama connected" : "Ollama offline"}</span>
            {state.ollamaOnline && state.ollamaModels?.length > 0 && (
              <span className="model-count">{state.ollamaModels.length} models</span>
            )}
          </div>
        </div>
      </aside>

      {/* Main conversation area */}
      <section className="conversation">
        <div className="conversation-head">
          <div>
            <div className="eyebrow">PRIVATE WORKSPACE / 01</div>
            <h1>Ask your archive<span className="cursor">_</span></h1>
            <p>Grounded answers from the documents you trust.</p>
          </div>

          {/* Model selector - redesigned as a dropdown panel */}
          <div className="model-selector-wrap" ref={modelPanelRef}>
            <button
              className={`model-pill ${switchingModel ? "switching" : ""}`}
              onClick={() => setShowModelPanel(!showModelPanel)}
              disabled={thinking || uploading || switchingModel}
              title="Switch language model"
            >
              <span className="pulse" />
              {switchingModel && <LoaderCircle size={12} className="pulse-icon" />}
              <span className="model-label">{activeLabel}</span>
              <ChevronDown size={14} className={showModelPanel ? "chevron-up" : ""} />
            </button>

            {showModelPanel && (
              <div className="model-dropdown">
                <div className="model-dropdown-header">SELECT MODEL</div>
                {modelChoices.map((choice) => {
                  const isActive = choice.provider === state.provider && choice.model === state.model;
                  const isDisabled = choice.requiresOllama && !state.ollamaOnline;
                  return (
                    <button
                      key={`${choice.provider}-${choice.model}`}
                      className={`model-option ${isActive ? "active" : ""} ${isDisabled ? "disabled" : ""}`}
                      onClick={() => !isDisabled && switchModel(choice)}
                      disabled={isDisabled}
                    >
                      <div className="model-option-left">
                        {choice.provider === "gemini" ? <Globe size={14} /> : <Cpu size={14} />}
                        <div>
                          <strong>{choice.label}</strong>
                          {isDisabled && <small className="model-warning">Ollama offline</small>}
                        </div>
                      </div>
                      {isActive && <Check size={14} className="model-check" />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Messages thread */}
        <div className="thread" ref={threadRef}>
          {messages.map((message, index) => <Message key={index} message={message} />)}
          {notice.text && (
            <div className={`notice notice-${notice.type}`}>
              {notice.type === "error" ? <AlertTriangle size={14} /> :
                notice.type === "warning" ? <AlertTriangle size={14} /> :
                  notice.type === "success" ? <Check size={14} /> :
                    <Zap size={14} />}
              {notice.text}
            </div>
          )}
        </div>

        {/* Composer */}
        <form className="composer-wrap" onSubmit={ask}>
          <div className="composer">
            <textarea
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(e); } }}
              placeholder={state.ready ? "Ask anything about your documents..." : "Upload a document to begin..."}
              rows="1"
            />
            <div className="composer-tools">
              <label className="attach" title="Upload PDF">
                <input ref={fileRef} type="file" accept="application/pdf" onChange={uploadDocument} />
                <Upload size={16} /> {uploading ? "Indexing..." : "Add source"}
              </label>
              <span className="shortcut">SHIFT + ENTER FOR NEW LINE</span>
              <button className="send" disabled={thinking || uploading} title="Send message">
                {thinking ? <LoaderCircle size={16} className="pulse-icon" /> : <ArrowUp size={18} />}
              </button>
            </div>
          </div>
          <div className="composer-foot">
            <span><span className="key">↵</span> Press enter to send</span>
            <span>Answers are grounded in your indexed sources</span>
          </div>
        </form>
      </section>

      {/* Inspector panel */}
      <aside className="inspector">
        <div className="inspector-head">
          <div>
            <div className="eyebrow">SYSTEM VIEW</div>
            <h2>RAG trace</h2>
          </div>
          <div className={`live-dot ${thinking ? "active" : ""}`}>
            {thinking ? "PROCESSING" : "LIVE"}
          </div>
        </div>

        {/* Search pipeline animation */}
        {searchFlow.length > 0 && <PipelineCard title="Query pipeline" steps={searchFlow} compact />}

        {/* RAG architecture visualization */}
        <div className="rag-flow">
          <div className="rag-flow-header">RAG PIPELINE</div>
          <div className={`trace-line ${thinking ? "animating" : ""}`}>
            <span className={`trace-node ${thinking ? "pulse-node" : "active"}`}><Search size={14} /></span>
            <div className="trace-connector" />
            <div>
              <strong>Semantic retrieval</strong>
              <p>Finds the closest passages by meaning.</p>
            </div>
          </div>
          <div className={`trace-line ${thinking ? "animating delay-1" : ""}`}>
            <span className={`trace-node ${thinking ? "pulse-node" : "active"}`}><FileText size={14} /></span>
            <div className="trace-connector" />
            <div>
              <strong>Context assembly</strong>
              <p>Compresses relevant chunks into context.</p>
            </div>
          </div>
          <div className={`trace-line last ${thinking ? "animating delay-2" : ""}`}>
            <span className={`trace-node ${thinking ? "pulse-node" : "active"}`}><Sparkles size={14} /></span>
            <div>
              <strong>Grounded generation</strong>
              <p>Responds using retrieved evidence.</p>
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="metrics">
          <div>
            <span>DOCUMENTS</span>
            <b>{state.documentCount.toString().padStart(2, "0")}</b>
          </div>
          <div>
            <span>CHUNKS</span>
            <b>{state.chunkCount.toString().padStart(2, "0")}</b>
          </div>
          <div>
            <span>PROVIDER</span>
            <b className="metric-provider">{state.provider === "gemini" ? "GEM" : "OLL"}</b>
          </div>
          <div>
            <span>MODEL</span>
            <b className="metric-model">{state.model.split("/").pop().split(":")[0].slice(0, 6)}</b>
          </div>
        </div>

        {/* Sources */}
        <div className="source-panel">
          <div className="eyebrow">LATEST SOURCES</div>
          {messages.slice().reverse().find((m) => m.sources)?.sources?.map((source, index) => (
            <div className="source" key={`${source.name}-${index}`}>
              <FileText size={15} />
              <div>
                <strong>{source.name}</strong>
                <span>Page {source.page} · {source.preview}</span>
              </div>
            </div>
          )) || <div className="source-empty">Sources from your next answer will appear here.</div>}
        </div>
      </aside>
    </div>
  </main>;
}

function PipelineCard({ title, steps, compact = false }) {
  const iconMap = { file: FileText, layers: Layers, brain: Brain, database: Database, search: Search, sparkles: Sparkles };
  return <div className={`pipeline-card ${compact ? "compact" : ""}`}>
    <div className="pipeline-title">{title}</div>
    <div className="pipeline-steps">
      {steps.map((step, index) => {
        const Icon = iconMap[step.icon] || Zap;
        return (
          <div className={`pipeline-step ${step.active ? "active" : ""} ${step.done ? "done" : ""}`} key={`${title}-${step.label}-${index}`}>
            <span className="pipeline-dot"><Icon size={11} /></span>
            <div className="pipeline-step-content">
              <span className="pipeline-step-label">{step.label}</span>
              {step.detail && <span className="pipeline-step-detail">{step.detail}</span>}
            </div>
            {step.done && <Check size={12} className="pipeline-check" />}
          </div>
        );
      })}
    </div>
    {/* Connection lines between steps */}
    <div className="pipeline-flow-line" />
  </div>;
}

function Message({ message }) {
  if (message.loading) return (
    <div className="message assistant">
      <div className="message-avatar"><Sparkles size={15} /></div>
      <div className="loading-message">
        <div className="loading-dots">
          <span /><span /><span />
        </div>
        <span className="loading-text">Analyzing your documents...</span>
      </div>
    </div>
  );

  return (
    <div className={`message ${message.role} ${message.error ? "is-error" : ""}`}>
      {message.role === "assistant" && <div className="message-avatar"><Sparkles size={15} /></div>}
      <div className="message-body">
        {message.role === "user" && <div className="message-label">YOU</div>}
        <div className="message-text">{message.text}</div>
        {message.trace && (
          <div className="trace-result">
            <span className="trace-success"><Check size={13} /> {message.trace.chunks} passages retrieved</span>
            <span className="trace-timing">{message.trace.retrievalMs}ms search · {message.trace.generationMs}ms generation</span>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
