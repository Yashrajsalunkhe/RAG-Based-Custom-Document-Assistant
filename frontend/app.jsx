import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowUp, Check, ChevronDown, FileText, FolderOpen, LoaderCircle, Plus, RotateCcw, Search, Sparkles, Upload, X, Zap } from "lucide-react";

const API = `${window.location.protocol}//${window.location.hostname || "localhost"}:5000/api`;
const starterMessages = [{ role: "assistant", text: "Your research space is ready. Add a document and I’ll map the useful passages before answering.", trace: null }];

function App() {
  const [state, setState] = useState({ documents: [], documentCount: 0, chunkCount: 0, model: "qwen2.5-coder:7b", ready: false });
  const [messages, setMessages] = useState(starterMessages);
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [notice, setNotice] = useState("");
  const [theme, setTheme] = useState("dark");
  const [uploadFlow, setUploadFlow] = useState([]);
  const [searchFlow, setSearchFlow] = useState([]);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const statusLabel = uploading ? "Indexing document..." : state.ready ? "Ready to search" : "Waiting for a document";

  useEffect(() => { fetch(`${API}/state`).then((response) => response.json()).then(setState).catch(() => setNotice("API offline — start api.py to connect your workspace.")); }, []);

  function animatePipeline(setter, steps) {
    setter(steps.map((step, index) => ({ label: step, active: index === 0, done: false })));
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
        setTimeout(() => setter(steps.map((step) => ({ label: step, active: false, done: true }))), 500);
      }
    }, 600);
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true); setNotice(`Indexing ${file.name} into your workspace...`);
    animatePipeline(setUploadFlow, ["Parsing PDF", "Chunking text", "Embedding chunks", "Storing vectors"]);
    const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch(`${API}/documents`, { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error);
      setState(payload.state); setNotice(`${file.name} is indexed and ready to query.`);
    } catch (error) { setNotice(error.message || "Could not index this document."); }
    finally {
      setUploading(false);
      setTimeout(() => setUploadFlow([]), 1200);
      event.target.value = "";
    }
  }

  async function ask(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text || thinking) return;
    if (!state.ready) { setNotice("Index a document first, then ask anything inside it."); return; }
    setQuery(""); setNotice(""); setMessages((current) => [...current, { role: "user", text }, { role: "assistant", loading: true }]); setThinking(true);
    animatePipeline(setSearchFlow, ["Encoding query", "Semantic search", "Ranking matches", "Grounding answer"]);
    try {
      const response = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: text }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error);
      setMessages((current) => [...current.slice(0, -1), { role: "assistant", text: payload.answer, sources: payload.sources, trace: payload.trace }]);
    } catch (error) { setMessages((current) => [...current.slice(0, -1), { role: "assistant", text: error.message || "The assistant could not complete that request.", error: true }]); }
    finally {
      setThinking(false);
      setTimeout(() => setSearchFlow([]), 1200);
    }
  }

  async function reset() {
    try {
      const response = await fetch(`${API}/reset`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not reset the workspace.");
      setState(payload); setMessages(starterMessages); setNotice("Workspace cleared.");
    } catch (error) { setNotice(error.message || "API offline — could not clear the workspace."); }
  }

  return <main className={`app-shell ${theme}`}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><span>archive</span><span className="brand-tag">DOCUMENT INTELLIGENCE</span></div>
      <div className="top-actions"><span className="status"><i /> LOCAL ENGINE ACTIVE</span><button className="icon-button theme-toggle" onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")} title="Toggle theme">{theme === "dark" ? "☀" : "☾"}</button><button className="icon-button" onClick={reset} title="Reset workspace"><RotateCcw size={16} /></button><div className="avatar">YS</div></div>
    </header>
    <div className="workspace">
      <aside className="sidebar">
        <button className="new-chat" onClick={() => { setMessages(starterMessages); inputRef.current?.focus(); }}><Plus size={16} /> New conversation <span>⌘ K</span></button>
        <div className="side-label">YOUR LIBRARY <span>{state.documentCount}</span></div>
        <div className="document-list">
          {state.documents.length ? state.documents.map((document) => <div className="document-item" key={document.name}><div className="doc-icon"><FileText size={16} /></div><div><strong>{document.name}</strong><small>{document.chunks} chunks · indexed</small></div><Check className="doc-check" size={14} /></div>) : <div className="empty-library"><FolderOpen size={21} /><p>No documents yet</p><span>Upload a PDF to start building your knowledge base.</span></div>}
        </div>
        <div className="sidebar-bottom"><div className="side-label">INDEX HEALTH</div><div className="health-row"><span><i className={uploading ? "processing" : state.ready ? "healthy" : ""} /> {statusLabel}</span><b>{uploading ? "..." : `${state.chunkCount} chunks`}</b></div>{uploadFlow.length > 0 && <PipelineCard title="Indexing pipeline" steps={uploadFlow} />}{uploading && <div className="notice upload-progress" aria-live="polite"><LoaderCircle size={14} className="pulse-icon" /> Indexing uploaded PDF into the vector store…</div>}<div className="storage"><div style={{ width: `${Math.min(100, state.chunkCount ? 18 + state.chunkCount / 5 : 4)}%` }} /></div></div>
      </aside>
      <section className="conversation">
        <div className="conversation-head"><div><div className="eyebrow">PRIVATE WORKSPACE / 01</div><h1>Ask your archive<span className="cursor">_</span></h1><p>Grounded answers from the documents you trust.</p></div><div className="model-pill"><span className="pulse" /> {state.model}<ChevronDown size={14} /></div></div>
        <div className="thread">
          {messages.map((message, index) => <Message key={index} message={message} />)}
          {notice && <div className="notice"><Zap size={14} /> {notice}</div>}
        </div>
        <form className="composer-wrap" onSubmit={ask}><div className="composer"><textarea ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(event); } }} placeholder={state.ready ? "Ask anything about your documents..." : "Upload a document to begin..."} rows="1" /><div className="composer-tools"><label className="attach" title="Upload PDF"><input ref={fileRef} type="file" accept="application/pdf" onChange={uploadDocument} /><Upload size={16} /> {uploading ? "Indexing..." : "Add source"}</label><span className="shortcut">SHIFT + ENTER FOR NEW LINE</span><button className="send" disabled={thinking || uploading} title="Send message"><ArrowUp size={18} /></button></div></div><div className="composer-foot"><span><span className="key">↵</span> Press enter to send</span><span>Answers are grounded in your indexed sources</span></div></form>
      </section>
      <aside className="inspector"><div className="inspector-head"><div><div className="eyebrow">SYSTEM VIEW</div><h2>RAG trace</h2></div><div className="live-dot">LIVE</div></div>{searchFlow.length > 0 && <PipelineCard title="Query pipeline" steps={searchFlow} compact /> }<div className="trace-line"><span className="trace-node active"><Search size={14} /></span><div><strong>Semantic retrieval</strong><p>Finds the closest passages by meaning.</p></div></div><div className="trace-line"><span className="trace-node active"><FileText size={14} /></span><div><strong>Context assembly</strong><p>Compresses relevant chunks into context.</p></div></div><div className="trace-line last"><span className="trace-node active"><Sparkles size={14} /></span><div><strong>Grounded generation</strong><p>Responds using retrieved evidence.</p></div></div><div className="metrics"><div><span>DOCUMENTS</span><b>{state.documentCount.toString().padStart(2, "0")}</b></div><div><span>CHUNKS</span><b>{state.chunkCount.toString().padStart(2, "0")}</b></div></div><div className="source-panel"><div className="eyebrow">LATEST SOURCES</div>{messages.slice().reverse().find((message) => message.sources)?.sources?.map((source, index) => <div className="source" key={`${source.name}-${index}`}><FileText size={15} /><div><strong>{source.name}</strong><span>Page {source.page} · {source.preview}</span></div></div>) || <div className="source-empty">Sources from your next answer will appear here.</div>}</div></aside>
    </div>
  </main>;
}

function PipelineCard({ title, steps, compact = false }) {
  return <div className={`pipeline-card ${compact ? "compact" : ""}`}>
    <div className="pipeline-title">{title}</div>
    <div className="pipeline-steps">
      {steps.map((step, index) => <div className={`pipeline-step ${step.active ? "active" : ""} ${step.done ? "done" : ""}`} key={`${title}-${step.label}-${index}`}><span className="pipeline-dot" /><span>{step.label}</span></div> )}
    </div>
  </div>;
}

function Message({ message }) { if (message.loading) return <div className="message assistant"><div className="message-avatar"><Sparkles size={15} /></div><div className="loading-message"><span /><span /><span /></div></div>; return <div className={`message ${message.role} ${message.error ? "is-error" : ""}`}>{message.role === "assistant" && <div className="message-avatar"><Sparkles size={15} /></div>}<div className="message-body">{message.role === "user" && <div className="message-label">YOU</div>}<div className="message-text">{message.text}</div>{message.trace && <div className="trace-result"><span><Check size={13} /> {message.trace.chunks} passages retrieved</span><span>{message.trace.retrievalMs}ms search · {message.trace.generationMs}ms generation</span></div>}</div></div>; }

createRoot(document.getElementById("root")).render(<App />);

