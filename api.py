"""HTTP API for the React document assistant client."""

import os
import tempfile
import time
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

_state_lock = Lock()
_vector_store: Any | None = None
_processor: Any | None = None
_rag: Any | None = None
_documents: list[dict[str, Any]] = []


def _get_services() -> tuple[Any, Any, Any]:
    global _vector_store, _processor, _rag
    with _state_lock:
        from document_processor import DocumentProcessor
        from rag import RAG
        from vector_store import VectorStore

        if _vector_store is None:
            _vector_store = VectorStore()
        if _processor is None:
            _processor = DocumentProcessor()
        if _rag is None:
            _rag = RAG(_vector_store)
    return _vector_store, _processor, _rag


def _snapshot() -> dict[str, Any]:
    return {
        "documents": _documents,
        "documentCount": len(_documents),
        "chunkCount": sum(document["chunks"] for document in _documents),
        "model": os.getenv("OLLAMA_MODEL") or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "ready": bool(_documents),
    }


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True, "service": "document-assistant"})


@app.get("/api/state")
def state() -> Any:
    return jsonify(_snapshot())


@app.post("/api/documents")
def upload_document() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "Attach a PDF file in the file field."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename or not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF documents are supported."}), 400

    started = time.perf_counter()
    temp_path = ""
    try:
        vector_store, processor, _ = _get_services()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            uploaded.save(temp_file)
            temp_path = temp_file.name
        chunks = processor.process_document(temp_path)
        vector_store.add_documents(chunks)
        document = {
            "name": uploaded.filename,
            "size": os.path.getsize(temp_path),
            "chunks": len(chunks),
            "status": "indexed",
        }
        _documents[:] = [item for item in _documents if item["name"] != uploaded.filename]
        _documents.append(document)
        return jsonify({"document": document, "state": _snapshot(), "elapsedMs": round((time.perf_counter() - started) * 1000)})
    except ModuleNotFoundError as error:
        return jsonify({
            "error": "Document processing dependencies are not installed. Run: python -m pip install -r requirements.txt",
            "detail": str(error),
        }), 503
    except Exception as error:
        return jsonify({"error": f"Could not index the document: {error}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Ask a question to search your documents."}), 400
    if not _documents:
        return jsonify({"error": "Upload and index a document before asking a question."}), 400

    vector_store, _, rag = _get_services()
    retrieval_started = time.perf_counter()
    sources = vector_store.similarity_search(query, k=4)
    retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000)
    generation_started = time.perf_counter()
    answer = rag.generate_answer(query)
    generation_ms = round((time.perf_counter() - generation_started) * 1000)
    return jsonify({
        "answer": answer,
        "sources": [
            {
                "name": source.metadata.get("source", "Unknown document"),
                "page": source.metadata.get("page", "Unknown"),
                "preview": source.page_content[:220].replace("\n", " "),
            }
            for source in sources
        ],
        "trace": {
            "retrievalMs": retrieval_ms,
            "generationMs": generation_ms,
            "chunks": len(sources),
        },
    })


@app.post("/api/reset")
def reset() -> Any:
    global _vector_store, _rag
    from rag import RAG
    from vector_store import VectorStore

    with _state_lock:
        _vector_store = VectorStore()
        _rag = RAG(_vector_store)
        _documents.clear()
    return jsonify(_snapshot())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("DEBUG", "0") == "1")
