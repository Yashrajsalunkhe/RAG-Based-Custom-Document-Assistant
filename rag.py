import os
from typing import Any, Optional, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional until Gemini is selected
    ChatGoogleGenerativeAI = None

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - compatibility for older LangChain versions
    from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain.schema import Document

load_dotenv()

class RAG:
    """Retrieval Augmented Generation for question answering."""
    
    def __init__(self, vector_store, temperature: float = 0.7):
        """
        Initialize the RAG system.

        Args:
            vector_store: Vector store instance
            temperature: Model temperature for response generation
        """
        self.vector_store = vector_store
        self.temperature = temperature
        self.model: Optional[Any] = None
        self.provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        self.model_error: Optional[str] = None

        try:
            self.model = self._get_model()
        except Exception as exc:  # pragma: no cover - normal when the local model is unavailable
            self.model_error = str(exc)
            self.model = None

    def _get_model(self) -> Any:
        """Create the configured local Ollama or Google Gemini client."""
        if self.model is not None:
            return self.model

        if self.provider in {"local", "ollama"}:
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api_key = os.getenv("OPENAI_API_KEY", "ollama")

            # Verify Ollama is reachable before creating the client
            import urllib.request
            import urllib.error
            try:
                health_url = base_url.replace("/v1", "")
                req = urllib.request.Request(health_url, method="GET")
                urllib.request.urlopen(req, timeout=3)
            except (urllib.error.URLError, OSError) as conn_err:
                raise RuntimeError(
                    f"Cannot reach the Ollama server at {base_url}. "
                    "Start it with: ollama serve"
                ) from conn_err

            self.model = ChatOpenAI(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=self.temperature,
            )
            return self.model

        if self.provider == "gemini":
            if ChatGoogleGenerativeAI is None:
                raise RuntimeError(
                    "Gemini support is not installed. Run: "
                    "python -m pip install -r requirements.txt"
                )
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Set GOOGLE_API_KEY (or GEMINI_API_KEY) when LLM_PROVIDER=gemini."
                )
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=self.temperature,
            )
            return self.model

        raise ValueError(
            f"Unsupported LLM_PROVIDER '{self.provider}'. Use 'ollama' or 'gemini'."
        )
        
    def generate_answer(self, query: str, k: int = 4) -> str:
        """
        Generate an answer to a query using RAG.
        
        Args:
            query: User query
            k: Number of top documents to retrieve
            
        Returns:
            Generated answer
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant information found. Please try a different question or upload documents."
        
        # Format context from documents
        context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}" for doc in docs])
        
        # Create prompt for the model
        messages = [
            SystemMessage(content=f"You are a helpful assistant that answers questions based on the provided documents. Use only the information in the documents to answer the question. If the answer cannot be found in the documents, say so directly.\n\nDocuments:\n{context}"),
            HumanMessage(content=query)
        ]
        
        # Generate response
        try:
            if self.model is None:
                model = self._get_model()
            else:
                model = self.model

            response = model.invoke(messages)
            if isinstance(response, BaseMessage):
                return cast(str, response.content)
            if hasattr(response, "content"):
                return cast(str, response.content)
            return cast(str, response)
        except Exception as e:
            error_str = str(e)
            # Provide actionable error messages
            if "ollama" in self.provider:
                return (
                    "Cannot connect to Ollama. Please ensure Ollama is running "
                    "with: ollama serve\n\nThen try again or switch to Gemini using "
                    "the model selector."
                )
            if self.model_error:
                return (
                    "Unable to connect to the configured language model. "
                    f"Please ensure the model server is running: {self.model_error}"
                )
            return f"Error generating response: {error_str}"