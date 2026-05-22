"""
RAG Application — Centralized Configuration
=============================================
All configurable parameters in one place.
Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1          # Low temperature for factual answers
LLM_MAX_TOKENS: int = 2048
LLM_STREAMING: bool = True


# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = "cpu"          # Set to "cuda" if GPU is available


# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000                 # Characters per chunk
CHUNK_OVERLAP: int = 200              # Overlap between consecutive chunks
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = str(_PROJECT_ROOT / "data" / "chroma_db")
CHROMA_COLLECTION_NAME: str = "rag_documents"


# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
TOP_K: int = 5                         # Number of chunks to retrieve
SEMANTIC_WEIGHT: float = 0.7           # Weight for semantic (vector) search
BM25_WEIGHT: float = 0.3              # Weight for keyword (BM25) search
ENABLE_HYBRID_SEARCH: bool = True      # Toggle hybrid vs. pure semantic


# ---------------------------------------------------------------------------
# Conversational Memory
# ---------------------------------------------------------------------------
MEMORY_WINDOW_SIZE: int = 10           # Number of past exchanges to keep


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------
ENABLE_AUTO_EVAL: bool = False         # Auto-evaluate every response
RAGAS_LLM_MODEL: str = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: str = str(_PROJECT_ROOT / "logs")
LOG_LEVEL: str = "INFO"
LOG_FILE: str = "rag_app.log"


# ---------------------------------------------------------------------------
# Upload Limits
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = 200
ALLOWED_EXTENSIONS: list[str] = [".pdf"]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are a document-grounded AI assistant.

STRICT RULES:
1. Answer ONLY from the provided context below.
2. Do NOT use any outside knowledge or make assumptions.
3. If the answer is not present in the context, respond EXACTLY with:
   "The answer was not found in the uploaded documents."
4. Always cite the source document name and page number in your answer.
5. Format citations as: [Source: filename.pdf, Page X]
6. Be concise, accurate, and helpful.

CONTEXT:
{context}
"""


QUERY_REWRITE_PROMPT: str = """Given the following conversation history and a new question, 
reformulate the question into a standalone, specific question that can be understood 
without the conversation history. If the question is already clear, return it as-is.

Chat History:
{chat_history}

New Question: {question}

Reformulated Question:"""
