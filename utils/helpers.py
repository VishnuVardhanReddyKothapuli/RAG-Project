"""
RAG Application — Helper Utilities
====================================
Shared helper functions used across the application.
"""

import hashlib
import re
from typing import List

from langchain_core.documents import Document


def format_sources(docs: List[Document]) -> str:
    """
    Format a list of retrieved Documents into a human-readable source string.

    Args:
        docs: List of LangChain Document objects with metadata.

    Returns:
        Formatted string listing each source with file name, page, and snippet.
    """
    if not docs:
        return "No sources retrieved."

    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        snippet = truncate_text(doc.page_content, max_len=200)
        lines.append(f"**Source {i}:** {source} — Page {page}\n> {snippet}")

    return "\n\n".join(lines)


def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len characters, appending '…' if truncated."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Compute a SHA-256 hash of file contents for deduplication.

    Args:
        file_bytes: Raw bytes of the file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def sanitize_filename(name: str) -> str:
    """
    Remove or replace characters that are unsafe for file names.

    Args:
        name: Original file name.

    Returns:
        Sanitized file name.
    """
    # Remove anything that isn't alphanumeric, dash, underscore, dot, or space
    sanitized = re.sub(r"[^\w\s\-.]", "", name)
    # Collapse multiple spaces / underscores
    sanitized = re.sub(r"[\s_]+", "_", sanitized).strip("_")
    return sanitized


def format_chat_history(chat_history: list[dict]) -> str:
    """
    Format chat history into a string for the query rewriter.

    Args:
        chat_history: List of dicts with 'role' and 'content' keys.

    Returns:
        Formatted string of the conversation.
    """
    if not chat_history:
        return "No previous conversation."

    lines: list[str] = []
    for msg in chat_history[-10:]:  # Last 10 messages
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def get_similarity_color(score: float) -> str:
    """
    Return a CSS color based on similarity score.

    Args:
        score: Similarity score between 0 and 1.

    Returns:
        Hex color string.
    """
    if score >= 0.8:
        return "#00d4aa"   # Teal-green — high relevance
    elif score >= 0.5:
        return "#f0b429"   # Amber — moderate relevance
    else:
        return "#e53e3e"   # Red — low relevance


def get_metric_color(score: float) -> str:
    """
    Return a CSS color for RAGAS metric scores.

    Args:
        score: RAGAS metric score between 0 and 1.

    Returns:
        Hex color string.
    """
    if score >= 0.8:
        return "#00d4aa"
    elif score >= 0.5:
        return "#f0b429"
    else:
        return "#e53e3e"
