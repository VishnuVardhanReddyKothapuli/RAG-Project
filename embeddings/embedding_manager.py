"""
RAG Application — Embedding Manager
=====================================
Manages the HuggingFace embedding model with singleton pattern
to avoid reloading the model on every call.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get or create a cached HuggingFace embedding model instance.

    Uses the model specified in config.settings.EMBEDDING_MODEL.
    The model is loaded once and cached for subsequent calls.

    Returns:
        HuggingFaceEmbeddings instance ready for encoding.
    """
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,   # L2 normalize for cosine similarity
                "batch_size": 64,               # Optimize batch processing
            },
        )
        logger.info("Embedding model loaded successfully")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(
            f"Could not load embedding model '{settings.EMBEDDING_MODEL}'. "
            f"Ensure sentence-transformers is installed. Error: {e}"
        ) from e
