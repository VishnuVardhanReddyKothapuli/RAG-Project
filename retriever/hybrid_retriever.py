"""
RAG Application — Hybrid Retriever
=====================================
Combines semantic (vector) search with keyword (BM25) search
using LangChain's EnsembleRetriever with Reciprocal Rank Fusion.
"""

from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that combines ChromaDB semantic search
    with BM25 keyword search for improved retrieval quality.
    """

    def __init__(self, vector_store):
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: ChromaVectorStore instance.
        """
        self._vector_store = vector_store
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._ensemble_retriever: Optional[EnsembleRetriever] = None
        self._documents_indexed = False

        logger.info(
            f"HybridRetriever initialized — "
            f"hybrid={settings.ENABLE_HYBRID_SEARCH}, "
            f"semantic_weight={settings.SEMANTIC_WEIGHT}, "
            f"bm25_weight={settings.BM25_WEIGHT}"
        )

    def build_bm25_index(self, documents: Optional[List[Document]] = None) -> None:
        """
        Build or rebuild the BM25 index from documents.

        Args:
            documents: List of Documents to index. If None, retrieves
                       all documents from the vector store.
        """
        if not settings.ENABLE_HYBRID_SEARCH:
            logger.info("Hybrid search disabled — skipping BM25 index build")
            return

        try:
            if documents is None:
                documents = self._vector_store.get_all_documents()

            if not documents:
                logger.warning("No documents available for BM25 indexing")
                self._bm25_retriever = None
                self._documents_indexed = False
                return

            self._bm25_retriever = BM25Retriever.from_documents(
                documents, k=settings.TOP_K
            )
            self._documents_indexed = True
            logger.info(f"BM25 index built with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._bm25_retriever = None
            self._documents_indexed = False

    def get_retriever(self, k: Optional[int] = None):
        """
        Get the ensemble retriever (or fallback to vector-only).

        Args:
            k: Number of documents to retrieve.

        Returns:
            An EnsembleRetriever or VectorStoreRetriever.
        """
        k = k or settings.TOP_K
        vector_retriever = self._vector_store.get_retriever(k=k)

        # If hybrid search is enabled and BM25 index is ready
        if (
            settings.ENABLE_HYBRID_SEARCH
            and self._bm25_retriever is not None
            and self._documents_indexed
        ):
            self._bm25_retriever.k = k
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=[settings.SEMANTIC_WEIGHT, settings.BM25_WEIGHT],
            )
            logger.debug("Using hybrid (ensemble) retriever")
            return self._ensemble_retriever

        logger.debug("Using vector-only retriever (BM25 not available)")
        return vector_retriever

    def retrieve(
        self, query: str, k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's search query.
            k: Number of documents to retrieve.

        Returns:
            List of relevant Document objects.
        """
        retriever = self.get_retriever(k=k)

        try:
            docs = retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents for query: '{query[:80]}...'")
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.

        Falls back to ChromaDB's native scored search since
        EnsembleRetriever doesn't directly expose scores.

        Args:
            query: User's search query.
            k: Number of documents to retrieve.

        Returns:
            List of (Document, score) tuples.
        """
        return self._vector_store.similarity_search_with_scores(query, k=k)
