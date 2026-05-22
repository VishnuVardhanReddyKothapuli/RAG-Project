"""
RAG Application — ChromaDB Vector Store
==========================================
Handles all interactions with ChromaDB: indexing, retrieval,
deletion, and document management.
"""

import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from embeddings.embedding_manager import get_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Wrapper around ChromaDB for document indexing and retrieval.

    Provides methods to add, search, delete, and manage documents
    in a persistent ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._embeddings = get_embeddings()

        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize the text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        # Initialize or load existing Chroma collection
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )

        logger.info(
            f"ChromaDB initialized — collection: '{self.collection_name}', "
            f"persist: '{self.persist_directory}', "
            f"existing docs: {self.get_document_count()}"
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Split documents into chunks and add them to the vector store.

        Args:
            documents: List of LangChain Document objects (page-level).

        Returns:
            Number of chunks added to the store.
        """
        if not documents:
            logger.warning("No documents provided to add")
            return 0

        # Split into chunks while preserving metadata
        chunks = self._splitter.split_documents(documents)

        # Add chunk index to metadata for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(
            f"Splitting {len(documents)} pages into {len(chunks)} chunks "
            f"(size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})"
        )

        # Add to ChromaDB in batches to avoid memory issues
        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._vectorstore.add_documents(batch)
            logger.debug(
                f"Added batch {start // batch_size + 1}: "
                f"{len(batch)} chunks"
            )

        logger.info(f"Successfully indexed {len(chunks)} chunks")
        return len(chunks)

    def get_retriever(self, k: Optional[int] = None):
        """
        Get a LangChain retriever from the vector store.

        Args:
            k: Number of documents to retrieve. Defaults to settings.TOP_K.

        Returns:
            A VectorStoreRetriever instance.
        """
        k = k or settings.TOP_K
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def similarity_search_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.

        Args:
            query: Search query string.
            k: Number of results. Defaults to settings.TOP_K.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        k = k or settings.TOP_K
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )
        logger.debug(f"Similarity search returned {len(results)} results")
        return results

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the vector store.

        Returns:
            List of all Document objects in the collection.
        """
        try:
            collection = self._vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])

            documents = []
            if result["documents"] and result["metadatas"]:
                for doc_text, metadata in zip(result["documents"], result["metadatas"]):
                    documents.append(
                        Document(page_content=doc_text, metadata=metadata or {})
                    )

            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve all documents: {e}")
            return []

    def delete_documents_by_source(self, source_name: str) -> int:
        """
        Delete all chunks from a specific source PDF.

        Args:
            source_name: The 'source' metadata value (filename) to delete.

        Returns:
            Number of documents deleted.
        """
        try:
            collection = self._vectorstore._collection
            # Get IDs of documents matching the source
            result = collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )

            ids_to_delete = result["ids"]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} chunks from source '{source_name}'"
                )
                return len(ids_to_delete)
            else:
                logger.info(f"No chunks found for source '{source_name}'")
                return 0

        except Exception as e:
            logger.error(f"Failed to delete documents for '{source_name}': {e}")
            return 0

    def list_sources(self) -> List[str]:
        """
        List all unique source file names in the vector store.

        Returns:
            Sorted list of unique source names.
        """
        try:
            collection = self._vectorstore._collection
            result = collection.get(include=["metadatas"])

            sources = set()
            if result["metadatas"]:
                for metadata in result["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])

            return sorted(sources)

        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []

    def get_document_count(self) -> int:
        """
        Get the total number of chunks in the vector store.

        Returns:
            Integer count of stored chunks.
        """
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    def clear_collection(self) -> None:
        """Delete the entire collection and reinitialize."""
        try:
            self._vectorstore.delete_collection()
            logger.info(f"Cleared collection '{self.collection_name}'")

            # Reinitialize
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("Collection reinitialized")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
