"""
RAG Application — RAG Chain
==============================
Core chain that orchestrates retrieval, context assembly,
prompt construction, and LLM generation with streaming.
"""

from typing import Generator, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from config import settings
from utils.logger import get_logger
from utils.helpers import format_chat_history

logger = get_logger(__name__)


class RAGChain:
    """
    Orchestrates the RAG pipeline: retrieval → context assembly →
    LLM generation with strict grounding and source citations.
    """

    def __init__(self, retriever):
        """
        Initialize the RAG chain.

        Args:
            retriever: A HybridRetriever instance.
        """
        self._retriever = retriever

        self._llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=settings.LLM_STREAMING,
        )

        logger.info(
            f"RAGChain initialized — model: {settings.LLM_MODEL}, "
            f"temp: {settings.LLM_TEMPERATURE}"
        )

    def _build_context(self, documents: list[Document]) -> str:
        """
        Build a context string from retrieved documents.

        Args:
            documents: List of retrieved Document objects.

        Returns:
            Formatted context string with source annotations.
        """
        if not documents:
            return "No relevant context was found in the uploaded documents."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Document {i} | Source: {source} | Page: {page}]\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
    ) -> list:
        """
        Build the message list for the LLM.

        Args:
            query: User's question.
            context: Assembled context from retrieval.
            chat_history: Previous conversation turns.

        Returns:
            List of LangChain message objects.
        """
        messages = []

        # System message with context injected
        system_content = settings.SYSTEM_PROMPT.format(context=context)
        messages.append(SystemMessage(content=system_content))

        # Add conversation history (limited window)
        if chat_history:
            for msg in chat_history[-(settings.MEMORY_WINDOW_SIZE * 2):]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Current user query
        messages.append(HumanMessage(content=query))

        return messages

    def ask(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Process a user query through the full RAG pipeline (non-streaming).

        Args:
            query: User's natural language question.
            chat_history: Previous conversation turns.

        Returns:
            Dict with keys: 'answer', 'sources', 'retrieved_chunks',
                            'context', 'rewritten_query'
        """
        try:
            # Step 1: Retrieve relevant documents
            docs_with_scores = self._retriever.retrieve_with_scores(query)
            documents = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]

            logger.info(f"Retrieved {len(documents)} chunks for: '{query[:80]}'")

            # Step 2: Build context
            context = self._build_context(documents)

            # Step 3: Build messages
            messages = self._build_messages(query, context, chat_history)

            # Step 4: Generate answer
            response = self._llm.invoke(messages)
            answer = response.content

            # Step 5: Build source metadata
            sources = []
            for doc, score in zip(documents, scores):
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:300],
                    "score": round(score, 4),
                })

            logger.info(f"Generated answer ({len(answer)} chars) with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": documents,
                "context": context,
            }

        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
                "context": "",
            }

    def ask_stream(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[Generator, list[dict], list[Document]]:
        """
        Process a user query and return a streaming generator.

        Args:
            query: User's natural language question.
            chat_history: Previous conversation turns.

        Returns:
            Tuple of:
            - Generator yielding answer token strings
            - List of source dicts
            - List of retrieved Document objects
        """
        try:
            # Step 1: Retrieve relevant documents
            docs_with_scores = self._retriever.retrieve_with_scores(query)
            documents = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]

            # Step 2: Build context
            context = self._build_context(documents)

            # Step 3: Build messages
            messages = self._build_messages(query, context, chat_history)

            # Step 4: Build source metadata
            sources = []
            for doc, score in zip(documents, scores):
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:300],
                    "score": round(score, 4),
                })

            # Step 5: Create streaming generator
            def token_generator():
                for chunk in self._llm.stream(messages):
                    if chunk.content:
                        yield chunk.content

            logger.info(
                f"Streaming response for: '{query[:80]}' "
                f"with {len(sources)} sources"
            )

            return token_generator(), sources, documents

        except Exception as e:
            logger.error(f"RAG stream failed: {e}")

            def error_generator():
                yield f"An error occurred: {str(e)}"

            return error_generator(), [], []
