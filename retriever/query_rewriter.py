"""
RAG Application — Query Rewriter
==================================
Reformulates vague or follow-up queries into standalone questions
using the LLM, incorporating conversation history.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """
    Rewrites user queries for better retrieval, especially
    for follow-up questions that depend on chat history.
    """

    def __init__(self):
        self._llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=256,
        )

        self._prompt = ChatPromptTemplate.from_template(
            settings.QUERY_REWRITE_PROMPT
        )

        self._chain = self._prompt | self._llm | StrOutputParser()
        logger.info("QueryRewriter initialized")

    def rewrite(self, question: str, chat_history: str = "") -> str:
        """
        Rewrite a user question into a standalone query.

        Args:
            question: The user's latest question.
            chat_history: Formatted string of previous conversation turns.

        Returns:
            Reformulated standalone question.
        """
        # If no chat history, return as-is (no need to rewrite)
        if not chat_history or chat_history == "No previous conversation.":
            logger.debug(f"No history — returning original query: '{question[:80]}'")
            return question

        try:
            rewritten = self._chain.invoke({
                "question": question,
                "chat_history": chat_history,
            })
            rewritten = rewritten.strip()

            if rewritten:
                logger.info(
                    f"Query rewritten: '{question[:60]}' → '{rewritten[:60]}'"
                )
                return rewritten
            else:
                logger.warning("Query rewriter returned empty — using original")
                return question

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return question  # Graceful fallback
