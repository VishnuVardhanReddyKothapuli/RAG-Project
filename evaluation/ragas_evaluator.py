"""
RAG Application — RAGAS Evaluator
====================================
Evaluates RAG pipeline quality using RAGAS metrics:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are relevant chunks ranked higher?
"""

from typing import Optional

from datasets import Dataset

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using the RAGAS framework.
    Uses Groq LLM as the evaluation judge.
    """

    def __init__(self):
        self._initialized = False
        self._metrics = None
        self._llm = None
        self._embeddings = None
        logger.info("RAGASEvaluator created (lazy initialization)")

    def _initialize(self) -> bool:
        """
        Lazily initialize RAGAS components.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return True

        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )
            from langchain_groq import ChatGroq
            from embeddings.embedding_manager import get_embeddings

            self._llm = ChatGroq(
                model=settings.RAGAS_LLM_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.0,
            )

            self._embeddings = get_embeddings()

            self._metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]

            self._metric_names = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
            ]

            self._initialized = True
            logger.info("RAGAS evaluator initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"RAGAS import failed — install with: pip install ragas. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"RAGAS initialization failed: {e}")
            return False

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a single question-answer pair.

        Args:
            question: The user's question.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            ground_truth: Optional ground truth answer (for context_recall).

        Returns:
            Dict with metric names as keys and scores (0-1) as values.
            Returns scores of -1.0 if evaluation fails.
        """
        if not self._initialize():
            return {name: -1.0 for name in ["faithfulness", "answer_relevancy", "context_precision"]}

        try:
            from ragas import evaluate

            # Build the evaluation dataset
            eval_data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }

            # Add ground truth if provided (needed for context_recall)
            if ground_truth:
                eval_data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(eval_data)

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )

            scores = {}
            for metric_name in self._metric_names:
                score = result.get(metric_name, -1.0)
                if score is None or (isinstance(score, float) and score != score):  # NaN check
                    score = -1.0
                scores[metric_name] = round(float(score), 4)

            logger.info(f"RAGAS evaluation complete: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {name: -1.0 for name in self._metric_names}

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
    ) -> dict:
        """
        Evaluate a batch of question-answer pairs.

        Args:
            questions: List of questions.
            answers: List of generated answers.
            contexts_list: List of context lists (one per question).

        Returns:
            Dict with metric names and their average scores.
        """
        if not self._initialize():
            return {name: -1.0 for name in ["faithfulness", "answer_relevancy", "context_precision"]}

        try:
            from ragas import evaluate

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
            })

            result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
            )

            scores = {}
            for metric_name in self._metric_names:
                score = result.get(metric_name, -1.0)
                if score is None or (isinstance(score, float) and score != score):
                    score = -1.0
                scores[metric_name] = round(float(score), 4)

            logger.info(f"Batch RAGAS evaluation ({len(questions)} samples): {scores}")
            return scores

        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            return {name: -1.0 for name in self._metric_names}

    @staticmethod
    def format_scores(scores: dict) -> str:
        """
        Format evaluation scores into a readable string.

        Args:
            scores: Dict of metric name → score.

        Returns:
            Formatted multi-line string.
        """
        labels = {
            "faithfulness": "🎯 Faithfulness",
            "answer_relevancy": "📊 Answer Relevancy",
            "context_precision": "🔍 Context Precision",
            "context_recall": "📈 Context Recall",
        }

        lines = []
        for metric, score in scores.items():
            label = labels.get(metric, metric)
            if score < 0:
                lines.append(f"{label}: N/A")
            else:
                lines.append(f"{label}: {score:.2%}")

        return "\n".join(lines)
