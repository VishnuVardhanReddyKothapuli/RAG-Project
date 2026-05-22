"""
RAG Application — Metrics UI Component
=========================================
Displays RAGAS evaluation metrics with visual gauges
and historical score tracking.
"""

import streamlit as st

from utils.helpers import get_metric_color
from utils.logger import get_logger

logger = get_logger(__name__)


def render_metrics():
    """Render the RAGAS evaluation metrics panel."""

    st.markdown("### 📊 RAG Evaluation")

    # Check if we have a recent response to evaluate
    last_query = st.session_state.get("last_query")
    last_answer = st.session_state.get("last_answer")
    last_contexts = st.session_state.get("last_contexts")

    if not all([last_query, last_answer, last_contexts]):
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 2rem;
                border: 1px dashed #333;
                border-radius: 10px;
                color: #666;
            ">
                <p style="font-size: 1.5rem;">📊</p>
                <p>Evaluation scores will appear here after a Q&A interaction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Evaluate button
    if st.button("🧪 Run RAGAS Evaluation", type="primary", use_container_width=True):
        _run_evaluation(last_query, last_answer, last_contexts)

    # Display existing scores
    eval_scores = st.session_state.get("eval_scores", [])
    if eval_scores:
        latest = eval_scores[-1]
        _render_score_cards(latest)

        # History chart
        if len(eval_scores) > 1:
            _render_score_history(eval_scores)


def _run_evaluation(query: str, answer: str, contexts: list[str]):
    """Run RAGAS evaluation on the last Q&A pair."""
    from evaluation.ragas_evaluator import RAGASEvaluator

    evaluator = st.session_state.get("evaluator")
    if not evaluator:
        evaluator = RAGASEvaluator()
        st.session_state["evaluator"] = evaluator

    with st.spinner("🧪 Running RAGAS evaluation... This may take a moment."):
        scores = evaluator.evaluate_single(
            question=query,
            answer=answer,
            contexts=contexts,
        )

    # Store scores
    eval_scores = st.session_state.get("eval_scores", [])
    eval_scores.append({
        "query": query[:60],
        **scores,
    })
    st.session_state["eval_scores"] = eval_scores

    logger.info(f"RAGAS evaluation complete: {scores}")
    st.rerun()


def _render_score_cards(scores: dict):
    """Render metric score cards."""

    metric_info = {
        "faithfulness": {
            "label": "🎯 Faithfulness",
            "desc": "Is the answer grounded in context?",
        },
        "answer_relevancy": {
            "label": "📊 Answer Relevancy",
            "desc": "Does the answer address the question?",
        },
        "context_precision": {
            "label": "🔍 Context Precision",
            "desc": "Are relevant chunks ranked higher?",
        },
    }

    cols = st.columns(len(metric_info))

    for col, (metric_key, info) in zip(cols, metric_info.items()):
        score = scores.get(metric_key, -1.0)

        with col:
            if score < 0:
                display_score = "N/A"
                color = "#666"
                pct = 0
            else:
                display_score = f"{score:.0%}"
                color = get_metric_color(score)
                pct = int(score * 100)

            st.markdown(
                f"""
                <div style="
                    background: #1a1d23;
                    border: 1px solid #2a2d35;
                    border-radius: 12px;
                    padding: 1.2rem;
                    text-align: center;
                ">
                    <p style="font-size: 0.85rem; color: #aaa; margin-bottom: 0.3rem;">
                        {info['label']}
                    </p>
                    <p style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: {color};
                        margin: 0.3rem 0;
                    ">{display_score}</p>
                    <div style="
                        background: #0e1117;
                        border-radius: 10px;
                        height: 6px;
                        margin: 0.5rem 0;
                        overflow: hidden;
                    ">
                        <div style="
                            background: {color};
                            height: 100%;
                            width: {pct}%;
                            border-radius: 10px;
                            transition: width 0.5s ease;
                        "></div>
                    </div>
                    <p style="font-size: 0.7rem; color: #666; margin: 0;">
                        {info['desc']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_score_history(eval_scores: list[dict]):
    """Render a line chart of historical evaluation scores."""

    st.markdown("#### 📈 Score History")

    import pandas as pd

    # Build DataFrame
    rows = []
    for i, scores in enumerate(eval_scores, 1):
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            val = scores.get(metric, -1.0)
            if val >= 0:
                rows.append({
                    "Query #": i,
                    "Metric": metric.replace("_", " ").title(),
                    "Score": val,
                })

    if rows:
        df = pd.DataFrame(rows)
        # Pivot for line chart
        pivot_df = df.pivot(index="Query #", columns="Metric", values="Score")
        st.line_chart(pivot_df, use_container_width=True)
    else:
        st.caption("No valid scores to display yet.")
