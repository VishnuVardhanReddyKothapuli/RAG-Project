"""
RAG Application — Sources UI Component
=========================================
Displays retrieved source chunks with metadata and similarity scores.
Provides search transparency before and after generation.
"""

import streamlit as st

from utils.helpers import get_similarity_color
from utils.logger import get_logger

logger = get_logger(__name__)


def render_sources():
    """Render the source viewer panel showing retrieved chunks."""

    st.markdown("### 🔍 Retrieved Sources")

    last_sources = None
    chat_history = st.session_state.get("chat_history", [])

    # Get sources from the last assistant message
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant" and "sources" in msg:
            last_sources = msg["sources"]
            break

    if not last_sources:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 2rem;
                border: 1px dashed #333;
                border-radius: 10px;
                color: #666;
            ">
                <p style="font-size: 1.5rem;">🔍</p>
                <p>Sources will appear here after you ask a question.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Display each source
    for i, src in enumerate(last_sources, 1):
        score = src.get("score", 0)
        color = get_similarity_color(score)
        score_pct = max(0, min(100, int(score * 100)))

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1a1d23, #1e2129);
                    border: 1px solid #2a2d35;
                    border-left: 4px solid {color};
                    border-radius: 0 10px 10px 0;
                    padding: 1rem 1.2rem;
                    margin-bottom: 0.8rem;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-weight: 600; color: #fafafa;">
                            📄 Chunk {i}
                        </span>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span style="
                                background: {color}20;
                                color: {color};
                                padding: 3px 10px;
                                border-radius: 20px;
                                font-size: 0.75rem;
                                font-weight: 700;
                            ">{score_pct}%</span>
                        </div>
                    </div>
                    <div style="
                        display: flex;
                        gap: 1rem;
                        margin-bottom: 0.5rem;
                        font-size: 0.8rem;
                        color: #888;
                    ">
                        <span>📁 {src.get('source', 'Unknown')}</span>
                        <span>📃 Page {src.get('page', 'N/A')}</span>
                    </div>
                    <p style="
                        color: #ccc;
                        font-size: 0.85rem;
                        line-height: 1.6;
                        margin: 0;
                        padding: 0.5rem;
                        background: #0e1117;
                        border-radius: 6px;
                    ">{src.get('snippet', '')[:400]}{'…' if len(src.get('snippet', '')) > 400 else ''}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Similarity score legend
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            padding: 0.5rem;
            font-size: 0.75rem;
            color: #888;
        ">
            <span><span style="color: #00d4aa;">●</span> High (≥80%)</span>
            <span><span style="color: #f0b429;">●</span> Medium (50-79%)</span>
            <span><span style="color: #e53e3e;">●</span> Low (<50%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
