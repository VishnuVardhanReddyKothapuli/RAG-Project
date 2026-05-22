"""
RAG Application — Chat UI Component
=======================================
Chat interface with streaming responses, message history,
and integrated source display.
"""

import streamlit as st

from utils.logger import get_logger
from utils.helpers import format_chat_history

logger = get_logger(__name__)


def render_chat():
    """Render the main chat interface."""

    # ---- Header ----
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="
                background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7c3aed 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            ">Document Intelligence Chat</h2>
            <p style="color: #888; font-size: 0.95rem;">
                Ask questions about your uploaded documents — answers are grounded in your PDFs only.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Check if documents are uploaded ----
    vector_store = st.session_state.get("vector_store")
    if vector_store and vector_store.get_document_count() == 0:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 3rem 2rem;
                border: 2px dashed #333;
                border-radius: 12px;
                margin: 2rem 0;
            ">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">📄</p>
                <p style="color: #aaa; font-size: 1.1rem;">
                    Upload PDF documents in the sidebar to get started.
                </p>
                <p style="color: #666; font-size: 0.85rem;">
                    Your documents will be processed, chunked, and indexed for intelligent retrieval.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Chat History ----
    chat_history = st.session_state.get("chat_history", [])

    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(content)

            # Show sources inline if available
            if role == "assistant" and "sources" in msg:
                _render_inline_sources(msg["sources"])

    # ---- Chat Input ----
    if prompt := st.chat_input("Ask a question about your documents..."):
        _handle_user_query(prompt)


def _handle_user_query(query: str):
    """Process a user query through the RAG pipeline."""
    from retriever.query_rewriter import QueryRewriter
    from ui.sources import render_sources

    # Check for documents
    vector_store = st.session_state.get("vector_store")
    if not vector_store or vector_store.get_document_count() == 0:
        st.warning("⚠️ Please upload documents first before asking questions.")
        return

    chat_history = st.session_state.get("chat_history", [])

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Add to history
    chat_history.append({"role": "user", "content": query})

    # ---- Query Rewriting ----
    rewritten_query = query
    if len(chat_history) > 2:  # Only rewrite if there's history
        try:
            rewriter = st.session_state.get("query_rewriter")
            if rewriter:
                history_str = format_chat_history(chat_history[:-1])
                rewritten_query = rewriter.rewrite(query, history_str)

                if rewritten_query != query:
                    st.caption(f"🔄 Reformulated: *{rewritten_query}*")
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")

    # ---- Retrieval & Generation ----
    rag_chain = st.session_state.get("rag_chain")
    if not rag_chain:
        st.error("RAG chain not initialized. Please refresh the app.")
        return

    with st.chat_message("assistant", avatar="🤖"):
        # Show retrieval phase
        with st.status("🔍 Searching documents...", expanded=False) as status:
            try:
                token_gen, sources, retrieved_docs = rag_chain.ask_stream(
                    rewritten_query, chat_history
                )
                status.update(
                    label=f"✅ Found {len(sources)} relevant chunks",
                    state="complete",
                )
            except Exception as e:
                status.update(label="❌ Retrieval failed", state="error")
                st.error(f"Error: {e}")
                return

        # Stream the response
        full_response = st.write_stream(token_gen)

        # Show sources
        if sources:
            _render_inline_sources(sources)

    # Add assistant response to history
    chat_history.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
        "retrieved_chunks": [doc.page_content for doc in retrieved_docs],
    })

    st.session_state["chat_history"] = chat_history

    # Store last response for evaluation
    st.session_state["last_query"] = rewritten_query
    st.session_state["last_answer"] = full_response
    st.session_state["last_contexts"] = [doc.page_content for doc in retrieved_docs]


def _render_inline_sources(sources: list[dict]):
    """Render source citations inline below an answer."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} chunks retrieved)", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            score_pct = max(0, min(100, int(score * 100)))

            # Color based on score
            if score >= 0.8:
                color = "#00d4aa"
            elif score >= 0.5:
                color = "#f0b429"
            else:
                color = "#e53e3e"

            st.markdown(
                f"""
                <div style="
                    background: #1a1d23;
                    border-left: 3px solid {color};
                    padding: 0.8rem 1rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #00d4aa; font-weight: 600;">
                            📄 {src.get('source', 'Unknown')} — Page {src.get('page', 'N/A')}
                        </span>
                        <span style="
                            background: {color}22;
                            color: {color};
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 600;
                        ">{score_pct}% match</span>
                    </div>
                    <p style="color: #aaa; font-size: 0.82rem; margin-top: 0.4rem; line-height: 1.5;">
                        {src.get('snippet', '')[:250]}{'…' if len(src.get('snippet', '')) > 250 else ''}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_controls():
    """Render chat control buttons (clear, etc.)."""
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["eval_scores"] = []
            st.session_state.pop("last_query", None)
            st.session_state.pop("last_answer", None)
            st.session_state.pop("last_contexts", None)
            st.rerun()
