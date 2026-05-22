"""
RAG Application — Sidebar UI Component
=========================================
Handles PDF upload, document management, and settings.
"""

import streamlit as st
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def render_sidebar():
    """Render the sidebar with upload, management, and settings panels."""

    with st.sidebar:
        # ---- Header ----
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="
                    font-size: 1.8rem;
                    background: linear-gradient(135deg, #00d4aa, #00b4d8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.2rem;
                ">📄 RAG Assistant</h1>
                <p style="color: #888; font-size: 0.85rem;">
                    Upload PDFs • Ask Questions • Get Grounded Answers
                </p>
            </div>
            <hr style="border-color: #333; margin: 0.5rem 0 1rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # ---- PDF Upload ----
        st.markdown("### 📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Drop your PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help=f"Max file size: {settings.MAX_FILE_SIZE_MB}MB per file",
        )

        if uploaded_files:
            if st.button("📥 Process & Index", type="primary", use_container_width=True):
                _process_uploads(uploaded_files)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Document Management ----
        _render_document_management()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Settings ----
        _render_settings()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Footer ----
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0; border-top: 1px solid #333;">
                <p style="color: #666; font-size: 0.75rem;">
                    Built with LangChain • ChromaDB • Groq
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _process_uploads(uploaded_files):
    """Process and index uploaded PDF files."""
    from loaders.pdf_loader import load_uploaded_pdfs

    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        st.error("⚠️ Vector store not initialized. Please refresh the app.")
        return

    existing_hashes = st.session_state.get("file_hashes", set())

    with st.spinner("📖 Extracting text from PDFs..."):
        documents, updated_hashes, skipped = load_uploaded_pdfs(
            uploaded_files, existing_hashes
        )

    if skipped:
        for name in skipped:
            st.warning(f"⚠️ Skipped: {name} (duplicate or unreadable)")

    if documents:
        with st.spinner("🔗 Chunking & embedding documents..."):
            num_chunks = vector_store.add_documents(documents)

        # Rebuild BM25 index for hybrid search
        if settings.ENABLE_HYBRID_SEARCH:
            with st.spinner("🔍 Building search index..."):
                hybrid_retriever = st.session_state.get("hybrid_retriever")
                if hybrid_retriever:
                    hybrid_retriever.build_bm25_index()

        st.session_state["file_hashes"] = updated_hashes
        st.success(f"✅ Indexed {num_chunks} chunks from {len(uploaded_files) - len(skipped)} file(s)")
        logger.info(f"Indexed {num_chunks} chunks from uploads")
    else:
        st.info("ℹ️ No new documents to process.")


def _render_document_management():
    """Render document management section."""
    st.markdown("### 📚 Indexed Documents")

    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        st.caption("No vector store initialized")
        return

    sources = vector_store.list_sources()
    total_chunks = vector_store.get_document_count()

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Documents", len(sources))
    with col2:
        st.metric("🧩 Chunks", total_chunks)

    if not sources:
        st.caption("No documents uploaded yet.")
        return

    # Document list with delete buttons
    for source in sources:
        col_name, col_btn = st.columns([3, 1])
        with col_name:
            st.markdown(
                f"<span style='color: #00d4aa;'>📄</span> {source}",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("🗑️", key=f"del_{source}", help=f"Delete {source}"):
                _delete_document(source)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear all
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        _clear_all_documents()
        st.rerun()


def _delete_document(source_name: str):
    """Delete a specific document from the vector store."""
    vector_store = st.session_state.get("vector_store")
    if vector_store:
        deleted = vector_store.delete_documents_by_source(source_name)
        if deleted > 0:
            # Rebuild BM25 index
            hybrid_retriever = st.session_state.get("hybrid_retriever")
            if hybrid_retriever and settings.ENABLE_HYBRID_SEARCH:
                hybrid_retriever.build_bm25_index()
            st.toast(f"Deleted {deleted} chunks from '{source_name}'", icon="🗑️")
            logger.info(f"Deleted document: {source_name}")


def _clear_all_documents():
    """Clear all documents from the vector store."""
    vector_store = st.session_state.get("vector_store")
    if vector_store:
        vector_store.clear_collection()
        st.session_state["file_hashes"] = set()
        st.session_state["chat_history"] = []
        st.session_state["eval_scores"] = []
        st.toast("All documents cleared", icon="🗑️")
        logger.info("All documents cleared")


def _render_settings():
    """Render the settings panel."""
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("**Retrieval Settings**")

        # Top-K slider
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=15,
            value=settings.TOP_K,
            step=1,
            help="Number of document chunks to retrieve per query",
        )
        settings.TOP_K = top_k

        # Hybrid search toggle
        hybrid = st.toggle(
            "Enable Hybrid Search (BM25 + Semantic)",
            value=settings.ENABLE_HYBRID_SEARCH,
            help="Combine keyword and semantic search for better results",
        )
        settings.ENABLE_HYBRID_SEARCH = hybrid

        if hybrid:
            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=settings.SEMANTIC_WEIGHT,
                step=0.1,
            )
            settings.SEMANTIC_WEIGHT = semantic_weight
            settings.BM25_WEIGHT = round(1.0 - semantic_weight, 1)
            st.caption(f"BM25 Weight: {settings.BM25_WEIGHT}")

        st.markdown("**Chunking Settings**")
        chunk_size = st.slider(
            "Chunk Size (chars)",
            min_value=200,
            max_value=3000,
            value=settings.CHUNK_SIZE,
            step=100,
        )
        settings.CHUNK_SIZE = chunk_size

        chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=500,
            value=settings.CHUNK_OVERLAP,
            step=50,
        )
        settings.CHUNK_OVERLAP = chunk_overlap
