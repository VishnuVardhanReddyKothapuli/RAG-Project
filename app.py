"""
RAG Document Assistant — Main Application
============================================
A production-grade Retrieval-Augmented Generation system.
Upload PDFs, ask questions, get grounded answers with source citations.

Run with:  streamlit run app.py
"""

import streamlit as st

# ---- Page Configuration (MUST be first Streamlit call) ----
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "RAG Document Assistant — Ask questions about your PDFs.",
    },
)

# ---- Custom CSS for premium look ----
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat input styling */
    .stChatInput {
        border-color: #333 !important;
    }
    .stChatInput > div {
        background-color: #1a1d23 !important;
        border-color: #333 !important;
        border-radius: 12px !important;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4aa, #00b4d8) !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1d23 !important;
        border-radius: 8px !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1d23;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 10px !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Status widget */
    [data-testid="stStatusWidget"] {
        background-color: #1a1d23 !important;
        border-radius: 10px !important;
    }

    /* Toast */
    .stToast {
        border-radius: 10px !important;
    }

    /* Sidebar separator */
    hr {
        border-color: #333;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d4aa !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background-color: #00d4aa !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Initialize Session State ----
def _init_session_state():
    """Initialize all session state variables."""

    if "initialized" in st.session_state:
        return

    from config import settings

    # Validate API key
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        st.error(
            "⚠️ **Groq API key not configured!**\n\n"
            "1. Copy `.env.example` to `.env`\n"
            "2. Add your Groq API key from [console.groq.com](https://console.groq.com)\n"
            "3. Restart the application"
        )
        st.stop()

    # Initialize vector store
    from vectorstore.chroma_store import ChromaVectorStore
    from retriever.hybrid_retriever import HybridRetriever
    from retriever.query_rewriter import QueryRewriter
    from chains.rag_chain import RAGChain

    with st.spinner("🚀 Initializing RAG pipeline..."):
        # Vector Store
        vector_store = ChromaVectorStore()
        st.session_state["vector_store"] = vector_store

        # Hybrid Retriever
        hybrid_retriever = HybridRetriever(vector_store)
        # Build BM25 index from existing documents (if any)
        if vector_store.get_document_count() > 0:
            hybrid_retriever.build_bm25_index()
        st.session_state["hybrid_retriever"] = hybrid_retriever

        # Query Rewriter
        query_rewriter = QueryRewriter()
        st.session_state["query_rewriter"] = query_rewriter

        # RAG Chain
        rag_chain = RAGChain(hybrid_retriever)
        st.session_state["rag_chain"] = rag_chain

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # File hashes for deduplication
        if "file_hashes" not in st.session_state:
            st.session_state["file_hashes"] = set()

        # Evaluation scores
        if "eval_scores" not in st.session_state:
            st.session_state["eval_scores"] = []

    st.session_state["initialized"] = True


# ---- Main Application ----
def main():
    """Main application entry point."""

    _init_session_state()

    # ---- Sidebar ----
    from ui.sidebar import render_sidebar
    render_sidebar()

    # ---- Main Content Area ----
    from ui.chat import render_chat, render_chat_controls

    # Top controls
    render_chat_controls()

    # Tabs for different views
    tab_chat, tab_sources, tab_metrics = st.tabs([
        "💬 Chat",
        "🔍 Sources",
        "📊 Evaluation",
    ])

    with tab_chat:
        render_chat()

    with tab_sources:
        from ui.sources import render_sources
        render_sources()

    with tab_metrics:
        from ui.metrics import render_metrics
        render_metrics()


if __name__ == "__main__":
    main()
