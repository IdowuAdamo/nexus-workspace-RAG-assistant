"""
Nexus Workplace Services - Policy Assistant RAG Application.
Frontend built with Streamlit and powered by ChromaDB, OpenAI, and Google Gemini.
"""

# SQLite compatibility patch for ChromaDB on Streamlit Community Cloud (Linux)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import config
from rag_engine import RAGEngine
from vector_store import VectorStore

# Configure Streamlit page
st.set_page_config(
    page_title="Nexus Policy Assistant (RAG)",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-primary {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-fallback {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #FEF08A;
        color: #854D0E;
    }
    .source-box {
        border-left: 3px solid #3B82F6;
        padding: 0.6rem 0.9rem;
        margin: 0.5rem 0;
        background-color: #F8FAFC;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing ChromaDB vector store...")
def get_rag_engine() -> RAGEngine:
    """Initializes and caches the RAGEngine instance."""
    engine = RAGEngine()
    # Initialize index on first load
    engine.initialize()
    return engine


# Initialize engine
rag_engine = get_rag_engine()
vector_store = rag_engine.vector_store

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/documents.png", width=64)
    st.title("Settings & Status")

    # API Keys Status
    st.subheader("API Status")
    openai_ok = config.has_openai_credentials()
    gemini_ok = config.has_gemini_credentials()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"**OpenAI**<br>{'🟢 Configured' if openai_ok else '🔴 Missing'}",
            unsafe_allow_html=True
        )
    with col_s2:
        st.markdown(
            f"**Gemini (Fallback)**<br>{'🟢 Configured' if gemini_ok else '🔴 Missing'}",
            unsafe_allow_html=True
        )

    st.divider()

    # Provider Execution Mode
    st.subheader("Model Provider")
    mode_options = [
        "Auto (OpenAI primary -> Gemini fallback)",
        "Force OpenAI",
        "Force Gemini (Test Fallback)"
    ]
    selected_mode = st.radio(
        "Execution Mode:",
        mode_options,
        index=0,
        help="Select 'Auto' for normal operation, or test fallback directly by selecting 'Force Gemini'."
    )

    # Retrieval parameter
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Context Chunks (Top-K):",
        min_value=1,
        max_value=5,
        value=config.DEFAULT_TOP_K,
        help="Number of most similar policy sections to retrieve from ChromaDB."
    )

    st.divider()

    # Vector DB Collection Statistics
    st.subheader("Vector Database (ChromaDB)")
    stats = vector_store.get_stats()
    st.write(f"• **OpenAI Index**: {stats['openai_count']} chunks")
    st.write(f"• **Gemini Index**: {stats['gemini_count']} chunks")

    if st.button("🔄 Re-index Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing policy documents into ChromaDB..."):
            res = vector_store.index_documents(force_reindex=True)
            st.success(f"Indexed {res['total_chunks']} chunks successfully!")
            st.rerun()

    st.divider()
    with st.expander("📖 Knowledge Base Documents"):
        st.markdown("""
        - **Expense Reimbursement Guide** (`FIN-EXP-006`)
          *Version 1.4 | Effective: 1 Oct 2026*
        - **IT Acceptable Use Policy** (`IT-SEC-002`)
          *Version 3.0 | Effective: 15 Aug 2026*
        - **Staff Travel Policy** (`HR-TRV-004`)
          *Version 2.1 | Effective: 1 Sep 2026*
        """)


# Main Interface Header
st.markdown('<div class="main-title">🏢 Nexus Policy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation (RAG) assistant for Nexus Workplace Services policy documents. '
    'Backed by <b>ChromaDB</b> with automatic <b>OpenAI &rarr; Gemini Fallback</b>.</div>',
    unsafe_allow_html=True
)

# Determine provider from sidebar choice
preferred_provider = "openai"
if "Force Gemini" in selected_mode:
    preferred_provider = "gemini"
elif "Force OpenAI" in selected_mode:
    preferred_provider = "openai"
else:
    preferred_provider = "openai"

# Suggested questions for quick testing
st.markdown("**Quick sample questions:**")
suggested_queries = [
    "What is the receipt requirement for expenses?",
    "Can I use public Wi-Fi for sensitive work without a VPN?",
    "What class of travel is permitted for domestic flights?",
    "What should I do if my company laptop is stolen?"
]

cols = st.columns(len(suggested_queries))
clicked_query = None
for i, q in enumerate(suggested_queries):
    if cols[i].button(q, key=f"sugg_{i}", use_container_width=True):
        clicked_query = q

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            is_fallback = meta.get("is_fallback", False)
            provider = meta.get("llm_provider", "Unknown")
            model = meta.get("model_name", "")

            badge_html = (
                f'<span class="badge-fallback">⚠️ Fallback to {provider} ({model})</span>'
                if is_fallback else
                f'<span class="badge-primary">✓ {provider} ({model})</span>'
            )
            st.markdown(badge_html, unsafe_allow_html=True)

            if meta.get("fallback_reason"):
                st.caption(f"Reason: {meta['fallback_reason']}")

            # Sources dropdown
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Policy Sources ({len(sources)} sections)"):
                    for idx, s in enumerate(sources, 1):
                        st.markdown(
                            f"**[{idx}] {s['doc_title']}** — Section: `{s['section_title']}` "
                            f"(Doc ID: `{s['doc_id']}` | Similarity: `{s['similarity']:.2f}`)"
                        )
                        st.markdown(f"> *{s['text'].splitlines()[-1][:200]}...*")

# Handle user input (either from chat input or clicked sample query)
user_input = st.chat_input("Ask a question about Nexus company policies...")
prompt_to_run = clicked_query or user_input

if prompt_to_run:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt_to_run})
    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents in ChromaDB..."):
            try:
                res = rag_engine.ask(
                    question=prompt_to_run,
                    n_results=top_k,
                    preferred_provider=preferred_provider
                )

                answer = res["answer"]
                st.markdown(answer)

                is_fallback = res["is_fallback"]
                llm_prov = res["llm_provider"]
                model_name = res["model_name"]
                fallback_reason = res["fallback_reason"]

                badge_html = (
                    f'<span class="badge-fallback">⚠️ Fallback triggered: Generated by {llm_prov} ({model_name})</span>'
                    if is_fallback else
                    f'<span class="badge-primary">✓ Generated by {llm_prov} ({model_name})</span>'
                )
                st.markdown(badge_html, unsafe_allow_html=True)

                if fallback_reason:
                    st.caption(f"Fallback reason: {fallback_reason}")

                # Display sources
                sources = res["sources"]
                if sources:
                    with st.expander(f"📚 Retrieved Policy Sources ({len(sources)} sections)"):
                        for idx, s in enumerate(sources, 1):
                            st.markdown(
                                f"**[{idx}] {s['doc_title']}** — Section: `{s['section_title']}` "
                                f"(Doc ID: `{s['doc_id']}` | Similarity: `{s['similarity']:.2f}`)"
                            )
                            st.text(s['text'])

                # Store assistant response with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "is_fallback": is_fallback,
                        "llm_provider": llm_prov,
                        "model_name": model_name,
                        "fallback_reason": fallback_reason,
                        "sources": sources
                    }
                })

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

