"""
app.py
------
Streamlit UI for the RAG pipeline.
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0e0e10; color: #e8e6e3; }
.stApp { background-color: #0e0e10; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #151517;
    border-right: 1px solid #2a2a2e;
}

/* Header */
.rag-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.rag-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #e94560;
    margin: 0;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: #8b8fa8;
    margin: 6px 0 0 0;
    font-size: 0.9rem;
}

/* Chat messages */
.user-msg {
    background: #1e1e2e;
    border-left: 3px solid #e94560;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.95rem;
}
.bot-msg {
    background: #151520;
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.source-pill {
    display: inline-block;
    background: #1e2a3a;
    border: 1px solid #0f3460;
    color: #60a3d9;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 4px 4px 0 0;
}
.status-ok  { color: #00d4aa; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.status-err { color: #e94560; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.section-label {
    color: #8b8fa8;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session():
    defaults = {
        "vectorstore": None,
        "rag_chain": None,
        "retriever": None,
        "chat_history": [],
        "chunks": [],
        "docs_loaded": False,
        "doc_count": 0,
        "chunk_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚙ Configuration</p>', unsafe_allow_html=True)

    llm_provider = st.selectbox(
        "LLM Provider",
        ["groq", "ollama"],
        index=0,
        help="Groq = free hosted API | Ollama = local (install separately)",
    )

    if llm_provider == "groq":
        groq_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get free key at console.groq.com",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        )
        os.environ["GROQ_MODEL"] = groq_model

    else:
        ollama_model = st.text_input("Ollama Model", value="llama3")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        os.environ["OLLAMA_MODEL"] = ollama_model
        os.environ["OLLAMA_BASE_URL"] = ollama_url

    st.markdown("---")
    st.markdown('<p class="section-label">📐 Chunking</p>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top-K Retrieval", 1, 10, 4)

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Stats</p>', unsafe_allow_html=True)
    if st.session_state.docs_loaded:
        st.markdown(f'<p class="status-ok">● Vectorstore ready</p>', unsafe_allow_html=True)
        st.caption(f"Chunks indexed: **{st.session_state.chunk_count}**")
    else:
        st.markdown('<p class="status-err">○ No documents loaded</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Vectorstore", use_container_width=True):
        from vectorstore import clear_vectorstore
        clear_vectorstore()
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.retriever = None
        st.session_state.docs_loaded = False
        st.session_state.chunk_count = 0
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.success("Cleared!")
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1>⟨ RAG Pipeline ⟩</h1>
  <p>Retrieval-Augmented Generation · sentence-transformers + ChromaDB + Groq/Ollama</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# ── Left Column: Upload & Index ───────────────────────────────────────────────
with col1:
    st.markdown('<p class="section-label">📁 Upload Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, DOCX, or MD files",
        type=["pdf", "txt", "md", "docx", "html"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

    if st.button("⚡ Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    from ingestion import load_documents, chunk_documents
                    from vectorstore import build_vectorstore, load_vectorstore, add_to_vectorstore
                    from llm import get_llm
                    from rag_chain import build_rag_chain

                    # Save uploaded files to temp dir
                    tmp_paths = []
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for uf in uploaded_files:
                            tmp_path = os.path.join(tmp_dir, uf.name)
                            with open(tmp_path, "wb") as f:
                                f.write(uf.getbuffer())
                            tmp_paths.append(tmp_path)

                        # Ingest
                        st.info("📄 Loading documents...")
                        docs = load_documents(tmp_paths)

                        st.info("✂️ Chunking...")
                        chunks = chunk_documents(docs, chunk_size, chunk_overlap)

                        st.info("🔢 Embedding & indexing...")
                        # Append to existing or create new
                        if st.session_state.vectorstore is not None:
                            vs = add_to_vectorstore(st.session_state.vectorstore, chunks)
                        else:
                            vs = build_vectorstore(chunks)

                        st.info("🤖 Loading LLM...")
                        llm = get_llm(llm_provider)

                        st.session_state.chunks.extend(chunks)
                        chain, retriever = build_rag_chain(
                            vs, llm, top_k, st.session_state.chunks
                        )

                        st.session_state.vectorstore = vs
                        st.session_state.rag_chain = chain
                        st.session_state.retriever = retriever
                        st.session_state.docs_loaded = True
                        st.session_state.chunk_count += len(chunks)

                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

    # Load existing vectorstore
    st.markdown("---")
    st.markdown('<p class="section-label">💾 Or Load Saved Vectorstore</p>', unsafe_allow_html=True)
    if st.button("📦 Load from Disk", use_container_width=True):
        with st.spinner("Loading vectorstore..."):
            try:
                from vectorstore import load_vectorstore
                from llm import get_llm
                from rag_chain import build_rag_chain

                vs = load_vectorstore()
                if vs is None:
                    st.warning("No saved vectorstore found. Index documents first.")
                else:
                    llm = get_llm(llm_provider)
                    chain, retriever = build_rag_chain(vs, llm, top_k)
                    count = vs._collection.count()

                    st.session_state.vectorstore = vs
                    st.session_state.rag_chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.docs_loaded = True
                    st.session_state.chunk_count = count

                    st.success(f"✅ Loaded vectorstore ({count} chunks)")
            except Exception as e:
                st.error(f"❌ {e}")


# ── Right Column: Chat Interface ──────────────────────────────────────────────
with col2:
    st.markdown('<p class="section-label">💬 Ask Your Documents</p>', unsafe_allow_html=True)

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="color:#555;text-align:center;padding:60px 0;font-size:0.9rem;">'
                'Index your documents, then ask anything.<br>'
                '<span style="font-size:1.5rem">🔍</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                sources_html = "".join(
                    f'<span class="source-pill">📎 {s}</span>'
                    for s in msg.get("sources", [])
                )
                st.markdown(
                    f'<div class="bot-msg">🤖 {msg["content"]}'
                    f'{"<br><br>" + sources_html if sources_html else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Question",
            placeholder="Ask something about your documents...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

    if submitted and user_input:
        if not st.session_state.docs_loaded:
            st.warning("⚠️ Please index documents first.")
        elif len(user_input.strip()) < 3:
            st.warning("⚠️ Please ask a more specific question.")
        elif len(user_input.strip()) > 500:
            st.warning("⚠️ Question is too long. Please keep it under 500 characters.")
        else:
            with st.spinner("Retrieving & generating..."):
                try:
                    from rag_chain import query_with_sources

                    result = query_with_sources(
                        user_input,
                        st.session_state.rag_chain,
                        st.session_state.retriever,
                        chat_history=st.session_state.chat_history,
                    )

                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
                    st.exception(e)

    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
