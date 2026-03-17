# 🔍 RAG Pipeline — Free Tier Stack

A complete **Retrieval-Augmented Generation (RAG)** pipeline using 100% free tools.

```
PDF/Docs → Unstructured → LangChain Chunking
    → sentence-transformers (embeddings)
        → ChromaDB (vector store)
            → Ollama / Groq (LLM)
                → Streamlit (UI)
```

---

## 📁 Project Structure

```
rag_pipeline/
├── app.py            ← Streamlit UI (run this)
├── ingestion.py      ← Document loading & chunking
├── vectorstore.py    ← ChromaDB + sentence-transformers
├── llm.py            ← Groq / Ollama LLM abstraction
├── rag_chain.py      ← RAG chain (retriever + prompt + LLM)
├── requirements.txt  ← Python dependencies
├── .env.example      ← Environment variables template
└── README.md
```

---

## ⚡ Quick Setup

### 1. Clone & Install

```bash
cd rag_pipeline
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your keys
```

### 3. Choose Your LLM

**Option A — Groq (Recommended, Fast)**
- Sign up free at [console.groq.com](https://console.groq.com)
- Copy your API key → paste in `.env` as `GROQ_API_KEY`

**Option B — Ollama (Fully Local, No Key)**
```bash
# Install Ollama from https://ollama.com
ollama pull llama3   # downloads ~4GB model
# Set LLM_PROVIDER=ollama in .env
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Open `http://localhost:8501` in your browser
2. Upload PDF / DOCX / TXT files in the left panel
3. Click **"⚡ Index Documents"** — this embeds and stores your docs
4. Ask questions in the right panel chat interface
5. Get answers with source citations!

---

## 🧩 How It Works

| Stage | Tool | What it does |
|-------|------|-------------|
| **Load** | Unstructured + PyMuPDF | Parses PDF, DOCX, TXT, HTML |
| **Chunk** | LangChain RecursiveTextSplitter | Splits text into overlapping windows |
| **Embed** | `sentence-transformers/all-MiniLM-L6-v2` | Converts text → 384-dim vectors |
| **Store** | ChromaDB | Persists vectors to disk |
| **Retrieve** | ChromaDB similarity search | Finds top-K relevant chunks |
| **Generate** | Groq (llama3) / Ollama | Synthesizes answer from context |
| **UI** | Streamlit | Chat interface |

---

## 🎛️ Configuration

All settings can be tweaked in `.env` or via the Streamlit sidebar:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | Your Groq API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vectorstore disk location |

---

## 💡 Tips

- **ChromaDB is persistent** — once indexed, reload with "Load from Disk"
- **Chunk size matters** — smaller chunks = precise retrieval, larger = more context
- **Top-K** — increase for broader answers, decrease for precision
- **Model swap** — try `mixtral-8x7b-32768` on Groq for stronger reasoning

---

## 📦 Free Tier Summary

| Tool | Cost |
|------|------|
| sentence-transformers | Free (local) |
| ChromaDB | Free (local) |
| Groq API | Free (rate-limited) |
| Ollama | Free (local) |
| Streamlit | Free |
| **Total** | **$0** |
