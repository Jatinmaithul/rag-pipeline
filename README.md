# RAG Pipeline

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline using free tools.

```
PDF/DOCX/TXT/JSON → Unstructured → Semantic Chunking
    → BGE-small Embeddings
        → Supabase pgvector (cloud, persistent)
            → Hybrid Search (BM25 + Vector)
                → Groq / Ollama (LLM)
                    → Streamlit (UI)
```

---

## Project Structure

```
rag-pipeline/
├── app.py            ← Streamlit UI (run this)
├── ingestion.py      ← Document loading, cleaning & chunking
├── vectorstore.py    ← Supabase + BGE embeddings
├── llm.py            ← Groq / Ollama LLM abstraction + caching
├── rag_chain.py      ← RAG chain (hybrid retrieval + prompt + LLM)
├── requirements.txt  ← Python dependencies
├── packages.txt      ← System dependencies (libmagic)
├── .env              ← Environment variables (never commit this)
└── README.md
```

---

## Features

| Feature | Details |
|---------|---------|
| **Semantic Chunking** | Splits by meaning, not character count |
| **Hybrid Search** | BM25 keyword + vector similarity combined |
| **Conversation Memory** | Condenses follow-up questions using chat history |
| **Persistent Storage** | Supabase pgvector — survives redeployments |
| **LLM Caching** | Repeated questions return instantly |
| **Retry Logic** | Auto-retries on network/timeout errors |
| **Input Guardrails** | Validates questions before hitting the LLM |
| **JSON Support** | Extracts readable text from JSON files automatically |
| **Multi-format** | PDF, DOCX, TXT, MD, HTML, JSON |

---

## Quick Setup

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install System Dependency (macOS)

```bash
brew install libmagic
```

### 3. Set Up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run:

**Block 1 — Create table:**
```sql
create extension if not exists vector;

create table document_embeddings (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(384)
);

grant all on document_embeddings to anon, authenticated, service_role;
```

**Block 2 — Create search function:**
```sql
create or replace function match_documents (
  query_embedding vector(384),
  match_count int default 4
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql stable
as $$
begin
  return query
  select
    d.id::uuid,
    d.content::text,
    d.metadata::jsonb,
    (1 - (d.embedding <=> query_embedding))::float
  from document_embeddings d
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 4. Configure Environment

Create a `.env` file:

```env
# LLM
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
LLM_PROVIDER=groq

# Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
HF_TOKEN=your_huggingface_token

# Supabase
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key

# Chunking
CHUNK_SIZE=300
CHUNK_OVERLAP=30
```

Get your keys:
- **Groq API key** → [console.groq.com](https://console.groq.com) (free)
- **HuggingFace token** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free)
- **Supabase keys** → Project Settings → API

### 5. Run the App

```bash
streamlit run app.py
```

---

## Usage

1. Open `http://localhost:8501`
2. Select **LLM Provider** and model in the sidebar
3. Upload PDF / DOCX / TXT / JSON files in the left panel
4. Click **"Index Documents"** — embeds and stores to Supabase
5. Ask questions in the chat panel
6. Get answers with source citations

---

## How It Works

| Stage | Tool | Details |
|-------|------|---------|
| **Load** | Unstructured + PyMuPDF | Parses PDF, DOCX, TXT, HTML, JSON |
| **Clean** | Custom JSON parser | Extracts readable text from JSON |
| **Chunk** | SemanticChunker + fallback | Splits by meaning, not character count |
| **Embed** | `BAAI/bge-small-en-v1.5` | 384-dim vectors, fast and accurate |
| **Store** | Supabase pgvector | Cloud-persistent vector storage |
| **Retrieve** | BM25 + vector hybrid | Keyword + semantic search combined |
| **Generate** | Groq / Ollama + cache | LLM with InMemory caching |
| **UI** | Streamlit | Chat interface with source citations |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `OLLAMA_MODEL` | `llama3` | Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `HF_TOKEN` | — | HuggingFace token |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | — | Supabase service role key |
| `CHUNK_SIZE` | `300` | Characters per chunk (fallback) |
| `CHUNK_OVERLAP` | `30` | Overlap between chunks (fallback) |

---

## Deploying to Streamlit Cloud

1. Push code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as entry point
4. Add secrets under **Advanced Settings → Secrets**:

```toml
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama-3.1-8b-instant"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_KEY = "your_service_role_key"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_TOKEN = "your_hf_token"
```

---

## Tips

- **Data quality matters more than quantity** — 50 clean docs beat 500 noisy ones
- **Ideal document size** — 2 to 20 pages per file
- **Re-index after changes** — truncate the Supabase table and re-upload if you change the embedding model
- **Supabase is persistent** — indexed documents survive app restarts and redeployments
- **Free tier Groq limit** — 6000 tokens/min; keep chunk size small to stay within limits

---

## Free Tier Summary

| Tool | Cost |
|------|------|
| BGE-small embeddings | Free (local) |
| Supabase pgvector | Free (500MB) |
| Groq API | Free (rate-limited) |
| Ollama | Free (local) |
| Streamlit Cloud | Free (public repos) |
| HuggingFace | Free |
| **Total** | **$0** |
