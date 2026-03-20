"""
vectorstore.py
--------------
Manages embeddings (BGE via sentence-transformers) and Supabase vector store.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ── Singleton embedding model (load once, reuse) ──────────────────────────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    global _embedding_model
    if _embedding_model is None:
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        print(f"🔢 Loading embedding model: {model_name}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Embedding model ready")
    return _embedding_model


def _get_supabase_client():
    """Return a Supabase client using env credentials."""
    from supabase import create_client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
        )
    return create_client(url, key)


# ── Supabase Vector Store ─────────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    table_name: str = "document_embeddings",
    query_name: str = "match_documents",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    """Embed chunks and store them in Supabase."""
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_community.vectorstores.utils import filter_complex_metadata

    embeddings = get_embeddings(embedding_model)
    client = _get_supabase_client()
    chunks = filter_complex_metadata(chunks)

    import time
    print(f"💾 Storing {len(chunks)} chunks in Supabase (batched)...")
    batch_size = 5
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            if vectorstore is None:
                vectorstore = SupabaseVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    client=client,
                    table_name=table_name,
                    query_name=query_name,
                )
            else:
                vectorstore.add_documents(batch)
            print(f"   Uploaded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
            time.sleep(0.5)
        except Exception as e:
            print(f"   ⚠️ Batch {i}-{i+batch_size} failed: {e}, retrying...")
            time.sleep(2)
            try:
                vectorstore.add_documents(batch)
            except Exception as e2:
                print(f"   ❌ Skipping batch: {e2}")
    print("   ✅ Vectorstore saved to Supabase")
    return vectorstore


def load_vectorstore(
    table_name: str = "document_embeddings",
    query_name: str = "match_documents",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    """Load existing vectorstore from Supabase. Returns None if empty."""
    from langchain_community.vectorstores import SupabaseVectorStore

    embeddings = get_embeddings(embedding_model)
    client = _get_supabase_client()

    try:
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=client,
            table_name=table_name,
            query_name=query_name,
        )
        # Check if table has data
        result = client.table(table_name).select("*", count="exact").limit(1).execute()
        count = result.count or 0
        if count == 0:
            return None
        print(f"📦 Loaded Supabase vectorstore ({count} chunks)")
        return vectorstore
    except Exception as e:
        print(f"⚠️  Could not load vectorstore: {e}")
        return None


def add_to_vectorstore(
    vectorstore,
    chunks: List[Document],
):
    """Add new chunks to existing Supabase vectorstore."""
    from langchain_community.vectorstores.utils import filter_complex_metadata
    chunks = filter_complex_metadata(chunks)
    vectorstore.add_documents(chunks)
    print(f"➕ Added {len(chunks)} new chunks to Supabase")
    return vectorstore


def clear_vectorstore(table_name: str = "document_embeddings") -> None:
    """Delete all documents from Supabase table."""
    client = _get_supabase_client()
    client.table(table_name).delete().neq("id", 0).execute()
    print(f"🗑️  Cleared Supabase vectorstore (table: {table_name})")
