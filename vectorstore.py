"""
vectorstore.py
--------------
Manages embeddings (sentence-transformers) and ChromaDB vector store.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ── Singleton embedding model (load once, reuse) ──────────────────────────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"🔢 Loading embedding model: {model_name}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Embedding model ready")
    return _embedding_model


# ── ChromaDB Vector Store ─────────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Chroma:
    """
    Embed chunks and store them in ChromaDB.
    Persists to disk so you don't re-embed on every run.
    """
    embeddings = get_embeddings(embedding_model)

    from langchain_community.vectorstores.utils import filter_complex_metadata
    chunks = filter_complex_metadata(chunks)

    print(f"💾 Storing {len(chunks)} chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    print(f"   ✅ Vectorstore saved to: {persist_dir}")
    return vectorstore


def load_vectorstore(
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Optional[Chroma]:
    """
    Load an existing ChromaDB vectorstore from disk.
    Returns None if no vectorstore exists yet.
    """
    if not os.path.exists(persist_dir):
        return None

    embeddings = get_embeddings(embedding_model)
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        count = vectorstore._collection.count()
        if count == 0:
            return None
        print(f"📦 Loaded vectorstore with {count} chunks from: {persist_dir}")
        return vectorstore
    except Exception as e:
        print(f"⚠️  Could not load vectorstore: {e}")
        return None


def add_to_vectorstore(
    vectorstore: Chroma,
    chunks: List[Document],
) -> Chroma:
    """Add new chunks to an existing vectorstore."""
    from langchain_community.vectorstores.utils import filter_complex_metadata
    chunks = filter_complex_metadata(chunks)
    vectorstore.add_documents(chunks)
    print(f"➕ Added {len(chunks)} new chunks to vectorstore")
    return vectorstore


def clear_vectorstore(persist_dir: str = "./chroma_db") -> None:
    """Delete the vectorstore from disk."""
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🗑️  Cleared vectorstore at: {persist_dir}")
