"""
ingestion.py
------------
Handles document loading, parsing, and chunking.
Supports: PDF, DOCX, TXT, HTML, Markdown
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Supported Extensions ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from a list of file paths.
    Uses Unstructured for rich parsing, falls back to simple loaders.
    """
    documents = []

    for file_path in file_paths:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"⚠️  Skipping unsupported file: {path.name}")
            continue

        print(f"📄 Loading: {path.name}")
        try:
            docs = _load_file(str(path), ext)
            documents.extend(docs)
            print(f"   ✅ Loaded {len(docs)} document(s)")
        except Exception as e:
            print(f"   ❌ Failed to load {path.name}: {e}")

    return documents


def _load_file(file_path: str, ext: str) -> List[Document]:
    """Route file to the correct loader based on extension."""

    # Try Unstructured first (handles most formats well)
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file_path, mode="elements")
        return loader.load()
    except Exception:
        pass  # Fall back to specific loaders

    # Fallback loaders
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()

    raise ValueError(f"No loader available for: {ext}")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into chunks using recursive character splitting.
    Preserves metadata from the original document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks
