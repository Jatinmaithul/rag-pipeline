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

        if not path.exists():
            print(f"❌ File not found: {path}")
            continue

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
    except Exception as e:
        print(f"   ⚠️  Unstructured failed ({e}), falling back to basic loader")

    # Fallback loaders
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader(file_path).load()

    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(file_path).load()

    raise ValueError(f"No loader available for: {ext}")


def _clean_content(text: str) -> str:
    """Extract readable text from JSON content if needed."""
    import json
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text
    try:
        data = json.loads(stripped)
        parts = []
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                for key in ["Content", "content", "Summary", "summary",
                            "Title", "title", "text", "Text", "body", "Body"]:
                    if key in item and item[key]:
                        parts.append(str(item[key]))
            else:
                parts.append(str(item))
        return "\n\n".join(parts) if parts else text
    except Exception:
        return text


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> List[Document]:
    """
    Split documents into chunks.
    Tries semantic chunking first, falls back to recursive character splitting.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    # Clean and filter documents
    for doc in documents:
        doc.page_content = _clean_content(doc.page_content)
    documents = [d for d in documents if d.page_content and d.page_content.strip()]
    if not documents:
        print("⚠️  No valid documents to chunk.")
        return []

    # Try semantic chunking (splits by meaning, not character count)
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from vectorstore import get_embeddings
        print("✂️  Using semantic chunking...")
        splitter = SemanticChunker(
            get_embeddings(),
            breakpoint_threshold_type="percentile",
        )
        chunks = splitter.split_documents(documents)
        print(f"   ✅ Created {len(chunks)} semantic chunks from {len(documents)} document(s)")
        return chunks
    except Exception as e:
        print(f"   ⚠️  Semantic chunking unavailable ({e}), using recursive splitting")

    # Fallback: recursive character splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks
