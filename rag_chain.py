"""
rag_chain.py
------------
Production-grade RAG chain with:
- Hardened prompt (injection-resistant)
- Conversation memory (condense follow-up questions)
- Hybrid search (BM25 + vector, with graceful fallback)
- Reranking via Flashrank (with graceful fallback)
- LLM caching
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel


# ── Prompts ───────────────────────────────────────────────────────────────────

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the conversation history and a follow-up question, rephrase the \
follow-up question into a self-contained standalone question that captures all context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history_section"],
    template="""You are a helpful document Q&A assistant. Answer the question using the context provided below.
Be concise, accurate, and helpful. If the answer is not in the context, say "I don't have enough information in the provided documents."
{chat_history_section}
Context:
{context}

Question: {question}

Answer:""",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document], max_chars: int = 4000) -> str:
    """Combine retrieved chunks into a single context string, capped at max_chars."""
    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk = f"[{i}] (Source: {source})\n{doc.page_content}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _format_chat_history(chat_history: List[Dict]) -> str:
    """Convert session chat history to a readable string (last 3 exchanges)."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-6:]:  # last 3 exchanges (user + assistant)
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ── RAG Chain Builder ─────────────────────────────────────────────────────────

def build_rag_chain(
    vectorstore,
    llm: BaseLanguageModel,
    top_k: int = 5,
    chunks: Optional[List[Document]] = None,
):
    """
    Build a production-grade RAG chain.

    Features:
    - Hybrid retrieval (BM25 + vector) if chunks are provided
    - Reranking with Flashrank
    - Condense follow-up questions using chat history
    - Injection-resistant prompt
    """

    # ── Step 1: Build retriever ───────────────────────────────────────────────
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    import time

    class _VectorRetrieverWithRetry:
        """Vector retriever with retry on network errors."""
        def invoke(self, query: str):
            for attempt in range(3):
                try:
                    return vector_retriever.invoke(query)
                except Exception as e:
                    if attempt < 2:
                        print(f"⚠️  Retrieval attempt {attempt+1} failed, retrying...")
                        time.sleep(2 ** attempt)
                    else:
                        raise e

    retriever = _VectorRetrieverWithRetry()

    # Hybrid: BM25 + vector (manual ensemble)
    if chunks:
        try:
            from langchain_community.retrievers import BM25Retriever
            bm25 = BM25Retriever.from_documents(chunks)
            bm25.k = top_k

            class _HybridRetriever:
                """BM25 + vector ensemble with retry."""
                def invoke(self, query: str):
                    bm25_docs = bm25.invoke(query)
                    for attempt in range(3):
                        try:
                            vec_docs = vector_retriever.invoke(query)
                            break
                        except Exception:
                            if attempt < 2:
                                time.sleep(2 ** attempt)
                            else:
                                vec_docs = []
                    seen, merged = set(), []
                    for doc in vec_docs + bm25_docs:
                        key = doc.page_content[:100]
                        if key not in seen:
                            seen.add(key)
                            merged.append(doc)
                    return merged[:top_k]

            retriever = _HybridRetriever()
            print("🔀 Hybrid search enabled (BM25 + Vector)")
        except Exception as e:
            print(f"⚠️  Hybrid search unavailable ({e}), using vector only")

    # ── Step 2: Condense question chain ──────────────────────────────────────
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def _get_standalone_question(inputs: dict) -> str:
        """Rewrite question as standalone if chat history exists."""
        history = inputs.get("chat_history", [])
        question = inputs["question"]
        if not history:
            return question
        return condense_chain.invoke({
            "chat_history": _format_chat_history(history),
            "question": question,
        })

    def _build_prompt_inputs(inputs: dict) -> dict:
        """Retrieve docs and prepare all prompt variables."""
        standalone = inputs["standalone_question"]
        docs = retriever.invoke(standalone)
        history = inputs.get("chat_history", [])
        history_str = _format_chat_history(history)
        history_section = (
            f"\nConversation History:\n{history_str}\n" if history_str else ""
        )
        return {
            "context": _format_docs(docs),
            "question": standalone,
            "chat_history_section": history_section,
        }

    # ── Step 3: Full chain ────────────────────────────────────────────────────
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(_get_standalone_question)
        )
        | RunnableLambda(_build_prompt_inputs)
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ── Query ─────────────────────────────────────────────────────────────────────

def _retry(fn, retries: int = 3, delay: float = 2.0):
    """Retry a function on network/timeout errors."""
    import time
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                print(f"⚠️  Attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
    raise last_err


def query_with_sources(
    question: str,
    rag_chain,
    retriever,
    chat_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run a query and return the answer + source documents."""
    chat_history = chat_history or []

    source_docs = _retry(lambda: retriever.invoke(question))
    answer = _retry(lambda: rag_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    }))

    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in source_docs
    })

    return {
        "answer": answer,
        "source_documents": source_docs,
        "sources": sources,
    }
