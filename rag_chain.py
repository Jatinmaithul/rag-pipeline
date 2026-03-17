"""
rag_chain.py
------------
Builds the RAG chain: retriever + prompt + LLM + output parser.
"""

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel


# ── Prompt Template ───────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Use the following context to answer the question accurately.
If the answer is not found in the context, say "I don't have enough information in the provided documents to answer this question."
Do not make up information. Be concise and clear.

Context:
{context}

Question: {question}

Answer:""",
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Combine retrieved chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


# ── RAG Chain Builder ─────────────────────────────────────────────────────────

def build_rag_chain(
    vectorstore: Chroma,
    llm: BaseLanguageModel,
    top_k: int = 4,
):
    """
    Build a LangChain RAG chain using LCEL (LangChain Expression Language).

    Flow: question → retriever → format_docs → prompt → llm → parse output
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def query_with_sources(
    question: str,
    rag_chain,
    retriever,
) -> Dict[str, Any]:
    """
    Run a query and return both the answer and the source documents.
    """
    # Get source docs
    source_docs = retriever.invoke(question)

    # Get answer
    answer = rag_chain.invoke(question)

    # Extract unique sources
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in source_docs
    })

    return {
        "answer": answer,
        "source_documents": source_docs,
        "sources": sources,
    }
