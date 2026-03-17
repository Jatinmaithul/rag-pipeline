"""
llm.py
------
LLM provider abstraction — supports Groq (hosted) and Ollama (local).
"""

import os
from langchain_core.language_models import BaseLanguageModel


def get_llm(provider: str = "groq") -> BaseLanguageModel:
    """
    Return the configured LLM.

    Providers:
      - "groq"   → Groq hosted API (free tier, fast) — needs GROQ_API_KEY
      - "ollama" → Ollama local models (fully free, offline)
    """
    provider = provider.lower().strip()

    if provider == "groq":
        return _get_groq_llm()
    elif provider == "ollama":
        return _get_ollama_llm()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose 'groq' or 'ollama'.")


# ── Groq ──────────────────────────────────────────────────────────────────────

def _get_groq_llm() -> BaseLanguageModel:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.2,
            max_tokens=1024,
        )
        print(f"🤖 Using Groq LLM: {model}")
        return llm
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")


# ── Ollama ────────────────────────────────────────────────────────────────────

def _get_ollama_llm() -> BaseLanguageModel:
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=0.2,
        )
        print(f"🤖 Using Ollama LLM: {model} @ {base_url}")
        return llm
    except ImportError:
        raise ImportError("Run: pip install langchain-community")
