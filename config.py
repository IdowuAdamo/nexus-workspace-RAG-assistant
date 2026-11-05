"""
Configuration settings for the Nexus RAG application.
Loads environment variables and sets defaults for models and paths.
Supports local .env files and Streamlit Community Cloud secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Load environment variables from local .env file (if present)
load_dotenv(BASE_DIR / ".env")


def get_env_var(key: str, default: str = "") -> str:
    """
    Retrieve configuration from environment variables or Streamlit secrets.
    Enables seamless deployment on Streamlit Community Cloud without code changes.
    """
    val = os.getenv(key, "")
    if val:
        return val.strip()
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


# OpenAI Configuration
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
OPENAI_MODEL = get_env_var("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = get_env_var("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"

# Gemini Configuration (Fallback)
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
GEMINI_MODEL = get_env_var("GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = get_env_var("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001") or "models/gemini-embedding-001"

# ChromaDB collection names
COLLECTION_OPENAI = "nexus_policies_openai"
COLLECTION_GEMINI = "nexus_policies_gemini"

# Default retrieval parameters
DEFAULT_TOP_K = 3


def has_openai_credentials() -> bool:
    """Check if OpenAI API key is present."""
    return bool(OPENAI_API_KEY and len(OPENAI_API_KEY) > 15)


def has_gemini_credentials() -> bool:
    """Check if Gemini API key is present."""
    return bool(GEMINI_API_KEY and len(GEMINI_API_KEY) > 10)
