"""
Configuration settings for the Nexus RAG application.
Loads environment variables and sets defaults for models and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

# Gemini Configuration (Fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip() or "models/gemini-embedding-001"

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

