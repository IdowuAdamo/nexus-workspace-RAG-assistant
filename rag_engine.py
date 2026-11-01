"""
High-level RAG Engine coordinating ChromaDB retrieval and LLM generation
with transparent fallback between OpenAI and Google Gemini.
"""

import logging
from typing import Dict, Any, Optional

import config
from vector_store import VectorStore
from llm_client import generate_answer_with_fallback

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orchestrates retrieval and generation for Nexus Workplace Services policy RAG."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Ensures document indices are ready in ChromaDB."""
        return self.vector_store.index_documents(force_reindex=force_reindex)

    def ask(
        self,
        question: str,
        n_results: int = config.DEFAULT_TOP_K,
        preferred_provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline:
        1. Embeds question and retrieves relevant policy sections from ChromaDB.
        2. Formats retrieved chunks into grounded prompt context.
        3. Calls primary LLM (OpenAI) with automatic fallback to Gemini if errors occur.
        4. Packages answer, source citations, and provider diagnostics.
        """
        question = question.strip()
        if not question:
            return {
                "question": "",
                "answer": "Please ask a question about the policy documents.",
                "sources": [],
                "retrieval_provider": None,
                "llm_provider": None,
                "is_fallback": False,
                "fallback_reason": None
            }

        # Step 1: Retrieval with fallback
        chunks, ret_provider, ret_fallback_reason = self.vector_store.query(
            query_text=question,
            n_results=n_results,
            preferred_provider=preferred_provider
        )

        # Step 2: Generation with fallback
        # If retrieval had to fall back to Gemini, we also prefer Gemini for generation to ensure consistency
        gen_preferred = ret_provider if ret_provider == "gemini" else preferred_provider

        answer, llm_provider, model_name, is_gen_fallback, gen_fallback_reason = generate_answer_with_fallback(
            question=question,
            context_chunks=chunks,
            preferred_provider=gen_preferred
        )

        # Check if any fallback was triggered
        is_fallback = (ret_provider == "gemini" and preferred_provider == "openai") or is_gen_fallback
        reasons = []
        if ret_fallback_reason:
            reasons.append(f"Retrieval: {ret_fallback_reason}")
        if gen_fallback_reason:
            reasons.append(f"Generation: {gen_fallback_reason}")
        fallback_reason = " | ".join(reasons) if reasons else None

        # Format sources for presentation
        sources = []
        for c in chunks:
            meta = c.get("metadata", {})
            sources.append({
                "chunk_id": c.get("id"),
                "doc_title": meta.get("doc_title", "Unknown Policy"),
                "doc_id": meta.get("doc_id", "N/A"),
                "section_title": meta.get("section_title", "General"),
                "source_file": meta.get("source_file", ""),
                "similarity": c.get("similarity", 0.0),
                "text": c.get("text", "")
            })

        return {
            "question": question,
            "answer": answer,
            "retrieval_provider": ret_provider,
            "llm_provider": llm_provider,
            "model_name": model_name,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "sources": sources
        }

