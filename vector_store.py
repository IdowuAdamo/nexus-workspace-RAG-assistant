"""
ChromaDB Vector Store Manager.

Manages persistent vector collections for both OpenAI and Gemini embeddings.
Maintains two separate collections to handle vector dimension differences:
- 'nexus_policies_openai' (1536 dims, OpenAI text-embedding-3-small)
- 'nexus_policies_gemini' (3072 dims, Gemini gemini-embedding-001)

This ensures seamless dual-indexing and zero-downtime retrieval fallback.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

import config
from chunking import load_and_chunk_all_documents
from embeddings import OpenAIEmbedder, GeminiEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB collections and provides indexing & search operations."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or config.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(self, collection_name: str):
        """Get or create a named ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of items in a collection."""
        try:
            col = self.client.get_collection(collection_name)
            return col.count()
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Return indexing statistics for both collections."""
        openai_count = self.get_collection_count(config.COLLECTION_OPENAI)
        gemini_count = self.get_collection_count(config.COLLECTION_GEMINI)
        return {
            "openai_collection": config.COLLECTION_OPENAI,
            "openai_count": openai_count,
            "gemini_collection": config.COLLECTION_GEMINI,
            "gemini_count": gemini_count,
            "persist_dir": self.persist_dir
        }

    def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Loads documents, splits into semantic section chunks, and indexes into ChromaDB.
        Attempts indexing for OpenAI and Gemini independently so fallback is always ready.
        """
        chunks = load_and_chunk_all_documents(config.KNOWLEDGE_BASE_DIR)
        num_chunks = len(chunks)
        results = {
            "total_chunks": num_chunks,
            "openai_indexed": False,
            "gemini_indexed": False,
            "errors": []
        }

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # 1. Index OpenAI collection if credentials exist and needed
        openai_count = self.get_collection_count(config.COLLECTION_OPENAI)
        if config.has_openai_credentials() and (force_reindex or openai_count < num_chunks):
            try:
                logger.info("Indexing documents into OpenAI Chroma collection...")
                embedder = OpenAIEmbedder()
                embeddings = embedder.embed_texts(texts)
                col_openai = self.get_or_create_collection(config.COLLECTION_OPENAI)
                col_openai.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                results["openai_indexed"] = True
                logger.info(f"Successfully indexed {len(ids)} chunks into OpenAI collection.")
            except Exception as e:
                err_msg = f"Failed to index OpenAI collection: {str(e)}"
                logger.warning(err_msg)
                results["errors"].append(err_msg)
        elif openai_count >= num_chunks:
            results["openai_indexed"] = True

        # 2. Index Gemini collection if credentials exist and needed
        gemini_count = self.get_collection_count(config.COLLECTION_GEMINI)
        if config.has_gemini_credentials() and (force_reindex or gemini_count < num_chunks):
            try:
                logger.info("Indexing documents into Gemini Chroma collection...")
                embedder = GeminiEmbedder()
                embeddings = embedder.embed_texts(texts)
                col_gemini = self.get_or_create_collection(config.COLLECTION_GEMINI)
                col_gemini.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                results["gemini_indexed"] = True
                logger.info(f"Successfully indexed {len(ids)} chunks into Gemini collection.")
            except Exception as e:
                err_msg = f"Failed to index Gemini collection: {str(e)}"
                logger.warning(err_msg)
                results["errors"].append(err_msg)
        elif gemini_count >= num_chunks:
            results["gemini_indexed"] = True

        return results

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        preferred_provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """
        Queries ChromaDB using vector similarity.
        Implements automatic fallback if the preferred provider fails or has no index.

        Returns:
            Tuple of:
            - List of retrieved document dictionaries with id, text, metadata, distance
            - Provider used for retrieval ('openai' or 'gemini')
            - Fallback reason (if fallback occurred, otherwise None)
        """
        fallback_reason = None

        # Try preferred provider first
        if preferred_provider == "openai":
            if self.get_collection_count(config.COLLECTION_OPENAI) > 0 and config.has_openai_credentials():
                try:
                    embedder = OpenAIEmbedder()
                    query_vector = embedder.embed_query(query_text)
                    col = self.client.get_collection(config.COLLECTION_OPENAI)
                    res = col.query(
                        query_embeddings=[query_vector],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    return self._format_query_results(res), "openai", None
                except Exception as err:
                    fallback_reason = f"OpenAI retrieval failed: {err}"
                    logger.warning(f"{fallback_reason}. Falling back to Gemini.")
            else:
                fallback_reason = "OpenAI collection is empty or credentials are not configured."

        # Fallback to Gemini
        try:
            if self.get_collection_count(config.COLLECTION_GEMINI) == 0:
                self.index_documents()

            embedder = GeminiEmbedder()
            query_vector = embedder.embed_query(query_text)
            col = self.client.get_collection(config.COLLECTION_GEMINI)
            res = col.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            return self._format_query_results(res), "gemini", fallback_reason
        except Exception as err:
            logger.error(f"Gemini retrieval also failed: {err}")
            raise RuntimeError(
                f"Both OpenAI and Gemini retrieval failed. Details: {fallback_reason}; Gemini error: {err}"
            )

    @staticmethod
    def _format_query_results(raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Standardizes ChromaDB query output into a clean list of dictionaries."""
        formatted = []
        if not raw_results or not raw_results.get("ids"):
            return formatted

        ids = raw_results["ids"][0]
        docs = raw_results["documents"][0] if "documents" in raw_results else [""] * len(ids)
        metas = raw_results["metadatas"][0] if "metadatas" in raw_results else [{}] * len(ids)
        distances = raw_results["distances"][0] if "distances" in raw_results else [0.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Cosine similarity score = 1 - cosine distance
            similarity = round(max(0.0, 1.0 - float(dist)), 4) if dist is not None else 1.0
            formatted.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity
            })

        return formatted
