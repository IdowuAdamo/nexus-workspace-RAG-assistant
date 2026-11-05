"""
Embedding generator module supporting both OpenAI and Google Gemini with fallback logic.
Uses resilient HTTP endpoints with timeout handling to prevent connection stalls.
"""

import logging
from typing import List, Optional, Tuple
import requests
import config

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """Generates embeddings using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_EMBEDDING_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key is missing or empty.")
            from openai import OpenAI
            # Configure explicit timeout to avoid connection hanging
            self._client = OpenAI(api_key=self.api_key, timeout=25.0)
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text chunks in batches."""
        client = self._get_client()
        safe_texts = [t if t.strip() else " " for t in texts]
        # OpenAI allows batch embedding inputs
        response = client.embeddings.create(input=safe_texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        results = self.embed_texts([query])
        return results[0]


class GeminiEmbedder:
    """Generates embeddings using Google Gemini API via REST endpoint."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        model_name = model or config.GEMINI_EMBEDDING_MODEL
        # Normalize model name format (e.g. models/gemini-embedding-001)
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        self.model = model_name

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not self.api_key:
            raise ValueError("Gemini API key is missing or empty.")

        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}"
        safe_text = text if text.strip() else " "
        payload = {
            "content": {
                "parts": [{"text": safe_text}]
            }
        }

        response = requests.post(url, json=payload, timeout=25.0)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini embedding API error ({response.status_code}): {response.text}")

        data = response.json()
        if "embedding" not in data or "values" not in data["embedding"]:
            raise ValueError(f"Unexpected response format from Gemini: {data}")

        return data["embedding"]["values"]

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed a list of text chunks."""
        embeddings = []
        for text in texts:
            vec = self.embed_single(text)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query using Gemini."""
        return self.embed_single(query)


def get_embedding_with_fallback(query: str, preferred_provider: str = "openai") -> Tuple[List[float], str]:
    """
    Attempts to embed the query using the preferred provider (OpenAI).
    If it fails, automatically falls back to Gemini.
    Returns: (embedding_vector, provider_used)
    """
    if preferred_provider == "openai":
        try:
            embedder = OpenAIEmbedder()
            vector = embedder.embed_query(query)
            return vector, "openai"
        except Exception as err:
            logger.warning(f"OpenAI embedding failed ({err}). Falling back to Gemini.")

    # Fallback to Gemini
    embedder = GeminiEmbedder()
    vector = embedder.embed_query(query)
    return vector, "gemini"


if __name__ == "__main__":
    # Interactive demonstration of embeddings
    sample_text = "Employees should submit expense claims within 10 working days."
    print("DEMONSTRATION: How Text Embeddings Work")
    print(f"\n1. Input Text:\n   \"{sample_text}\"")

    print("\n2. Generating Vector Embedding...")
    vector, provider = get_embedding_with_fallback(sample_text)

    print(f"\n   [✓] Provider used: {provider.upper()}")
    print(f"   [✓] Embedding dimensions (length): {len(vector)} numbers")
    print(f"   [✓] First 5 vector values: {vector[:5]}")
    print(f"   [✓] Last 5 vector values:  {vector[-5:]}")
    print(f"   [✓] Vector type: {type(vector[0])}")

    print(f"Teaching takeaway: An embedding converts human text into a dense numerical array of {len(vector)} floating-point coordinates in vector space.")

