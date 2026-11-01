"""
LLM Generation client with automatic fallback from OpenAI to Google Gemini.
Handles prompt construction, citation guidelines, and error recovery.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import requests
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Nexus Policy Assistant, an expert on Nexus Workplace Services policies:
- Expense Reimbursement Guide (FIN-EXP-006)
- IT Acceptable Use Policy (IT-SEC-002)
- Staff Travel Policy (HR-TRV-004)

Instructions:
1. Answer the question accurately using ONLY the provided Context sections.
2. Explicitly cite the source Document Title, Document ID, and Section name in your response.
3. If the answer is not contained in the provided Context, say:
   "I could not find information regarding this in the official Nexus Workplace Services policy documents."
4. Be concise, professional, and clear.
"""


def format_context_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved context chunks into a structured prompt."""
    if not context_chunks:
        return f"No relevant context found.\n\nQuestion: {question}"

    context_blocks = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        doc_title = meta.get("doc_title", "Unknown Policy")
        doc_id = meta.get("doc_id", "N/A")
        sec_title = meta.get("section_title", "General")

        header = f"[{i}] Document: {doc_title} (ID: {doc_id}) | Section: {sec_title}"
        text = chunk.get("text", "").strip()
        context_blocks.append(f"{header}\n{text}")

    joined_context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context from Nexus Policy Documents:\n"
        f"------------------------------------\n"
        f"{joined_context}\n"
        f"------------------------------------\n\n"
        f"Question: {question}\n\n"
        f"Answer with citations:"
    )


def generate_with_openai(prompt: str, model_name: Optional[str] = None) -> str:
    """Generates completion using OpenAI API."""
    if not config.has_openai_credentials():
        raise ValueError("OpenAI API key is missing or not configured.")

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)
    model = model_name or config.OPENAI_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def generate_with_gemini(prompt: str, model_name: Optional[str] = None) -> str:
    """Generates completion using Google Gemini API via REST."""
    if not config.has_gemini_credentials():
        raise ValueError("Gemini API key is missing or not configured.")

    model = model_name or config.GEMINI_MODEL
    # Strip any 'models/' prefix for URL construction if present
    clean_model = model.replace("models/", "")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={config.GEMINI_API_KEY}"

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    response = requests.post(url, json=payload, timeout=35.0)

    # If systemInstruction format is rejected, retry with combined user prompt
    if response.status_code != 200:
        combined_payload = {
            "contents": [
                {
                    "parts": [{"text": f"System Guidelines:\n{SYSTEM_PROMPT}\n\nUser Request:\n{prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        response = requests.post(url, json=combined_payload, timeout=35.0)

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No response candidates returned by Gemini: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_pieces = [p.get("text", "") for p in parts if "text" in p]
    return "".join(text_pieces).strip()


def generate_answer_with_fallback(
    question: str,
    context_chunks: List[Dict[str, Any]],
    preferred_provider: str = "openai"
) -> Tuple[str, str, str, bool, Optional[str]]:
    """
    Executes text generation with automatic fallback from OpenAI to Gemini.

    Returns:
        Tuple of:
        - generated_answer (str)
        - provider_used (str: 'OpenAI' or 'Gemini')
        - model_name (str)
        - is_fallback (bool)
        - fallback_reason (str or None)
    """
    prompt = format_context_prompt(question, context_chunks)
    fallback_reason = None

    if preferred_provider == "openai":
        try:
            answer = generate_with_openai(prompt)
            return answer, "OpenAI", config.OPENAI_MODEL, False, None
        except Exception as e:
            fallback_reason = f"OpenAI generation error: {e}"
            logger.warning(f"{fallback_reason}. Switching to Gemini fallback.")

    # Execute Gemini (either preferred or as fallback)
    try:
        answer = generate_with_gemini(prompt)
        is_fallback = (preferred_provider == "openai")
        return answer, "Gemini", config.GEMINI_MODEL, is_fallback, fallback_reason
    except Exception as ge:
        err_msg = f"Both OpenAI and Gemini generation failed. OpenAI error: {fallback_reason}; Gemini error: {ge}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)
