"""
Thin wrapper around the Groq API (OpenAI-compatible chat completions).

Isolated in its own module so the model/provider can be swapped by editing
only this file. Reads the API key from the GROQ_API_KEY environment variable.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Calls Groq's chat completions endpoint with JSON-mode forced, so the
    model is constrained to return valid JSON. Returns the raw JSON string
    from the model (caller is responsible for json.loads + validation).
    """
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
