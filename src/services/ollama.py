import httpx
from typing import List, Optional
import logging

from src.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using Ollama.
        Returns a 768-dimension vector.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "prompt": text,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

        embedding = data.get("embedding")
        if not embedding:
            logger.error(f"No embedding returned. Full response: {data}")
            raise RuntimeError("No embedding returned from Ollama")

        return embedding

    def chat(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the ChatQA model.
        Optionally accepts retrieved context.
        Forces concise, context-aware answers.
        """
        url = f"{self.base_url}/api/chat"

        system_prompt = (
            "You are a helpful AI assistant. Use the provided context to answer the user's question.\n"
            "Answer clearly, concisely, and directly. If the answer is not in the context, say you cannot find it.\n"
            "Do not invent or speculate. High-level explanations first; technical details optional.\n"
            "Cite sources if available."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            # Include context explicitly
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context}"
                }
            )

        # Add the user prompt
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,  # lower to reduce tangents
                "num_predict": 512,   # allow sufficiently long answers
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

        logger.info(f"Ollama response keys: {list(data.keys())}")

        message = data.get("message", {}).get("content", "").strip()
        if not message:
            logger.error(f"Empty response from Ollama. Full response: {data}")
            raise RuntimeError("No response returned from Ollama")

        return message

# Singleton client
ollama_client = OllamaClient()