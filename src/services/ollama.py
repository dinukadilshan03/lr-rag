import httpx
from typing import List, Optional

from src.config.settings import settings


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

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()

        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("No embedding returned from Ollama")

        return embedding

    def chat(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the ChatQA model.
        Optionally accepts retrieved context.
        """
        url = f"{self.base_url}/api/chat"

        system_prompt = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the provided context. "
            "If the answer is not in the context, say you do not know."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "stream": False,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        message = data.get("message", {}).get("content")

        if not message:
            raise RuntimeError("No response returned from Ollama")

        return message


# Singleton-style client
ollama_client = OllamaClient()
