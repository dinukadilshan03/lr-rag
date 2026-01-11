from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()


class Settings(BaseModel):
    # ---- Core Services ----
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ---- Models ----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")

    # ---- Vector DB ----
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "lr_rag_documents")
    VECTOR_DIMENSION: int = 768

    # ---- Retrieval ----
    TOP_K: int = 5


# Singleton-style access
settings = Settings()
