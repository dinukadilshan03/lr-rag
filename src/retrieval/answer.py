from typing import List, Dict, Tuple
import logging

from src.services.ollama import ollama_client

logger = logging.getLogger(__name__)


def build_context(
    retrieved_chunks: List[Dict],
) -> Tuple[str, Dict[int, str]]:
    """
    Assemble context and return a source map for citations.
    """
    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    context_parts = []
    source_map: Dict[int, str] = {}

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        source = chunk.get("source", "unknown")

        if not text:
            continue

        context_parts.append(
            f"[Source {idx}]\n{text}"
        )
        source_map[idx] = source

    if not context_parts:
        raise RuntimeError("No valid text found in retrieved chunks")

    return "\n\n".join(context_parts), source_map



def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
) -> Tuple[str, Dict[int, str]]:
    """
    Generate a grounded answer and return citation sources.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    context, source_map = build_context(retrieved_chunks)

    logger.info(f"Built context from {len(retrieved_chunks)} chunks, {len(context)} chars")
    logger.debug(f"Context preview: {context[:200]}...")

    answer = ollama_client.chat(
        prompt=question,
        context=context,
    )

    logger.info(f"LLM returned answer of {len(answer)} chars")

    if not answer:
        raise RuntimeError("LLM returned an empty answer")

    return answer, source_map
