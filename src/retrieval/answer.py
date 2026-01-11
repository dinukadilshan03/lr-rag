from typing import List, Dict

from src.services.ollama import ollama_client


def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Assemble retrieved text chunks into a single context block.
    """
    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    context_parts = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        if not text:
            continue

        context_parts.append(
            f"[Source {idx}]\n{text}"
        )

    if not context_parts:
        raise RuntimeError("No valid text found in retrieved chunks")

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Generate a grounded answer using retrieved context.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    context = build_context(retrieved_chunks)

    answer = ollama_client.chat(
        prompt=question,
        context=context,
    )

    if not answer:
        raise RuntimeError("LLM returned an empty answer")

    return answer
