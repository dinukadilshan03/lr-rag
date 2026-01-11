import sys
from pathlib import Path

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_nodes
from src.ingestion.upsert import upsert_nodes

from src.retrieval.query_embedder import embed_query
from src.retrieval.search import search_similar
from src.retrieval.answer import generate_answer


def ingest(pdf_path: str) -> None:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_id = path.name

    print(f"\nIngesting: {source_id}")

    text = load_pdf(path)
    nodes = chunk_text(text)
    embedded_nodes = embed_nodes(nodes)
    upsert_nodes(embedded_nodes, source_id)

    print("Ingestion complete\n")


def ask(question: str) -> None:
    print(f"\nQuestion:\n{question}\n")

    query_vector = embed_query(question)
    retrieved_chunks = search_similar(query_vector)

    answer, sources = generate_answer(question, retrieved_chunks)

    print("Answer:")
    print(answer)

    print("\nSources:")
    for idx, source in sources.items():
        print(f"[{idx}] {source}")

    print("")


def main():
    if len(sys.argv) < 3:
        print(
            "\nUsage:\n"
            "  uv run python cli.py ingest <pdf_path>\n"
            "  uv run python cli.py ask \"<question>\"\n"
        )
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        ingest(sys.argv[2])

    elif command == "ask":
        question = " ".join(sys.argv[2:])
        ask(question)

    else:
        print(f"\nUnknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
