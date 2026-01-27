from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_nodes
from src.ingestion.upsert import upsert_nodes
from src.retrieval.query_embedder import embed_query
from src.retrieval.search import search_similar
from src.retrieval.answer import generate_answer


st.set_page_config(page_title="lr-rag Demo", page_icon="📄", layout="wide")
st.title("lr-rag: Local RAG Demo")
st.write("Upload a PDF, ingest it into Qdrant, and ask grounded questions via Ollama.")


def ingest_uploaded_pdf(uploaded_file):
    """
    Save the uploaded PDF, run the ingestion pipeline, and return stats.
    """
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    source_id = Path(uploaded_file.name).name

    try:
        text = load_pdf(tmp_path)
        nodes = chunk_text(text)
        embedded_nodes = embed_nodes(nodes)
        upsert_nodes(embedded_nodes, source_id)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "source_id": source_id,
        "raw_chunks": len(nodes),
        "embedded_chunks": len(embedded_nodes),
    }


def answer_question(question: str):
    query_vector = embed_query(question)
    retrieved_chunks = search_similar(query_vector)
    answer, source_map = generate_answer(question, retrieved_chunks)
    return answer, retrieved_chunks, source_map


st.header("1) Ingest a PDF")
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="visible")

if st.button("Ingest PDF", type="primary", disabled=uploaded_file is None):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Ingesting..."):
            try:
                stats = ingest_uploaded_pdf(uploaded_file)
                st.success(
                    f"Ingested {stats['embedded_chunks']} chunks from {stats['source_id']}"
                )
                st.caption(
                    f"Raw chunks: {stats['raw_chunks']} • Embedded: {stats['embedded_chunks']}"
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

st.header("2) Ask a Question")
question = st.text_input("Your question", placeholder="e.g., What is a posting list?")

if st.button("Ask", type="primary", disabled=not question.strip() if question else True):
    with st.spinner("Thinking..."):
        try:
            answer, retrieved, sources = answer_question(question)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for idx, src in sources.items():
                st.write(f"[{idx}] {src}")

            with st.expander("Show retrieved chunks"):
                for idx, chunk in enumerate(retrieved, start=1):
                    st.markdown(f"**Chunk {idx} • Score: {chunk.get('score', 0):.3f}**")
                    st.write(chunk.get("text", ""))
                    st.caption(f"Source: {chunk.get('source', '')}")
        except Exception as exc:
            st.error(f"Question failed: {exc}")

st.sidebar.header("How to Run")
st.sidebar.markdown(
    """
    1. Ensure services are running:
       - Ollama at `OLLAMA_BASE_URL` (default http://localhost:11434)
       - Qdrant at `QDRANT_URL` (default http://localhost:6333)
    2. Export env vars if needed:
       - `LLM_MODEL`, `EMBEDDING_MODEL`, `COLLECTION_NAME`
    3. Start the app:
       - `uv run streamlit run streamlit_app.py`
    """
)
