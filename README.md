# lr-rag: Local Retrieval-Augmented Generation

> **One-liner:** A privacy-first, local RAG system that ingests PDFs and delivers grounded answers with citations using Ollama + Qdrant.

## 1) Executive Summary

`lr-rag` is a **local-first Retrieval-Augmented Generation (RAG)** implementation for document question answering. It is designed for AI/ML engineers, NLP practitioners, and privacy-conscious developers who need high-quality QA over private PDFs **without sending data to cloud APIs**.

**Project vision:** make trustworthy, auditable QA over documents simple to run on a laptop or workstation.

**Privacy-first positioning:** all core processing (embedding, retrieval, generation) runs locally with Ollama and Qdrant.

---

## 2) Problem Statement

Cloud RAG stacks often introduce practical and governance pain points:

- Sensitive PDFs must leave local infrastructure
- Recurring token/API usage costs
- Compliance and residency concerns (GDPR/internal policy)
- Vendor lock-in and opaque model behavior
- Higher latency from network round-trips
- Limited control over indexing/retrieval internals

---

## 3) Solution Overview

`lr-rag` uses a **local-first architecture**:

1. Ingest PDF text
2. Chunk text into overlapping semantic windows
3. Embed chunks with `nomic-embed-text`
4. Store vectors in Qdrant
5. Embed user query
6. Retrieve top relevant chunks via cosine similarity
7. Build grounded context + citations
8. Generate an answer with a local `llama3` model (commonly 8B; 70B variant if available in your Ollama setup)

### Benefits

- Data remains local
- Deterministic, inspectable retrieval
- Lower operational cost for repeated use
- Easier offline/private deployments

---

## 4) Key Features

- PDF ingestion pipeline
- Sentence-based chunking (`chunk_size=200`, `chunk_overlap=40`)
- Local embedding generation (`nomic-embed-text`, 768 dimensions)
- Semantic search over Qdrant (cosine)
- Grounded QA via Ollama chat model
- Citation mapping from answer context to source files
- CLI and Streamlit interfaces

---

## 5) Architecture Diagram

### Ingestion Pipeline

```text
PDF Upload/Path
   |
   v
load_pdf()  -->  chunk_text()  -->  embed_nodes()  -->  upsert_nodes()
(PDFReader)      (SentenceSplitter)    (Ollama)          (Qdrant)
```

### Retrieval + QA Pipeline

```text
User Question
   |
   v
embed_query() --> search_similar() --> build_context() --> generate_answer()
   (Ollama)         (Qdrant cosine)     (source map)         (Ollama chat)
```

---

## 6) Tech Stack (with versions)

| Layer | Technology | Version Evidence |
|---|---|---|
| Language | Python | `>=3.12` (`pyproject.toml`) |
| LLM runtime | Ollama | locally installed service |
| Vector DB | Qdrant | `qdrant-client>=1.16.2` |
| RAG framework | LlamaIndex | `>=0.14.12` |
| UI | Streamlit | `>=1.28.0` (`pyproject.toml`), `1.53.1` pinned in `requirements.txt` |
| HTTP | httpx | `>=0.28.1` |

---

## 7) System Components

### PDF Loader & OCR
- Module: `src/ingestion/loader.py`
- Uses `llama_index.readers.file.PDFReader` to extract text.
- Validates file existence/type and fails fast on empty extraction.
- **OCR note:** no explicit OCR pipeline is implemented yet; scanned-image PDFs may require OCR preprocessing.

### Text Chunking (`SentenceSplitter`)
- Module: `src/ingestion/chunker.py`
- Uses `SentenceSplitter(chunk_size=200, chunk_overlap=40)`.
- Returns `TextNode` list for downstream embedding/storage.

### Embedding Generation (`nomic-embed-text`)
- Modules: `src/ingestion/embedder.py`, `src/retrieval/query_embedder.py`, `src/services/ollama.py`
- Calls Ollama `/api/embeddings` for both documents and queries.
- Embeddings are attached directly to nodes for upsert.

### Vector Storage (Qdrant)
- Module: `src/services/qdrant.py`, `src/ingestion/upsert.py`
- Ensures collection exists with cosine distance.
- Stores payload fields: `text`, `source`.
- Uses deterministic UUIDv5 (`source + chunk_text`) to avoid duplicate inserts on re-ingestion.

### Semantic Search
- Module: `src/retrieval/search.py`
- Retrieves top-K from Qdrant and filters by `score >= 0.5`.

### Context Building
- Module: `src/retrieval/answer.py`
- Formats retrieved chunks into `[Source N]` blocks.
- Builds citation map `index -> source filename`.

### LLM Answer Generation
- Module: `src/retrieval/answer.py`, `src/services/ollama.py`
- Sends system + context + user prompt to Ollama `/api/chat`.
- Returns final answer and source mapping.

---

## 8) Setup & Installation

```bash
# 1) Clone
git clone https://github.com/dinukadilshan03/lr-rag.git
cd lr-rag

# 2) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 3) Install dependencies
pip install -r requirements.txt
# (or use pyproject-compatible workflow if you manage dependencies differently)

# 4) Configure environment
# This repository includes a `.env` file. Edit it directly for local setup.
# Optional local override: cp .env .env.local && edit .env.local
```

---

## 9) Prerequisites

### Ollama
1. Install Ollama: https://ollama.com
2. Pull models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   # optional larger model (tag availability may vary):
   # ollama pull llama3:70b
   # verify local tags with `ollama list` and model docs at https://ollama.com/library/llama3
   ```

### Qdrant
Run local Qdrant (Docker):
```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

---

## 10) Running the Application

### Streamlit UI
```bash
cd lr-rag
streamlit run streamlit_app.py
# open http://localhost:8501
```

### CLI
```bash
python cli.py ingest ./documents/sample.pdf
python cli.py ask "What are the key findings?"
```

---

## 11) Usage Examples

### Example A: Upload + ingest in UI
1. Open Streamlit app.
2. Upload a PDF.
3. Click **Ingest PDF**.
4. Confirm success message with chunk count.

### Example B: Ask a grounded question
1. Enter question: `What is cosine similarity?`
2. Click **Ask**.
3. Review answer + source list + retrieved chunk scores.

---

## 12) Model Specifications

- **Embedding model:** `nomic-embed-text`
  - Vector dimension: **768D** (`settings.VECTOR_DIMENSION = 768`)
- **LLM model:** `llama3` family via Ollama
  - Common local options: 8B and 70B variants (hardware-dependent)
- **LlamaIndex:** `0.14.12+`

---

## 13) Chunking Strategy

Current strategy (`src/ingestion/chunker.py`):
- Chunk size: **200 tokens**
- Overlap: **40 tokens**

Rationale:
- Preserves local context around boundaries
- Reduces semantic fragmentation
- Balances retrieval precision vs. context completeness

---

## 14) Embedding & Search

- Distance metric: **cosine similarity** (Qdrant collection config)
- Retrieval filter: **`score >= 0.5`** (`src/retrieval/search.py`)
- Tuning guidance:
  - Increase threshold for higher precision, lower recall
  - Decrease threshold for higher recall, potentially noisier context

---

## 15) Grounding Mechanism

Grounding is implemented by:
1. Retrieving relevant chunks only
2. Building explicit context blocks with source IDs
3. Prompting model to use provided context and avoid speculation
4. Returning citations mapped from retrieved chunks

This pattern reduces hallucinations compared to free-form generation.

---

## 16) Data Persistence

Qdrant collection defaults:
- Collection name: `lr_rag_documents`
- Vector size: `768`
- Distance: `COSINE`

Stored payload schema per point:
```json
{
  "text": "chunk content",
  "source": "document_name.pdf"
}
```

Namespacing/multi-tenant note:
- Current implementation uses a single collection with `source` in payload.
- You can namespace by custom `COLLECTION_NAME` per project/user via `.env`.

---

## 17) Performance Metrics

This repository does not include a committed benchmark suite yet. Practical local metrics to track:

- **Embedding latency/chunk:** typically single-digit to tens of ms on modern hardware
- **Query latency (embed + retrieve + answer):** hardware/model dependent
- **Retrieval quality:** % of questions where top-K includes answer-bearing chunk(s)

Recommended KPI baseline to add in your environment:
- p50 / p95 query latency
- ingestion throughput (chunks/sec)
- retrieval hit rate@K

---

## 18) Docker Setup

Current repo includes `docker/qdrant/docker-compose.yml` path. A practical compose setup is:

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

Run with:
```bash
docker compose -f docker/qdrant/docker-compose.yml up -d
```

---

## 19) Configuration

From `.env` / `src/config/settings.py`:

- `QDRANT_URL` (default `http://localhost:6333`)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `EMBEDDING_MODEL` (default `nomic-embed-text`)
- `LLM_MODEL` (default `llama3`)
- `COLLECTION_NAME` (default `lr_rag_documents`)
- `VECTOR_DIMENSION` (fixed `768`)
- `TOP_K` (default `5`)

Code-level retrieval/chunking knobs:
- `chunk_size` / `chunk_overlap`
- similarity threshold (`0.5`)

---

## 20) Limitations & Considerations

- No dedicated OCR stage for scanned PDFs yet
- Quality depends on local model and hardware
- Context-window limits constrain long-answer synthesis
- Single-collection approach may need stronger isolation for multi-tenant use
- No built-in reranker or hybrid lexical+vector retrieval currently

---

## 21) Comparison with Cloud RAG

| Dimension | Local-First (`lr-rag`) | Typical Cloud RAG |
|---|---|---|
| Data privacy | Strong (local processing) | Data leaves local boundary |
| Cost model | Infra/electricity | Token/API usage fees |
| Offline capability | Yes | Usually no |
| Setup effort | Moderate | Often lower initial setup |
| Control/customization | High | Varies by vendor |

---

## 22) Privacy & Security

- No mandatory cloud dependency for core QA flow
- Documents can remain fully on local infrastructure
- Data egress is controlled by your local deployment choices

Security best practices:
- Run services on trusted local network
- Restrict exposed ports for Qdrant/Ollama
- Use isolated collections or instances by environment/team

---

## 23) Contributing

Ways to extend:
- Add OCR fallback for scanned PDFs
- Add custom embedding/LLM models
- Add reranking and hybrid search
- Add automated evaluation + benchmark scripts

Contribution flow:
1. Fork and create a feature branch
2. Keep changes focused and testable
3. Open a PR with architecture/performance impact notes

---

## 24) Learning Outcomes

This project demonstrates hands-on capability in:
- End-to-end RAG pipeline design
- Vector database integration (Qdrant)
- Local LLM + embedding integration (Ollama)
- Retrieval grounding and citation mapping
- Building usable QA interfaces (CLI + Streamlit)

---

## 25) Future Enhancements

- Multi-file/session-aware context orchestration
- Hybrid retrieval (BM25 + dense vectors)
- Cross-encoder reranking
- Better citation alignment and evidence highlighting
- Benchmark suite for latency/quality regression tracking

---

## 26) License & Author

- **License:** Not found in repository at the time of writing (recommend adding `LICENSE`)
- **Author/Owner:** [dinukadilshan03](https://github.com/dinukadilshan03)
- **Repository:** https://github.com/dinukadilshan03/lr-rag

---

## Quick Start (TL;DR)

```bash
# start qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# make sure ollama models exist
ollama pull nomic-embed-text
ollama pull llama3
# If your setup exposes parameter-specific tags, pull the one you need.

# run app
cd lr-rag
streamlit run streamlit_app.py
```
