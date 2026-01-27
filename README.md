# lr-rag: Local Retrieval-Augmented Generation

A production-ready, self-hosted Retrieval-Augmented Generation (RAG) system that enables grounded question-answering over PDF documents using local LLMs and vector databases.

## Overview

**lr-rag** combines document ingestion, semantic search, and generative AI to answer user questions with citations sourced directly from uploaded documents. All processing runs locally—no cloud dependencies, full data privacy.

### Key Features

- **Local-First Architecture**: Runs entirely on-premises using Ollama and Qdrant
- **PDF Ingestion Pipeline**: Automatic text extraction, chunking, and embedding
- **Semantic Search**: Vector-based retrieval with configurable similarity thresholds
- **Grounded Answers**: LLM-generated responses with source citations
- **Streamlit UI**: Single-page web demo for upload, ingest, and Q&A workflows
- **CLI Interface**: Command-line tools for batch ingestion and querying

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM & Embeddings** | Ollama | Latest | Local language model inference and embedding generation |
| **Vector Database** | Qdrant | 1.16.2+ | Semantic search and vector storage |
| **Framework** | LlamaIndex | 0.14.12 | Document parsing, chunking, and orchestration |
| **Web UI** | Streamlit | 1.53.1 | Interactive demo interface |
| **Language** | Python | 3.9+ | Core implementation |
| **Package Manager** | uv | Latest | Fast Python dependency management |

### Dependencies (Full List)

See [requirements.txt](requirements.txt) for complete pinned versions. Core libraries:
- `llama-index-core`: Document processing and RAG pipeline
- `llama-index-readers-file`: PDF extraction via PDFReader
- `qdrant-client`: Vector database client
- `pydantic`: Data validation and settings management
- `python-dotenv`: Environment variable management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Upload PDF  →  Extract Text  →  Chunk Text  →  Embed Chunks  │
│  (Streamlit)    (PDFReader)     (LlamaIndex)    (Ollama)       │
│                                                      ↓          │
│                                            Upsert to Qdrant    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval & QA Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  User Question  →  Embed Query  →  Semantic Search  →  Context │
│  (Web/CLI)         (Ollama)        (Qdrant)         Assembly   │
│                                                      ↓          │
│                                      Generate Answer (Ollama)  │
│                                           ↓                    │
│                                      Return Answer + Sources   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
lr-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py         # PDF text extraction
│   │   ├── chunker.py        # Text splitting strategies
│   │   ├── embedder.py       # Embedding generation
│   │   └── upsert.py         # Qdrant vector storage
│   ├── retrieval/
│   │   ├── query_embedder.py # Query embedding
│   │   ├── search.py         # Semantic similarity search
│   │   └── answer.py         # LLM-based answer generation
│   ├── services/
│   │   └── ollama.py         # Ollama client wrapper
│   └── config/
│       └── settings.py       # Configuration management
├── cli.py                     # Command-line interface
├── streamlit_app.py          # Web UI
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB+ recommended for large PDFs)
- **Disk**: 10GB for Qdrant + model caches
- **OS**: macOS, Linux, or Windows (WSL2)

### External Services

1. **Ollama** (LLM + Embeddings)
   - Download: [ollama.ai](https://ollama.ai)
   - Default endpoint: `http://localhost:11434`
   - Models: `mistral`, `neural-chat`, `nomic-embed-text` (recommended for embeddings)

2. **Qdrant** (Vector Database)
   - Install via Docker: `docker run -p 6333:6333 qdrant/qdrant`
   - Default endpoint: `http://localhost:6333`
   - Dashboard: `http://localhost:6333/dashboard`

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dinukadilshan03/lr-rag.git
cd lr-rag
```

### 2. Install Python Dependencies

Using `uv` (recommended for speed):

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or with `pip`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Services

**Ollama** (in a separate terminal):
```bash
ollama serve
```

**Qdrant** (in another terminal):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Configure Environment Variables

Create a `.env` file in the repo root:

```env
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# Qdrant configuration
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=lr_rag_documents

# Optional: Vector dimension (must match embedding model output)
VECTOR_DIMENSION=384
```

---

## Usage

### Option 1: Streamlit Web UI (Recommended for Demo)

Start the interactive web app:

```bash
uv run streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

**Workflow**:
1. Upload a PDF in the "Ingest a PDF" section
2. Click "Ingest PDF" to process and store vectors
3. Enter a question in the "Ask a Question" section
4. Click "Ask" to retrieve relevant chunks and generate an answer
5. View sources and chunk details in expandable sections

### Option 2: Command-Line Interface

**Ingest a PDF**:
```bash
uv run python cli.py ingest /path/to/document.pdf
```

**Ask a Question**:
```bash
uv run python cli.py ask "What is a posting list?"
```

Output includes:
- Generated answer grounded in document content
- Source citations with page references
- Similarity scores (if enabled)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `LLM_MODEL` | `mistral` | LLM model name for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vectors |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server endpoint |
| `COLLECTION_NAME` | `lr_rag_documents` | Qdrant collection name |
| `VECTOR_DIMENSION` | `384` | Embedding dimension (model-dependent) |
| `CHUNK_SIZE` | `512` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks for context continuity |
| `TOP_K` | `5` | Number of chunks to retrieve per query |

### Tuning Parameters

**For better retrieval quality**:
- Increase `CHUNK_SIZE` (512–1024) for longer context windows
- Increase `TOP_K` (5–10) for more comprehensive source diversity
- Lower `CHUNK_OVERLAP` only if documents are repetitive

**For faster ingestion**:
- Decrease `CHUNK_SIZE` but risk losing coherence
- Use lighter embedding models (e.g., `all-minilm` for 384-dim vectors)

---

## API Reference

### Core Functions

#### Ingestion Pipeline

```python
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_nodes
from src.ingestion.upsert import upsert_nodes

# Load PDF text
text = load_pdf("document.pdf")

# Split into chunks
nodes = chunk_text(text)

# Generate embeddings
embedded_nodes = embed_nodes(nodes)

# Store in Qdrant
upsert_nodes(embedded_nodes, source_id="document.pdf")
```

#### Retrieval & QA

```python
from src.retrieval.query_embedder import embed_query
from src.retrieval.search import search_similar
from src.retrieval.answer import generate_answer

# Embed user query
query_vector = embed_query("What is information retrieval?")

# Find similar chunks
retrieved_chunks = search_similar(query_vector)

# Generate grounded answer
answer, sources = generate_answer("What is information retrieval?", retrieved_chunks)
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```

---

## Performance & Optimization

### Benchmarks (on M1 MacBook Pro, 16GB RAM)

| Task | Time | Notes |
|------|------|-------|
| Ingest 50-page PDF | ~15s | Including chunking + embedding + upload |
| Semantic search (Top-5) | ~200ms | Qdrant query only |
| Answer generation | ~3–5s | Ollama LLM inference |
| **Total Q&A latency** | **~3.5–5.5s** | End-to-end |

### Optimization Tips

1. **Use GPU acceleration**: Offload Ollama to GPU for 2–3x speedup
   ```bash
   ollama serve --gpu-layers 50
   ```

2. **Batch ingest**: Process multiple PDFs in parallel
   ```bash
   for pdf in *.pdf; do uv run python cli.py ingest "$pdf"; done
   ```

3. **Reduce chunk overlap**: Set `CHUNK_OVERLAP=0` if documents don't need context bridging

4. **Prune old data**: Periodically delete low-relevance vectors from Qdrant to save memory

---

## Troubleshooting

### Issue: `Connection refused` to Ollama

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

Check endpoint: `curl http://localhost:11434/api/tags`

### Issue: Empty or hallucinated answers

**Solution**:
1. Verify retrieval is working: Check `retrieved_chunks` in logs
2. Adjust `TOP_K` to retrieve more context
3. Increase `CHUNK_SIZE` for longer text passages
4. Confirm PDF quality (scanned images won't extract text well)

### Issue: Qdrant collection not found

**Solution**:
1. Restart Qdrant: `docker restart <container_id>`
2. Re-ingest PDFs: `uv run python cli.py ingest document.pdf`
3. Dashboard: Visit `http://localhost:6333/dashboard` to verify collection exists

### Issue: High latency on Mac with M1/M2

**Solution**:
1. Ensure Ollama is using GPU: Check `ollama serve` output for GPU status
2. Reduce `VECTOR_DIMENSION` by using a smaller embedding model
3. Increase `CHUNK_SIZE` to reduce number of chunks per document

---

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Quality

Format code with Black:
```bash
uv run black src/ cli.py streamlit_app.py
```

Lint with Ruff:
```bash
uv run ruff check src/
```

### Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test locally
3. Commit: `git commit -m "Add feature description"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

---

## Future Roadmap

- [ ] Support for multiple document formats (DOCX, TXT, Markdown)
- [ ] Advanced chunking strategies (semantic chunking, sliding window)
- [ ] Multi-language support with cross-lingual embeddings
- [ ] Chat history and conversation memory
- [ ] API server (FastAPI) for production deployments
- [ ] Fine-tuning pipeline for domain-specific LLMs
- [ ] Web crawler for dynamic document ingestion

---

## License

MIT License. See LICENSE for details.

---

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/dinukadilshan03/lr-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dinukadilshan03/lr-rag/discussions)

---

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for document orchestration
- [Ollama](https://ollama.ai) for local LLM inference
- [Qdrant](https://qdrant.tech) for vector database
- [Streamlit](https://streamlit.io) for interactive UI