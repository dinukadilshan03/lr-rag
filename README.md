# lr-rag: Local Retrieval-Augmented Generation

A self-hosted Retrieval-Augmented Generation (RAG) system for grounded question-answering over PDF documents using local LLMs and vector databases. All processing runs locally—no cloud dependencies, full data privacy.


![Demo](assets/RAG-app.gif)

## 🎯 What It Does

**lr-rag** enables users to:
1. **Upload and ingest PDF documents** - automatically extracts text, chunks content, generates embeddings
2. **Ask natural language questions** about the document content
3. **Receive grounded answers** with citations directly from source material

The system uses semantic search to retrieve document chunks most relevant to the user's query, feeds them as context to a local LLM, and generates answers grounded in actual document content (avoiding hallucinations).

---

## 📦 Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Backend LLM** | Ollama | Latest | Local LLM inference engine (no cloud calls) |
| **Embedding Model** | nomic-embed-text | - | 768-dimensional dense embeddings for semantic search |
| **Language Model** | llama3 | - | 7B parameter model for QA reasoning |
| **Vector Database** | Qdrant | 1.16.2+ | Vector storage with cosine similarity search |
| **Document Framework** | LlamaIndex | 0.14.12+ | PDF parsing, text chunking, pipeline orchestration |
| **Web Interface** | Streamlit | 1.53.1+ | Interactive single-page application |
| **Core Language** | Python | 3.12+ | Type-safe implementation with Pydantic |
| **HTTP Client** | httpx | 0.28.1+ | REST API communication |

### Ollama Models in Depth

#### **nomic-embed-text** (Embedding Model)

- **Architecture**: Dense vector embeddings (state-of-the-art for information retrieval)
- **Output Dimension**: 768 dimensions per embedding
- **Model Size**: 274MB
- **Inference Speed**: ~5-10ms per chunk on modern hardware
- **Distance Metric**: Cosine similarity (normalized dot product, range 0–1)
  - Score 1.0 = identical vectors
  - Score 0.5+ = considered relevant (configurable threshold)
  - Score 0.0 = orthogonal/unrelated

**Purpose in Pipeline:**
- Converts raw text (PDFs, queries) into comparable vectors in shared semantic space
- Enables similarity-based retrieval independent of keyword matching
- Same model used for both documents and queries ensures consistent representations

#### **llama3** (Language Model)

- **Architecture**: Transformer-based decoder with 7 billion parameters
- **Model Size**: ~3.8GB (4-bit quantized, CPU/GPU compatible)
- **Context Window**: 8,192 tokens
- **Inference**: Temperature=0.7 (controlled randomness for consistent outputs)
- **Quantization**: 4-bit for efficient local execution

**Purpose in Pipeline:**
- Reads retrieved document context + user question
- Generates coherent, contextually-relevant answers
- System prompt constrains responses to ONLY use provided context (grounding mechanism)

**Grounding Mechanism:**
```
System Prompt: "You are a document-grounded assistant.
You MUST answer using ONLY the information in the provided context.
If the context does NOT contain the answer, reply exactly:
'I cannot find this information in the provided documents.'
Do NOT use prior knowledge."
```

---

## 🏗️ Architecture

### Data Flow: Ingestion Pipeline

```
PDF Upload (Streamlit)
    ↓
┌─────────────────────────────────────────────────────┐
│ 1. Load PDF                      (src/ingestion/loader.py)    │
│    └─ PDFReader extracts text from all pages                  │
│    └─ Returns: single string of concatenated document text   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2. Chunk Text                   (src/ingestion/chunker.py)    │
│    └─ SentenceSplitter.split() with:                          │
│       • chunk_size = 200 tokens                               │
│       • chunk_overlap = 40 tokens                             │
│    └─ Returns: List[TextNode] with text, metadata            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3. Generate Embeddings           (src/ingestion/embedder.py)  │
│    └─ For each TextNode:                                      │
│       • POST to Ollama /api/embeddings                        │
│       • Model: nomic-embed-text                               │
│       • Returns: 768-dimensional vector                       │
│    └─ Attaches embedding to each node                        │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 4. Upsert to Qdrant              (src/ingestion/upsert.py)    │
│    └─ For each embedded node:                                 │
│       • Generate deterministic UUID5(source_id + text)       │
│       • POST to Qdrant /collections/lr_rag_documents/upsert │
│       • Point = {id, vector[768], payload{text, source}}    │
│    └─ Deduplication: same PDF re-ingested = updates          │
└─────────────────────────────────────────────────────┘
    ↓
Qdrant Collection (Cosine Distance Index)
```

### Data Flow: Retrieval & QA Pipeline

```
User Question (Streamlit/CLI)
    ↓
┌─────────────────────────────────────────────────────┐
│ 1. Embed Query                   (src/retrieval/query_embedder.py) │
│    └─ POST user question to Ollama /api/embeddings │
│    └─ Model: nomic-embed-text (same as documents)  │
│    └─ Returns: 768-dimensional vector              │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2. Semantic Search                (src/retrieval/search.py)   │
│    └─ POST query vector to Qdrant /collections/.../query   │
│    └─ Qdrant returns: Top-K points sorted by cosine score  │
│    └─ Filter: score >= 0.5 (configurable threshold)        │
│    └─ Returns: List[{text, source, score}]                 │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3. Build Context                  (src/retrieval/answer.py)   │
│    └─ Format retrieved chunks as numbered sources:           │
│       "[Source 1]\n{chunk1_text}\n\n[Source 2]..."         │
│    └─ Create source_map: {index → filename}                │
│    └─ Returns: context string + citation map               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 4. Generate Grounded Answer      (src/retrieval/answer.py)   │
│    └─ Assemble messages:                                     │
│       • System: grounding prompt (use ONLY context)         │
│       • Context: retrieved chunks                           │
│       • User: original question                             │
│    └─ POST to Ollama /api/chat                              │
│    └─ Model: llama3                                         │
│    └─ Returns: natural language answer                      │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 5. Return Answer + Citations                        │
│    └─ Display answer text                                   │
│    └─ Show source citations: [1] filename, [2] filename ... │
│    └─ Option to inspect raw retrieved chunks + scores      │
└─────────────────────────────────────────────────────┘
    ↓
User sees grounded answer with sources
```

---

## 🔄 Workflow Summary

### High-Level Process

```
INGESTION:  PDF  →  Extract  →  Chunk  →  Embed  →  Store in Qdrant
                    (text)      (200tok) (768dim)   (indexed by cosine)

RETRIEVAL:  Question  →  Embed Query  →  Search Qdrant  →  Retrieve chunks
                         (768dim)        (cosine ≥0.5)      (Top-K)

QA:         Context + Question  →  System Prompt + Llama3  →  Grounded Answer
            (number of sources)      (no hallucinations)      (with citations)
```

### Key Design Decisions

1. **Same Embedding Model for Queries & Documents**
   - Ensures query and document vectors exist in same semantic space
   - Enables direct similarity comparison

2. **Cosine Similarity with 0.5 Threshold**
   - Cosine distance: normalized, scale-invariant (0–1)
   - Threshold 0.5: filters low-relevance chunks, reduces noise

3. **Deterministic UUIDs (UUIDv5)**
   - Deduplicates: re-ingesting same PDF doesn't create duplicate points
   - ID = UUID5(namespace, source_id + chunk_text)

4. **Overlapping Chunks (200 tokens, 40-token overlap)**
   - Overlap prevents semantic fragmentation at chunk boundaries
   - Improves retrieval of concepts spanning multiple chunks

5. **System Prompt Grounding**
   - Instructs LLM to refuse answering if context lacks information
   - Reduces hallucinations and false claims

---

## 📂 Codebase Structure

```
src/
├── ingestion/
│   ├── loader.py         # PDFReader wrapper
│   ├── chunker.py        # SentenceSplitter with overlap
│   ├── embedder.py       # Ollama embedding calls
│   └── upsert.py         # Qdrant upsert with deterministic IDs
├── retrieval/
│   ├── query_embedder.py # Query embedding via Ollama
│   ├── search.py         # Qdrant semantic search
│   └── answer.py         # Context assembly + LLM answer generation
├── services/
│   ├── ollama.py         # HTTP client for /api/embeddings, /api/chat
│   └── qdrant.py         # Qdrant client with collection management
├── config/
│   └── settings.py       # Pydantic config from .env
└── models/
    └── schemas.py        # Data schemas (TextNode, etc.)

cli.py                     # CLI: `ingest <pdf>` and `ask <question>`
streamlit_app.py          # Web UI: upload, ingest, ask in browser
```

---

## 🔌 External Service Endpoints

### Ollama Endpoints Used

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/api/embeddings` | POST | Generate embeddings | `{model, prompt}` | `{embedding: [768 dims]}` |
| `/api/chat` | POST | Chat completion | `{model, messages, stream}` | `{message: {content}}` |
| `/api/tags` | GET | List models | - | `{models: [...]}` |

### Qdrant Endpoints Used

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/collections` | GET | List collections | - | `{collections: [...]}` |
| `/collections/{name}/points/upsert` | POST | Insert/update points | `{points: [...]}` | `{operation_id}` |
| `/collections/{name}/points/query` | POST | Semantic search | `{query: [768 dims], limit}` | `{points: [...]}` |

---

## ⚙️ Configuration Parameters

**From `src/config/settings.py`:**

```python
QDRANT_URL = "http://localhost:6333"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"    # Always 768-dim output
LLM_MODEL = "llama3"                    # 7B, can swap for other models
COLLECTION_NAME = "lr_rag_documents"    # Qdrant collection
TOP_K = 5                               # Retrieve top 5 chunks per query
VECTOR_DIMENSION = 768                  # Fixed for nomic-embed-text
```

**Tuning Parameters (in code):**

- **Chunking** (`src/ingestion/chunker.py`): `chunk_size=200`, `chunk_overlap=40` tokens
- **Similarity threshold** (`src/retrieval/search.py`): `score >= 0.5` (cosine)
- **System prompt** (`src/services/ollama.py`): Controls LLM grounding behavior

---

## 🧠 How Embeddings Work

**Embedding Space Visualization (Conceptual):**

```
3D Embedding Space (actually 768D):

        "machine learning"
              ↑
              |  /  "neural networks"
              | /
    "data"---•---"training"
            /
           /
    "algorithm"
```

**Cosine Similarity Calculation:**

```
Given:
  doc_vector = nomic-embed-text("Posting lists store inverted indices...")
  query_vector = nomic-embed-text("What is an inverted index?")

Cosine Similarity = (doc_vector · query_vector) / (||doc_vector|| × ||query_vector||)
                  = 0.87  (on scale 0–1)

Interpretation:
  0.87 >= 0.5 threshold → MATCH, retrieve this chunk
```

---

## 📊 Information Flow Example

**User Question:** "What does cosine mean?"

**Step 1 – Query Embedding:**
```
Question: "What does cosine mean?"
↓ (Ollama nomic-embed-text)
Vector: [0.12, -0.34, 0.78, ..., 0.45]  (768 values)
```

**Step 2 – Search:**
```
Qdrant searches collection for vectors closest to query vector
Found chunks:
  Chunk 1 (score 0.89): "Cosine similarity is a metric used in..."
  Chunk 2 (score 0.76): "In machine learning, distance metrics..."
  Chunk 3 (score 0.44): "Other metrics include Euclidean..."  ← filtered (< 0.5)
```

**Step 3 – Answer Generation:**
```
System Prompt: "You are document-grounded..."
Context: "[Source 1] Cosine similarity is a metric..."
User: "What does cosine mean?"
↓ (Ollama llama3)
Answer: "According to the documents, cosine similarity is a metric..."
```

---

## 🚀 Interfaces

### CLI
```bash
uv run python cli.py ingest /path/to/document.pdf
uv run python cli.py ask "What does this document discuss?"
```

### Web UI (Streamlit)
```bash
uv run streamlit run streamlit_app.py
# Opens http://localhost:8501
```

---

**Architecture Version:** 1.0  
**Last Updated:** January 27, 2026
