# Doc Query AI

**Version 2.1.0**

A document question-answering app built with retrieval-augmented generation (RAG). Upload multiple PDFs, index them into a local document library, and ask natural-language questions grounded in your content — with answers that cite their source documents.

## Features

- **Multi-PDF library** — index many PDFs together; new uploads are added incrementally without wiping existing documents
- **Source attribution** — every chunk is tagged with its document and page, and answers cite the source(s) used
- **Scoped search** — restrict the chat to a chosen subset of documents, or search the whole library
- **MMR retrieval** — Maximal Marginal Relevance returns relevant *and* diverse chunks for better multi-document recall
- **Vector search** — embed chunks with Google Gemini and store them in ChromaDB
- **Resilient embeddings** — automatic retry/back-off on Gemini rate limits (429), with a clear message if quota is exhausted
- **Gradio web UI** — chat interface with multi-file upload, a live document table, and per-document removal
- **CLI mode** — interactive terminal chatbot that reports the current library contents
- **Idempotent re-indexing** — re-uploading the same file refreshes it in place (matched by content hash)
- **Unit tests** — pytest suite with mocked dependencies (no API key required)

## Stack

| Component | Technology |
|-----------|------------|
| Vector store | [ChromaDB](https://www.trychroma.com/) via `langchain-chroma` |
| Orchestration | [LangChain](https://python.langchain.com/) |
| Embeddings | Google Gemini `gemini-embedding-2` |
| LLM | Google Gemini `gemini-2.5-flash` |
| Web UI | [Gradio](https://gradio.app/) 6.x |

## Models

| Role | Model | Configured in |
|------|-------|---------------|
| Embeddings | `gemini-embedding-2` | `rag/ingest.py` |
| Answer generation | `gemini-2.5-flash` | `rag/chain.py` |

Both require a [Google AI API key](https://aistudio.google.com/apikey). Free-tier quotas apply separately per model.

## Prerequisites

- **Python 3.12** (recommended). Python 3.14 is not supported by key dependencies such as `onnxruntime`.
- A Google AI API key for Gemini

## Setup

```bash
# Clone and enter the project
git clone https://github.com/le-koj/doc-query-ai.git
cd doc-query-ai

# Create a virtual environment (Python 3.12)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Do not commit `.env` — it is listed in `.gitignore`.

Optionally place a seed PDF at `documents/TechCorp_Official_Employee_Handbook.pdf` to index automatically on first run. Otherwise the app starts empty and waits for you to upload one or more PDFs through the UI.

## Usage

### Web UI (recommended)

```bash
source venv/bin/activate
python app.py
```

Open the URL shown in the terminal (default: `http://127.0.0.1:7860`).

- Upload one or more PDFs and click **Index documents** to add them to the library
- The **Indexed documents** table shows each document and its chunk count
- Use **Remove a document** to delete a single document from the library
- Use **Search in** to restrict the chat to selected documents (leave empty to search everything)
- Ask questions in the chat panel — answers cite the source document(s) and page(s) used
- On subsequent runs, the persisted store in `chroma_db/` is reused — no re-embedding unless you add or refresh a file

### CLI

```bash
source venv/bin/activate
python rag_app.py
```

Type your questions at the prompt. Enter `exit`, `quit`, or `bye` to close the session.

## Testing

```bash
source venv/bin/activate
pytest        # run all unit tests
pytest -v     # verbose output
```

The test suite covers pure logic and guard clauses with mocks — no API key required.

| File | What it covers |
|------|----------------|
| `test_chain.py` | `_format_doc_label`, `_format_docs`, `build_rag_chain` wiring |
| `test_ingest.py` | `load_vector_store` branching, `_load_chunks` metadata, `add_pdf`, `remove_pdf`, `list_documents` |
| `test_app.py` | `answer()` with source attribution, `ingest_pdfs()`, Gradio upload path resolution |

Integration tests that call Gemini or ChromaDB with real embeddings can be marked with `@pytest.mark.integration` and skipped in CI.

## How it works

```
PDF(s) → chunk + tag (doc_id, source, page) → embed (gemini-embedding-2) → ChromaDB library
                                                                                  ↓
User question → retrieve top-k chunks across all docs → prompt + gemini-2.5-flash → answer + sources
```

1. **Ingest** (`rag/ingest.py`) — loads each PDF, splits it into 500-character chunks with 50-character overlap, tags every chunk with `doc_id` (content hash), `source` filename, `page`, and `chunk_index`, embeds with `gemini-embedding-2`, and appends to `./chroma_db`. `add_pdf` grows the library; `remove_pdf` and `list_documents` manage it.
2. **Chain** (`rag/chain.py`) — retrieves chunks with MMR (top 6 from a pool of 20, balancing relevance and diversity), optionally filtered to selected `doc_id`s, formats them with source labels, fills a prompt template, and sends it to `gemini-2.5-flash` for a concise, source-citing answer
3. **UI** (`app.py`) — Gradio chat interface with multi-file upload, per-document removal, and a **Search in** selector to scope queries; documents are added incrementally and re-uploads refresh in place by `doc_id` without wiping other documents

## Project layout

```
doc-query-ai/
├── app.py              # Gradio web UI entry point
├── rag_app.py          # CLI chatbot entry point
├── rag/
│   ├── __init__.py
│   ├── ingest.py       # PDF loading, chunking, embedding, re-indexing
│   └── chain.py        # Retriever + prompt + LLM chain
├── tests/
│   ├── conftest.py     # Shared pytest fixtures
│   ├── test_app.py
│   ├── test_chain.py
│   └── test_ingest.py
├── documents/          # Source PDFs (gitignored)
├── chroma_db/          # ChromaDB persistence (gitignored)
├── pytest.ini
├── requirements.txt
├── .env                # API keys (gitignored)
├── .gitignore
└── README.md
```

## Changelog

### 2.1.0

- Scoped search: restrict the chat to selected documents via a `doc_id` metadata filter
- MMR retrieval (`fetch_k=20`, `lambda_mult=0.5`) for relevant, diverse multi-document context
- Resilient embeddings: retry with back-off on Gemini 429 rate limits; `EmbeddingQuotaError` surfaced cleanly in the UI
- Per-document **Remove** button and a synced **Search in** multi-select in the UI
- Graceful error handling in chat and ingestion instead of tracebacks
- Expand test suite to 53 tests

### 2.0.0

- **Multi-PDF library**: index many PDFs together; uploads are additive instead of replacing the store
- Tag every chunk with `doc_id`, `source`, `page`, and `chunk_index` metadata
- Source-aware retrieval and prompt: answers cite the document(s) and page(s) used
- Raise retrieval to top-6 chunks for better cross-document recall
- New ingest API: `add_pdf`, `remove_pdf`, `list_documents`, `compute_doc_id` (replaces `reindex_vector_store`)
- Multi-file upload in the UI plus a live **Indexed documents** table
- Idempotent re-indexing keyed on file content hash
- Expand test suite to 33 tests

### 1.1.0

- Add Gradio web UI with chat interface and PDF upload/re-indexing
- Refactor into `rag/` package (`ingest.py`, `chain.py`)
- Switch to `gemini-embedding-2` embeddings and `gemini-2.5-flash` LLM
- Fix ChromaDB re-index crash by resetting collections in place
- Add pytest unit test suite (18 tests)
- Add `langchain-chroma`, Gradio, and pytest dependencies

## License

MIT
