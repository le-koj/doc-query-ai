# Doc Query AI

**Version 1.1.0**

A document question-answering app built with retrieval-augmented generation (RAG). Upload PDFs, index them in a local vector store, and ask natural-language questions grounded in your content.

## Features

- **PDF ingestion** — load and chunk PDF documents with LangChain
- **Vector search** — embed chunks with Google Gemini and store them in ChromaDB
- **Grounded answers** — retrieve relevant context and generate responses with Gemini
- **Gradio web UI** — chat interface with example questions and live PDF re-indexing
- **CLI mode** — interactive terminal chatbot for quick testing
- **In-place re-indexing** — upload a new PDF in the UI without breaking the ChromaDB client
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

Place your source PDF in the `documents/` directory. The default document loaded at startup is `documents/TechCorp_Official_Employee_Handbook.pdf`.

## Usage

### Web UI (recommended)

```bash
source venv/bin/activate
python app.py
```

Open the URL shown in the terminal (default: `http://127.0.0.1:7860`).

- Ask questions in the chat panel
- Upload a new PDF and click **Index document** to replace the indexed content
- On subsequent runs, the persisted store in `chroma_db/` is reused — no re-embedding unless you upload a new file

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
| `test_chain.py` | `_format_docs`, `build_rag_chain` wiring |
| `test_ingest.py` | `load_vector_store` branching, `build_vector_store`, `reindex_vector_store` |
| `test_app.py` | `answer()`, `ingest_pdf()`, Gradio upload path resolution |

Integration tests that call Gemini or ChromaDB with real embeddings can be marked with `@pytest.mark.integration` and skipped in CI.

## How it works

```
PDF → chunk → embed (gemini-embedding-2) → ChromaDB
                                                ↓
User question → retrieve top-k chunks → prompt + gemini-2.5-flash → answer
```

1. **Ingest** (`rag/ingest.py`) — loads a PDF, splits it into 500-character chunks with 50-character overlap, embeds them with `gemini-embedding-2`, and persists to `./chroma_db`
2. **Chain** (`rag/chain.py`) — retrieves the top 2 most similar chunks, fills a prompt template, and sends it to `gemini-2.5-flash` for a concise answer
3. **UI** (`app.py`) — Gradio chat interface wired to the chain; re-indexes documents in place via `reset_collection()` without deleting the ChromaDB directory

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

### 1.1.0

- Add Gradio web UI with chat interface and PDF upload/re-indexing
- Refactor into `rag/` package (`ingest.py`, `chain.py`)
- Switch to `gemini-embedding-2` embeddings and `gemini-2.5-flash` LLM
- Fix ChromaDB re-index crash by resetting collections in place
- Add pytest unit test suite (18 tests)
- Add `langchain-chroma`, Gradio, and pytest dependencies

## License

MIT
