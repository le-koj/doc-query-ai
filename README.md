# Doc Query AI

A document question-answering app built with retrieval-augmented generation (RAG). Upload PDFs, index them in a local vector store, and ask natural-language questions grounded in your content.

## Features

- **PDF ingestion** — load and chunk PDF documents with LangChain
- **Vector search** — embed chunks with Google Gemini and store them in ChromaDB
- **Grounded answers** — retrieve relevant context and generate responses with Gemini
- **Gradio web UI** — chat interface with example questions and PDF upload/re-indexing
- **CLI mode** — interactive terminal chatbot for quick testing

## Stack

- **[ChromaDB](https://www.trychroma.com/)** — local vector database for document embeddings
- **[LangChain](https://python.langchain.com/)** — orchestration and retrieval pipeline
- **[Google Gemini](https://ai.google.dev/)** (`langchain-google-genai`) — embeddings and LLM for answer generation
- **[Gradio](https://gradio.app/)** — browser-based chat UI

## Prerequisites

- **Python 3.12** (recommended). Python 3.14 is not yet supported by key dependencies such as `onnxruntime`.
- A [Google AI API key](https://aistudio.google.com/apikey) for Gemini

## Setup

```bash
# Clone and enter the project
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
- Upload a new PDF and click **Index document** to replace the current vector store
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
pytest
```

The test suite covers pure logic and guard clauses with mocks — no API key required. Tests live in `tests/`:

| File | What it covers |
|------|----------------|
| `test_chain.py` | `_format_docs`, `build_rag_chain` wiring |
| `test_ingest.py` | `load_vector_store` branching, `build_vector_store` chunking |
| `test_app.py` | `answer()` and `ingest_pdf()` handlers |

Integration tests that call Gemini or ChromaDB with real embeddings can be marked with `@pytest.mark.integration` and skipped in CI.

## How it works

```
PDF → chunk → embed (Gemini) → ChromaDB
                                    ↓
User question → retrieve top-k chunks → prompt + Gemini → answer
```

1. **Ingest** (`rag/ingest.py`) — loads a PDF, splits it into 500-character chunks with 50-character overlap, embeds them with `gemini-embedding-2`, and persists to `./chroma_db`
2. **Chain** (`rag/chain.py`) — retrieves the top 2 most similar chunks, fills a prompt template, and sends it to `gemini-3.5-flash` for a concise answer
3. **UI** (`app.py`) — Gradio chat interface wired to the chain; supports live PDF re-indexing

## Project layout

```
doc-query-ai/
├── app.py              # Gradio web UI entry point
├── rag_app.py          # CLI chatbot entry point
├── rag/
│   ├── ingest.py       # PDF loading, chunking, embedding, ChromaDB persistence
│   └── chain.py        # Retriever + prompt + LLM chain
├── tests/              # Pytest unit tests
├── documents/          # Source PDFs (gitignored)
├── chroma_db/          # ChromaDB persistence (gitignored)
├── requirements.txt
├── .env                # API keys (gitignored)
└── README.md
```

## License

MIT (or update as needed)
