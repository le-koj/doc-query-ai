# Doc Query AI

A document question-answering app built with retrieval-augmented generation (RAG). Upload documents, index them in a vector store, and ask natural-language questions grounded in your content.

## Stack

- **[ChromaDB](https://www.trychroma.com/)** — local vector database for document embeddings
- **[LangChain](https://python.langchain.com/)** — orchestration and retrieval pipeline
- **[Google Gemini](https://ai.google.dev/)** (`langchain-google-genai`) — LLM for generating answers

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

## Project layout

```
doc-query-ai/
├── data/           # Source documents (gitignored)
├── chroma_db/      # ChromaDB persistence (gitignored)
├── requirements.txt
└── README.md
```

Application code will be added as the project grows.

## License

MIT (or update as needed)
