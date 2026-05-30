"""Document ingestion utilities for the RAG pipeline.

Loads PDF files, splits them into text chunks, embeds them with Google
Gemini, and persists the vectors to a local ChromaDB store.
"""

import os  # Check whether the ChromaDB directory already exists

from langchain_community.document_loaders import PyPDFLoader  # Extract text from PDF files
from langchain_chroma import Chroma  # Local vector database for document embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embedding model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks

CHROMA_DIR = "./chroma_db"  # Directory where ChromaDB persists its SQLite database
CHUNK_SIZE = 500  # Maximum number of characters per text chunk
CHUNK_OVERLAP = 50  # Number of overlapping characters between adjacent chunks
EMBEDDING_MODEL = "gemini-embedding-2"  # Google Gemini multimodal embedding model (GA)


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a Google Gemini embeddings client.

    Returns:
        GoogleGenerativeAIEmbeddings: Configured embedding model instance.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def load_vector_store(pdf_path: str) -> Chroma:
    """Load an existing Chroma store or build one from a PDF.

    If a persisted ChromaDB directory already exists and is non-empty, the
    existing store is loaded to avoid re-embedding on every startup. Otherwise,
    the PDF at ``pdf_path`` is ingested from scratch.

    Args:
        pdf_path: Filesystem path to the source PDF document.

    Returns:
        Chroma: A Chroma vector store ready for retrieval queries.
    """
    embeddings = _build_embeddings()  # Initialise the Gemini embedding model

    # Reuse the persisted store when available to skip re-embedding
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    return build_vector_store(pdf_path, embeddings)  # First run — ingest the PDF


def build_vector_store(
    pdf_path: str,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> Chroma:
    """Ingest a PDF: load, chunk, embed, and persist to Chroma.

    Args:
        pdf_path: Filesystem path to the PDF file to index.
        embeddings: Optional pre-built embedding model. A new instance is
            created when ``None``.

    Returns:
        Chroma: A freshly built and persisted Chroma vector store.
    """
    if embeddings is None:
        embeddings = _build_embeddings()  # Create embeddings client if not supplied

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)  # Point the loader at the target PDF
    documents = loader.load()  # Parse the PDF into page-level Document objects

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )  # Configure the text splitter with size and overlap settings
    chunks = splitter.split_documents(documents)  # Split pages into smaller chunks
    print(f"  → {len(chunks)} chunks")

    print("Embedding and persisting vector store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )  # Embed chunks and write them to the local ChromaDB directory
    print("  → Done")
    return vector_db
