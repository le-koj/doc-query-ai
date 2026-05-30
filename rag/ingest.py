"""Document ingestion utilities for the RAG pipeline.

Loads PDF files, splits them into text chunks, embeds them with Google
Gemini, and persists the vectors to a local ChromaDB store.
"""

import os  # Check whether the ChromaDB directory already exists

from langchain_community.document_loaders import PyPDFLoader  # Extract text from PDF files
from langchain_chroma import Chroma  # Local vector database for document embeddings
from langchain_core.documents import Document  # Typed document chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embedding model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks

CHROMA_DIR = "./chroma_db"  # Directory where ChromaDB persists its SQLite database
CHUNK_SIZE = 500  # Maximum number of characters per text chunk
CHUNK_OVERLAP = 50  # Number of overlapping characters between adjacent chunks
EMBEDDING_MODEL = "gemini-embedding-2"  # Google Gemini multimodal embedding model (GA)


class PdfIngestError(ValueError):
    """Raised when a PDF cannot be indexed because it has no usable text."""


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a Google Gemini embeddings client.

    Returns:
        GoogleGenerativeAIEmbeddings: Configured embedding model instance.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def _load_chunks(pdf_path: str) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        list[Document]: The PDF content split into chunk-sized documents.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)  # Point the loader at the target PDF
    documents = loader.load()  # Parse the PDF into page-level Document objects
    print(f"  → {len(documents)} pages")

    if not documents:
        raise PdfIngestError("The PDF has no pages.")

    if not any(doc.page_content.strip() for doc in documents):
        raise PdfIngestError(
            "No extractable text found in this PDF. "
            "It may be image-only or scanned — use a PDF with selectable text."
        )

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )  # Configure the text splitter with size and overlap settings
    chunks = splitter.split_documents(documents)  # Split pages into smaller chunks
    print(f"  → {len(chunks)} chunks")

    if not chunks:
        raise PdfIngestError(
            "The PDF produced no text chunks after splitting. "
            "Try a document with more readable text content."
        )

    return chunks


def load_vector_store(pdf_path: str) -> Chroma | None:
    """Load an existing Chroma store or build one from a PDF.

    If a persisted ChromaDB directory already exists and is non-empty, the
    existing store is loaded to avoid re-embedding on every startup. Otherwise,
    the PDF at ``pdf_path`` is ingested from scratch when that file exists.

    Args:
        pdf_path: Filesystem path to the source PDF document.

    Returns:
        Chroma | None: A Chroma vector store ready for retrieval queries, or
            ``None`` when no store exists and ``pdf_path`` is missing.
    """
    embeddings = _build_embeddings()  # Initialise the Gemini embedding model

    # Reuse the persisted store when available to skip re-embedding
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    if not os.path.isfile(pdf_path):
        print(f"No vector store found and PDF missing at {pdf_path}")
        return None

    return build_vector_store(pdf_path, embeddings)  # First run — ingest the PDF


def build_vector_store(
    pdf_path: str,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """Ingest a PDF: load, chunk, embed, and persist to Chroma.

    Args:
        pdf_path: Filesystem path to the PDF file to index.
        embeddings: Optional pre-built embedding model. A new instance is
            created when ``None``.
        persist_directory: Directory for ChromaDB persistence. Defaults to
            ``CHROMA_DIR``.

    Returns:
        Chroma: A freshly built and persisted Chroma vector store.
    """
    if embeddings is None:
        embeddings = _build_embeddings()  # Create embeddings client if not supplied

    target_dir = persist_directory or CHROMA_DIR  # Allow indexing into a custom directory
    chunks = _load_chunks(pdf_path)  # Load and split the source PDF

    print("Embedding and persisting vector store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=target_dir,
    )  # Embed chunks and write them to the ChromaDB directory
    print("  → Done")
    return vector_db


def reindex_vector_store(vector_db: Chroma, pdf_path: str) -> Chroma:
    """Replace all documents in an existing store with a new PDF's chunks.

    Reuses the live Chroma client and resets its collection in place instead of
    deleting the persist directory. This avoids the ChromaDB ``SQLITE_READONLY_DBMOVED``
    error that occurs when a cached persistent client's backing file is removed
    on disk.

    Args:
        vector_db: The open Chroma store to re-index. When ``None``, a fresh
            store is built from ``pdf_path``.
        pdf_path: Filesystem path to the new PDF to index.

    Returns:
        Chroma: The same vector store, now containing only the new PDF's chunks.
    """
    if vector_db is None:
        return build_vector_store(pdf_path)  # No live store yet — build from scratch

    chunks = _load_chunks(pdf_path)  # Load and split the new PDF first

    print("Resetting collection and embedding new documents...")
    vector_db.reset_collection()  # Drop and recreate the collection on the same client
    vector_db.add_documents(chunks)  # Embed and persist the new chunks
    print("  → Done")
    return vector_db
