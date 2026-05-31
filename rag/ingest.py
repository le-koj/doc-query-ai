"""Document ingestion utilities for the RAG pipeline.

Loads PDF files, splits them into text chunks, tags each chunk with source
metadata, embeds them with Google Gemini, and persists the vectors to a local
ChromaDB store. The store holds a *library* of documents: PDFs are added
incrementally without wiping previously indexed content.
"""

import hashlib  # Derive a stable document id from file contents
import os  # Check whether the ChromaDB directory already exists
import re  # Parse the retry delay out of rate-limit error messages
import time  # Back off between embedding retries

from langchain_community.document_loaders import PyPDFLoader  # Extract text from PDF files
from langchain_chroma import Chroma  # Local vector database for document embeddings
from langchain_core.documents import Document  # Typed document chunks
from langchain_core.embeddings import Embeddings  # Base interface for embedding wrappers
import torch
from langchain_huggingface import HuggingFaceEmbeddings  # Local Hugging Face embedding model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks

CHROMA_DIR = "./chroma_db"  # Directory where ChromaDB persists its SQLite database
CHUNK_SIZE = 500  # Maximum number of characters per text chunk
CHUNK_OVERLAP = 50  # Number of overlapping characters between adjacent chunks
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # Local Hugging Face embedding model
EMBED_BATCH_SIZE = 100  # Texts per embedding request (the Gemini API max)
EMBED_MAX_RETRIES = 5  # Attempts before giving up on a rate-limited batch
EMBED_DEFAULT_RETRY_SECONDS = 30.0  # Fallback wait when the API gives no retry hint
EMBED_MAX_RETRY_SECONDS = 75.0  # Cap a single back-off so the UI never hangs forever


class PdfIngestError(ValueError):
    """Raised when a PDF cannot be indexed because it has no usable text."""


class EmbeddingQuotaError(RuntimeError):
    """Raised when embeddings can't be generated because the API quota is exhausted."""


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True when an exception looks like an API rate-limit / quota error."""
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def _parse_retry_seconds(message: str) -> float:
    """Extract a retry delay (seconds) from a rate-limit error message.

    Args:
        message: The error text returned by the Gemini API.

    Returns:
        float: The suggested wait time, falling back to a default and capped to
            ``EMBED_MAX_RETRY_SECONDS``.
    """
    match = re.search(r"retry in ([\d.]+)s", message) or re.search(
        r"'retryDelay':\s*'(\d+)s'", message
    )
    seconds = float(match.group(1)) if match else EMBED_DEFAULT_RETRY_SECONDS
    return min(seconds + 1.0, EMBED_MAX_RETRY_SECONDS)  # Pad slightly, then cap


class ResilientEmbeddings(Embeddings):
    """Embeddings wrapper that retries rate-limited batches with back-off.

    Free-tier Gemini limits embedding requests per minute. This wrapper splits
    work into batches and, on a ``RESOURCE_EXHAUSTED`` (429) response, waits for
    the API-suggested delay and retries instead of crashing the request.
    """

    def __init__(
        self,
        base: Embeddings,
        batch_size: int = EMBED_BATCH_SIZE,
        max_retries: int = EMBED_MAX_RETRIES,
        sleep=time.sleep,
    ) -> None:
        self._base = base
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleep = sleep  # Injectable for tests

    def _call_with_retry(self, func, *args):
        """Invoke an embedding call, retrying on rate-limit errors."""
        for attempt in range(1, self._max_retries + 1):
            try:
                return func(*args)
            except Exception as error:  # noqa: BLE001 - inspect and re-raise below
                if not _is_rate_limit_error(error):
                    raise
                if attempt == self._max_retries:
                    raise EmbeddingQuotaError(
                        "Embedding quota exhausted. The Gemini free tier limits "
                        "embedding requests per minute — wait a minute and try "
                        "again, use fewer/smaller PDFs, or enable billing."
                    ) from error
                delay = _parse_retry_seconds(str(error))
                print(
                    f"Rate limited by embedding API; retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{self._max_retries - 1})..."
                )
                self._sleep(delay)
        raise EmbeddingQuotaError("Embedding retries exhausted.")  # Defensive fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batches, retrying rate-limited batches."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            embeddings.extend(self._call_with_retry(self._base.embed_documents, batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, retrying on rate-limit errors."""
        return self._call_with_retry(self._base.embed_query, text)


def _build_embeddings() -> Embeddings:
    """Create a rate-limit-resilient Hugging Face embeddings client.

    Returns:
        Embeddings: A wrapper around the configured Hugging Face embedding model that
            retries rate-limited batches with back-off.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return ResilientEmbeddings(base)


def compute_doc_id(pdf_path: str) -> str:
    """Derive a stable document id from a PDF's contents.

    Hashing the file bytes means re-uploading the same document yields the same
    id, which makes re-indexing idempotent regardless of the upload filename.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        str: A short hex digest uniquely identifying the file contents.
    """
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def _load_chunks(pdf_path: str) -> list[Document]:
    """Load a PDF and split it into overlapping, source-tagged text chunks.

    Each chunk carries ``doc_id``, ``source`` (filename), ``page``, and
    ``chunk_index`` metadata so retrieval results can be attributed to their
    originating document.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        list[Document]: The PDF content split into chunk-sized documents.

    Raises:
        PdfIngestError: If the PDF has no pages or no extractable text.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)  # Point the loader at the target PDF
    documents = loader.load()  # Parse the PDF into page-level Document objects
    print(f"  -> {len(documents)} pages")

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
    print(f"  -> {len(chunks)} chunks")

    if not chunks:
        raise PdfIngestError(
            "The PDF produced no text chunks after splitting. "
            "Try a document with more readable text content."
        )

    doc_id = compute_doc_id(pdf_path)  # Stable id derived from file contents
    source = os.path.basename(pdf_path)  # Human-readable filename for attribution
    for index, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id  # Group chunks by their source document
        chunk.metadata["source"] = source  # Display name shown in answers and the UI
        chunk.metadata["chunk_index"] = index  # Preserve original ordering within the doc

    return chunks


def load_vector_store(pdf_path: str | None = None) -> Chroma | None:
    """Load an existing Chroma store or build one from an optional seed PDF.

    If a persisted ChromaDB directory already exists and is non-empty, the
    existing store is loaded to avoid re-embedding on every startup. Otherwise,
    the PDF at ``pdf_path`` is ingested from scratch when that file exists.

    Args:
        pdf_path: Optional path to a seed PDF to index when no store exists.

    Returns:
        Chroma | None: A Chroma vector store ready for retrieval queries, or
            ``None`` when no store exists and no seed PDF is available.
    """
    embeddings = _build_embeddings()  # Initialise the Gemini embedding model

    # Reuse the persisted store when available to skip re-embedding
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    if not pdf_path or not os.path.isfile(pdf_path):
        print("No vector store found and no seed PDF available.")
        return None

    return build_vector_store(pdf_path, embeddings)  # First run — ingest the seed PDF


def build_vector_store(
    pdf_path: str,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """Create a new Chroma store seeded with a single PDF's chunks.

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
    print("  -> Done")
    return vector_db


def add_pdf(vector_db: Chroma | None, pdf_path: str) -> dict:
    """Add (or refresh) a PDF in the document library without wiping others.

    When the store does not yet exist, it is created. If the same document
    (by content hash) was already indexed, its old chunks are removed first so
    the operation is idempotent and safe to re-run.

    Args:
        vector_db: The open Chroma store to add to. When ``None``, a fresh
            store is built from ``pdf_path``.
        pdf_path: Filesystem path to the PDF to index.

    Returns:
        dict: Summary with ``doc_id``, ``source``, ``chunks``, and a
            ``replaced`` flag indicating whether prior chunks were removed.

    Raises:
        PdfIngestError: If the PDF has no extractable text.
    """
    if vector_db is None:
        # Validate before building so a bad PDF never creates an empty store.
        chunks = _load_chunks(pdf_path)
        vector_db = build_vector_store(pdf_path)
        return {
            "vector_db": vector_db,
            "doc_id": chunks[0].metadata["doc_id"],
            "source": chunks[0].metadata["source"],
            "chunks": len(chunks),
            "replaced": False,
        }

    chunks = _load_chunks(pdf_path)  # Load and split first so failures don't mutate the store
    doc_id = chunks[0].metadata["doc_id"]

    existing = vector_db.get(where={"doc_id": doc_id})  # Detect a previously indexed copy
    replaced = bool(existing and existing.get("ids"))
    if replaced:
        print(f"Replacing existing chunks for doc_id={doc_id}")
        vector_db.delete(where={"doc_id": doc_id})  # Drop only this document's chunks

    print(f"Adding {len(chunks)} chunks to the library...")
    vector_db.add_documents(chunks)  # Append the new chunks alongside existing docs
    print("  -> Done")
    return {
        "vector_db": vector_db,
        "doc_id": doc_id,
        "source": chunks[0].metadata["source"],
        "chunks": len(chunks),
        "replaced": replaced,
    }


def remove_pdf(vector_db: Chroma, doc_id: str) -> None:
    """Remove all chunks belonging to a single document from the library.

    Args:
        vector_db: The open Chroma store to delete from.
        doc_id: The ``doc_id`` metadata value identifying the document.
    """
    print(f"Removing doc_id={doc_id} from the library...")
    vector_db.delete(where={"doc_id": doc_id})  # Delete only the matching document's chunks


def list_documents(vector_db: Chroma | None) -> list[dict]:
    """Summarise the documents currently indexed in the library.

    Args:
        vector_db: The open Chroma store, or ``None`` when nothing is indexed.

    Returns:
        list[dict]: One entry per document with ``doc_id``, ``source``, and
            ``chunks`` (count), sorted by source filename.
    """
    if vector_db is None:
        return []

    records = vector_db.get(include=["metadatas"])  # Pull all chunk metadata
    summary: dict[str, dict] = {}
    for metadata in records.get("metadatas", []) or []:
        doc_id = metadata.get("doc_id", "unknown")
        entry = summary.setdefault(
            doc_id,
            {"doc_id": doc_id, "source": metadata.get("source", "unknown"), "chunks": 0},
        )
        entry["chunks"] += 1

    return sorted(summary.values(), key=lambda item: item["source"].lower())
