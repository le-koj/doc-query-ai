import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "gemini-embedding-001"


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def load_vector_store(pdf_path: str) -> Chroma:
    """Load an existing Chroma store if present, otherwise build one from the PDF."""
    embeddings = _build_embeddings()

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    return build_vector_store(pdf_path, embeddings)


def build_vector_store(pdf_path: str, embeddings: GoogleGenerativeAIEmbeddings | None = None) -> Chroma:
    """Ingest a PDF: load, chunk, embed, and persist to Chroma."""
    if embeddings is None:
        embeddings = _build_embeddings()

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"  → {len(chunks)} chunks")

    print("Embedding and persisting vector store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("  → Done")
    return vector_db
