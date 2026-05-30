"""Gradio web UI for the Doc Query AI chatbot.

Provides a browser-based interface for asking questions about a library of
indexed PDF documents and uploading new PDFs to grow the library. Multiple PDFs
can be indexed together and answers cite the source document(s) used.

Usage:
    python app.py
"""

import os  # Detect a seed PDF on startup

import gradio as gr  # Web UI framework for the chat interface
from dotenv import load_dotenv  # Load API keys from a .env file

from rag.chain import build_rag_chain, build_retriever, _format_doc_label  # Retrieval + LLM chain
from rag.ingest import (  # Library ingestion helpers
    EmbeddingQuotaError,
    PdfIngestError,
    add_pdf,
    list_documents,
    load_vector_store,
    remove_pdf,
)

load_dotenv()  # Read GOOGLE_API_KEY and other secrets from .env

SEED_PDF_PATH = "documents/TechCorp_Official_Employee_Handbook.pdf"  # Optional seed document

_vector_db = None  # Chroma vector store; initialised by init_pipeline()
_rag_chain = None  # LCEL chain; initialised by init_pipeline()
_retriever = None  # Retriever used to surface source documents alongside answers


def init_pipeline(seed_pdf_path: str = SEED_PDF_PATH) -> None:
    """Load the vector store and build the RAG chain.

    Reuses a persisted library when present. Otherwise seeds the library from
    ``seed_pdf_path`` when that file exists, or stays uninitialised until a
    document is uploaded via the UI.

    Args:
        seed_pdf_path: Optional path to a PDF to index when no store exists.
    """
    global _vector_db, _rag_chain, _retriever
    print("Initialising RAG pipeline...")
    _vector_db = load_vector_store(seed_pdf_path)  # Load existing library or seed it
    _rebuild_chain()
    if _rag_chain is None:
        print("Ready. Upload one or more PDFs and click 'Index documents' to begin.\n")
    else:
        print("Ready.\n")


def _rebuild_chain() -> None:
    """Rebuild the chain and retriever from the current vector store."""
    global _rag_chain, _retriever
    if _vector_db is None:
        _rag_chain = None
        _retriever = None
        return
    _rag_chain = build_rag_chain(_vector_db)  # Wire retriever → prompt → LLM → parser
    _retriever = build_retriever(_vector_db)  # Separate retriever for source attribution


def answer(question: str, history: list) -> str:
    """Handle a chat message from the Gradio interface.

    Args:
        question: The user's natural-language question.
        history: Conversation history supplied by Gradio (unused; chain is stateless).

    Returns:
        str: A plain-text answer grounded in the library, with a list of the
            source documents used.
    """
    if not question.strip():
        return "Please enter a question."  # Reject empty submissions early

    if _rag_chain is None:
        return "Upload one or more PDFs and click 'Index documents' before asking questions."

    try:
        result = _rag_chain.invoke(question)  # Retrieve context and generate an answer
        sources = _sources_for(question)  # Identify which documents the answer drew from
    except EmbeddingQuotaError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surface API errors instead of crashing the UI
        return f"Sorry, something went wrong answering that: {exc}"

    if sources:
        result += "\n\n_Sources: " + ", ".join(sources) + "_"
    return result


def _sources_for(question: str) -> list[str]:
    """Return the distinct source labels retrieved for a question.

    Args:
        question: The user's question.

    Returns:
        list[str]: Unique ``[source, p. N]`` labels, preserving retrieval order.
    """
    if _retriever is None:
        return []
    seen: list[str] = []
    for doc in _retriever.invoke(question):
        label = _format_doc_label(doc).strip("[]")
        if label and label not in seen:
            seen.append(label)
    return seen


def _resolve_upload_path(file) -> str | None:
    """Extract the on-disk PDF path from a Gradio file upload.

    Gradio 6 passes a ``FileData`` object with a ``path`` attribute. Older
    versions and test doubles may supply a plain string or an object with
    ``name``.

    Args:
        file: Gradio upload value, file-like object, path string, or dict.

    Returns:
        str | None: Absolute or server-side path to the uploaded PDF, or
            ``None`` when no path can be resolved.
    """
    if file is None:
        return None
    if isinstance(file, str):
        return file
    if isinstance(file, dict):
        return file.get("path") or file.get("name")
    return getattr(file, "path", None) or getattr(file, "name", None)


def ingest_pdfs(files) -> tuple[str, list[list]]:
    """Index one or more uploaded PDFs into the library and rebuild the chain.

    PDFs are added incrementally: existing documents are preserved, and
    re-uploading a document refreshes it in place.

    Args:
        files: A Gradio multi-file upload value (list), a single upload, or
            ``None``.

    Returns:
        tuple: A status message, the refreshed document library table rows, and
            an updated dropdown of indexed documents.
    """
    global _vector_db

    if not files:
        # Guard against clicking with no selection
        return "No files uploaded.", _library_rows(), _document_dropdown_update()

    if not isinstance(files, list):
        files = [files]  # Normalise a single upload into a list

    indexed: list[str] = []
    errors: list[str] = []
    for file in files:
        pdf_path = _resolve_upload_path(file)
        if not pdf_path:
            continue
        name = os.path.basename(pdf_path)
        try:
            summary = add_pdf(_vector_db, pdf_path)  # Append (or refresh) this document
            _vector_db = summary["vector_db"]
            verb = "Re-indexed" if summary["replaced"] else "Indexed"
            indexed.append(f"{verb} {summary['source']} ({summary['chunks']} chunks)")
        except PdfIngestError as exc:
            errors.append(f"Skipped {name}: {exc}")
        except EmbeddingQuotaError as exc:
            # Quota is shared across files, so stop early rather than retry each one.
            errors.append(f"Stopped at {name}: {exc}")
            break

    _rebuild_chain()  # Rebuild the chain so new documents are searchable

    status_lines = indexed + errors
    status = "\n".join(status_lines) if status_lines else "No valid PDFs were indexed."
    return status, _library_rows(), _document_dropdown_update()


def remove_document(doc_id: str) -> tuple:
    """Remove a single document from the library and rebuild the chain.

    Args:
        doc_id: The ``doc_id`` of the document to remove (from the selector).

    Returns:
        tuple: A status message, the refreshed library table rows, and an
            updated dropdown of remaining documents.
    """
    global _vector_db

    if not doc_id:
        return "Select a document to remove.", _library_rows(), _document_dropdown_update()

    if _vector_db is None:
        return "No documents are indexed.", _library_rows(), _document_dropdown_update()

    source = next(
        (doc["source"] for doc in list_documents(_vector_db) if doc["doc_id"] == doc_id),
        doc_id,
    )  # Resolve a friendly name for the status message before deleting
    remove_pdf(_vector_db, doc_id)  # Delete only this document's chunks
    _rebuild_chain()  # Rebuild so removed content is no longer retrievable

    return f"Removed {source}.", _library_rows(), _document_dropdown_update()


def _library_rows() -> list[list]:
    """Return the indexed-document library as table rows for the UI.

    Returns:
        list[list]: Rows of ``[source, chunks]`` for each indexed document.
    """
    return [[doc["source"], doc["chunks"]] for doc in list_documents(_vector_db)]


def _document_choices() -> list[tuple[str, str]]:
    """Return ``(label, doc_id)`` choices for the remove-document selector.

    Returns:
        list[tuple[str, str]]: One ``(display_label, doc_id)`` pair per document.
    """
    return [
        (f"{doc['source']} ({doc['chunks']} chunks)", doc["doc_id"])
        for doc in list_documents(_vector_db)
    ]


def _document_dropdown_update():
    """Build a Gradio update that refreshes the selector's choices."""
    return gr.update(choices=_document_choices(), value=None)


with gr.Blocks(title="Doc Query AI") as demo:
    gr.Markdown(
        """
        # Doc Query AI
        Ask natural-language questions across a library of PDF documents, powered
        by **ChromaDB** vector search and **Google Gemini**. Answers cite the
        source documents they came from.
        """
    )  # Page title and description shown at the top of the UI

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document library")  # Section heading for the library panel
            upload = gr.File(
                label="PDF files",
                file_types=[".pdf"],
                file_count="multiple",
            )  # File picker accepting one or more PDFs
            upload_btn = gr.Button("Index documents")  # Trigger indexing of the uploaded files
            upload_status = gr.Textbox(label="Status", interactive=False, lines=4)  # Status feedback
            library = gr.Dataframe(
                headers=["Document", "Chunks"],
                datatype=["str", "number"],
                label="Indexed documents",
                interactive=False,
                value=_library_rows(),
            )  # Live view of the indexed document library
            remove_select = gr.Dropdown(
                label="Remove a document",
                choices=_document_choices(),
                value=None,
            )  # Selector mapping a document label to its doc_id
            remove_btn = gr.Button("Remove document")  # Trigger removal of the selected document
            upload_btn.click(
                fn=ingest_pdfs,
                inputs=upload,
                outputs=[upload_status, library, remove_select],
            )  # Wire button to the multi-PDF ingestion handler
            remove_btn.click(
                fn=remove_document,
                inputs=remove_select,
                outputs=[upload_status, library, remove_select],
            )  # Wire button to the document removal handler

        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=answer,
                examples=[
                    "What is the vacation policy?",
                    "What are the code of conduct rules?",
                    "How does the performance review process work?",
                ],
            )  # Main chat panel with starter example questions

if __name__ == "__main__":
    init_pipeline()  # Build the vector store and chain before serving requests
    demo.launch()  # Start the local Gradio server (default: http://127.0.0.1:7860)
