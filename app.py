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

from rag.chain import (  # Retrieval + LLM chain and its tuned settings
    MMR_LAMBDA,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    _format_doc_label,
    build_rag_chain,
    build_retriever,
    retrieval_status_line,
)
from rag.ingest import (  # Library ingestion helpers
    CHROMA_DIR,
    EMBEDDING_MODEL,
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
_rag_chain = None  # Unscoped LCEL chain (searches the whole library)
_retriever = None  # Unscoped retriever used to surface source documents
_scoped_cache: dict = {}  # Cache of (chain, retriever) keyed by selected doc_id set


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
    global _rag_chain, _retriever, _scoped_cache
    _scoped_cache = {}  # Invalidate scoped pipelines whenever the library changes
    if _vector_db is None:
        _rag_chain = None
        _retriever = None
        return
    _rag_chain = build_rag_chain(_vector_db)  # Wire retriever → prompt → LLM → parser
    _retriever = build_retriever(_vector_db)  # Separate retriever for source attribution


def _scoped_pipeline(doc_ids):
    """Return a (chain, retriever) pair scoped to the selected documents.

    An empty or missing selection uses the unscoped, library-wide pipeline.
    Scoped pipelines are cached per selection so repeated questions don't
    rebuild the LLM each time.

    Args:
        doc_ids: List of selected ``doc_id`` values, or a falsy value for all.

    Returns:
        tuple: A ``(chain, retriever)`` pair to answer the question with.
    """
    if not doc_ids:
        return _rag_chain, _retriever
    key = tuple(sorted(doc_ids))
    if key not in _scoped_cache:
        _scoped_cache[key] = (
            build_rag_chain(_vector_db, list(key)),
            build_retriever(_vector_db, list(key)),
        )
    return _scoped_cache[key]


def answer(question: str, history: list, scope=None) -> str:
    """Handle a chat message from the Gradio interface.

    Args:
        question: The user's natural-language question.
        history: Conversation history supplied by Gradio (unused; chain is stateless).
        scope: Optional list of ``doc_id`` values to restrict the search to.
            An empty or missing selection searches the whole library.

    Returns:
        str: A plain-text answer grounded in the library, with a list of the
            source documents used.
    """
    if not question.strip():
        return "Please enter a question."  # Reject empty submissions early

    if _rag_chain is None:
        return "Upload one or more PDFs and click 'Index documents' before asking questions."

    chain, retriever = _scoped_pipeline(scope)  # Restrict to selected docs when provided

    try:
        result = chain.invoke(question)  # Retrieve context and generate an answer
        sources = _sources_for(question, retriever)  # Identify which documents were used
    except EmbeddingQuotaError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surface API errors instead of crashing the UI
        return f"Sorry, something went wrong answering that: {exc}"

    if sources:
        result += "\n\n_Sources: " + ", ".join(sources) + "_"
    return result


def _sources_for(question: str, retriever=None) -> list[str]:
    """Return the distinct source labels retrieved for a question.

    Args:
        question: The user's question.
        retriever: The retriever to query. Defaults to the unscoped retriever.

    Returns:
        list[str]: Unique ``[source, p. N]`` labels, preserving retrieval order.
    """
    retriever = retriever or _retriever
    if retriever is None:
        return []
    seen: list[str] = []
    for doc in retriever.invoke(question):
        label = _format_doc_label(doc).strip("[]")
        if label and label not in seen:
            seen.append(label)
    return seen


def _resolve_upload_path(file) -> str | None:
    """Extract the on-disk PDF path from a Gradio file upload."""
    if file is None:
        return None
    if isinstance(file, str):
        return file
    if isinstance(file, dict):
        return file.get("path") or file.get("name")
    return getattr(file, "path", None) or getattr(file, "name", None)


def ingest_pdfs(files) -> tuple:
    """Index one or more uploaded PDFs into the library and rebuild the chain."""
    global _vector_db

    if not files:
        return _ingest_result("No files uploaded.")

    if not isinstance(files, list):
        files = [files]

    indexed: list[str] = []
    errors: list[str] = []
    for file in files:
        pdf_path = _resolve_upload_path(file)
        if not pdf_path:
            continue
        name = os.path.basename(pdf_path)
        try:
            summary = add_pdf(_vector_db, pdf_path)
            _vector_db = summary["vector_db"]
            verb = "Re-indexed" if summary["replaced"] else "Indexed"
            indexed.append(f"{verb} {summary['source']} ({summary['chunks']} chunks)")
        except PdfIngestError as exc:
            errors.append(f"Skipped {name}: {exc}")
        except EmbeddingQuotaError as exc:
            errors.append(f"Stopped at {name}: {exc}")
            break

    _rebuild_chain()
    status_lines = indexed + errors
    status = "\n".join(status_lines) if status_lines else "No valid PDFs were indexed."
    return _ingest_result(status)


def remove_document(doc_id: str) -> tuple:
    """Remove a single document from the library and rebuild the chain."""
    global _vector_db

    if not doc_id:
        return _ingest_result("Select a document to remove.")

    if _vector_db is None:
        return _ingest_result("No documents are indexed.")

    source = next(
        (doc["source"] for doc in list_documents(_vector_db) if doc["doc_id"] == doc_id),
        doc_id,
    )
    remove_pdf(_vector_db, doc_id)
    _rebuild_chain()
    return _ingest_result(f"Removed {source}.")


def _ingest_result(status: str) -> tuple:
    """Build the full UI refresh tuple after an ingest or remove action."""
    remove_update, scope_update = _dropdown_updates()
    return status, _library_rows(), remove_update, scope_update, _status_summary(), []


def _library_rows() -> list[list]:
    """Return the indexed-document library as table rows for the UI."""
    return [[doc["source"], doc["chunks"]] for doc in list_documents(_vector_db)]


def _document_choices() -> list[tuple[str, str]]:
    """Return ``(label, doc_id)`` choices for document selectors."""
    return [
        (f"{doc['source']} ({doc['chunks']} chunks)", doc["doc_id"])
        for doc in list_documents(_vector_db)
    ]


def _dropdown_updates():
    """Build refreshed updates for the remove and scope selectors."""
    choices = _document_choices()
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
    )


def _store_is_persisted() -> bool:
    """Return True when a non-empty ChromaDB store exists on disk."""
    return os.path.isdir(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))


def _status_summary() -> str:
    """Build a markdown panel describing the current store and retrieval setup."""
    docs = list_documents(_vector_db)
    total_chunks = sum(doc["chunks"] for doc in docs)

    if _vector_db is None:
        store_line = "- **Vector store:** not loaded — index a PDF to begin"
    elif _store_is_persisted():
        store_line = f"- **Vector store:** loaded (persisted at `{CHROMA_DIR}`)"
    else:
        store_line = "- **Vector store:** loaded (in memory)"

    scope_line = (
        "- **Scoping:** available — pick documents under *Search in*"
        if docs
        else "- **Scoping:** searches all documents once indexed"
    )

    return "\n".join(
        [
            "### System status",
            store_line,
            f"- **Indexed documents:** {len(docs)} ({total_chunks} chunks)",
            f"- **Embeddings:** `{EMBEDDING_MODEL}`",
            f"- **Retrieval:** {retrieval_status_line()}",
            scope_line,
        ]
    )


def _sync_scope(selected) -> list:
    """Copy the scope dropdown value into the chat's scope state."""
    return selected or []


def build_demo() -> gr.Blocks:
    """Build the Gradio UI after ``init_pipeline`` has loaded the vector store.

    The UI is constructed here (not at import time) so persisted documents and
    status appear immediately without a ``demo.load`` event that can deadlock
    with ``ChatInterface``.
    """
    with gr.Blocks(title="Doc Query AI") as demo:
        scope_state = gr.State([])  # Passed to the chat; kept separate from scope_select

        gr.Markdown(
            """
            # Doc Query AI
            Ask natural-language questions across a library of PDF documents, powered
            by **ChromaDB** vector search and **Google Gemini**. Answers cite the
            source documents they came from.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                status_panel = gr.Markdown(_status_summary())
                gr.Markdown("### Document library")
                upload = gr.File(label="PDF files", file_types=[".pdf"], file_count="multiple")
                upload_btn = gr.Button("Index documents")
                upload_status = gr.Textbox(label="Status", interactive=False, lines=4)
                library = gr.Dataframe(
                    headers=["Document", "Chunks"],
                    datatype=["str", "number"],
                    label="Indexed documents",
                    interactive=False,
                    value=_library_rows(),
                )
                remove_select = gr.Dropdown(
                    label="Remove a document",
                    choices=_document_choices(),
                    value=None,
                )
                remove_btn = gr.Button("Remove document")
                scope_select = gr.Dropdown(
                    label="Search in (leave empty to search all documents)",
                    choices=_document_choices(),
                    value=[],
                    multiselect=True,
                )

            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=answer,
                    additional_inputs=[scope_state],
                    cache_examples=False,
                    run_examples_on_click=False,
                    fill_height=False,
                    examples=[
                        ["What is the vacation policy?", []],
                        ["What are the code of conduct rules?", []],
                        ["How does the performance review process work?", []],
                    ],
                )

        scope_select.change(
            fn=_sync_scope,
            inputs=scope_select,
            outputs=scope_state,
            queue=False,
        )

        library_outputs = [upload_status, library, remove_select, scope_select, status_panel, scope_state]
        upload_btn.click(fn=ingest_pdfs, inputs=upload, outputs=library_outputs, queue=True)
        remove_btn.click(fn=remove_document, inputs=remove_select, outputs=library_outputs, queue=True)

    return demo


if __name__ == "__main__":
    init_pipeline()
    demo = build_demo()
    demo.queue(default_concurrency_limit=4)
    demo.launch()
