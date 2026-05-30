"""Unit tests for app.py Gradio handlers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import app


class TestResolveUploadPath:
    """Tests for _resolve_upload_path."""

    def test_none_returns_none(self):
        assert app._resolve_upload_path(None) is None

    def test_string_path(self):
        assert app._resolve_upload_path("/tmp/doc.pdf") == "/tmp/doc.pdf"

    def test_filedata_uses_path(self):
        upload = SimpleNamespace(path="/tmp/gradio/doc.pdf", orig_name="invoice.pdf")
        assert app._resolve_upload_path(upload) == "/tmp/gradio/doc.pdf"

    def test_dict_uses_path(self):
        assert app._resolve_upload_path({"path": "/tmp/doc.pdf"}) == "/tmp/doc.pdf"


class TestAnswer:
    """Tests for the answer() chat handler."""

    def test_empty_question_returns_prompt(self):
        result = app.answer("   ", history=[])

        assert result == "Please enter a question."

    def test_uninitialised_chain_returns_message(self, monkeypatch):
        monkeypatch.setattr(app, "_rag_chain", None)

        result = app.answer("What is PTO?", history=[])

        assert result == (
            "Upload one or more PDFs and click 'Index documents' before asking questions."
        )

    def test_delegates_to_rag_chain_without_sources(self, mock_rag_chain, monkeypatch):
        monkeypatch.setattr(app, "_rag_chain", mock_rag_chain)
        monkeypatch.setattr(app, "_retriever", None)

        result = app.answer("What is PTO?", history=[])

        mock_rag_chain.invoke.assert_called_once_with("What is PTO?")
        assert result == "Mocked answer."

    def test_quota_error_returns_friendly_message(self, monkeypatch):
        chain = MagicMock()
        chain.invoke.side_effect = app.EmbeddingQuotaError("Embedding quota exhausted.")
        monkeypatch.setattr(app, "_rag_chain", chain)

        result = app.answer("What is PTO?", history=[])

        assert result == "Embedding quota exhausted."

    def test_appends_source_attribution(self, mock_rag_chain, monkeypatch):
        monkeypatch.setattr(app, "_rag_chain", mock_rag_chain)
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="x", metadata={"source": "handbook.pdf", "page": 0}),
            Document(page_content="y", metadata={"source": "handbook.pdf", "page": 0}),
            Document(page_content="z", metadata={"source": "benefits.pdf", "page": 4}),
        ]
        monkeypatch.setattr(app, "_retriever", retriever)

        result = app.answer("What is PTO?", history=[])

        assert "Mocked answer." in result
        assert "handbook.pdf, p. 1" in result
        assert "benefits.pdf, p. 5" in result


class TestIngestPdfs:
    """Tests for the ingest_pdfs() upload handler."""

    def test_no_files_returns_message(self, monkeypatch):
        monkeypatch.setattr(app, "list_documents", lambda db: [])

        status, rows, _dropdown = app.ingest_pdfs(None)

        assert status == "No files uploaded."
        assert rows == []

    @patch.object(app, "build_retriever")
    @patch.object(app, "build_rag_chain")
    @patch.object(app, "list_documents")
    @patch.object(app, "add_pdf")
    def test_indexes_multiple_pdfs(
        self, mock_add, mock_list, mock_build_chain, mock_build_retriever, monkeypatch
    ):
        monkeypatch.setattr(app, "_vector_db", None)
        new_db = MagicMock()
        mock_add.side_effect = [
            {"vector_db": new_db, "doc_id": "a", "source": "a.pdf", "chunks": 3, "replaced": False},
            {"vector_db": new_db, "doc_id": "b", "source": "b.pdf", "chunks": 5, "replaced": False},
        ]
        mock_list.return_value = [
            {"doc_id": "a", "source": "a.pdf", "chunks": 3},
            {"doc_id": "b", "source": "b.pdf", "chunks": 5},
        ]
        files = [
            SimpleNamespace(path="/tmp/a.pdf", orig_name="a.pdf"),
            SimpleNamespace(path="/tmp/b.pdf", orig_name="b.pdf"),
        ]

        status, rows, _dropdown = app.ingest_pdfs(files)

        assert mock_add.call_count == 2
        assert "Indexed a.pdf (3 chunks)" in status
        assert "Indexed b.pdf (5 chunks)" in status
        assert rows == [["a.pdf", 3], ["b.pdf", 5]]
        assert app._vector_db is new_db

    @patch.object(app, "build_retriever")
    @patch.object(app, "build_rag_chain")
    @patch.object(app, "list_documents", return_value=[])
    @patch.object(app, "add_pdf")
    def test_reports_ingest_errors_per_file(
        self, mock_add, mock_list, mock_build_chain, mock_build_retriever, monkeypatch
    ):
        monkeypatch.setattr(app, "_vector_db", MagicMock())
        mock_add.side_effect = app.PdfIngestError("No extractable text found in this PDF.")
        files = [SimpleNamespace(path="/tmp/scan.pdf", orig_name="scan.pdf")]

        status, rows, _dropdown = app.ingest_pdfs(files)

        assert "Skipped scan.pdf" in status
        assert "No extractable text" in status

    @patch.object(app, "build_retriever")
    @patch.object(app, "build_rag_chain")
    @patch.object(app, "list_documents", return_value=[])
    @patch.object(app, "add_pdf")
    def test_stops_on_quota_error(
        self, mock_add, mock_list, mock_build_chain, mock_build_retriever, monkeypatch
    ):
        monkeypatch.setattr(app, "_vector_db", None)
        mock_add.side_effect = app.EmbeddingQuotaError("Embedding quota exhausted.")
        files = [
            SimpleNamespace(path="/tmp/a.pdf", orig_name="a.pdf"),
            SimpleNamespace(path="/tmp/b.pdf", orig_name="b.pdf"),
        ]

        status, rows, _dropdown = app.ingest_pdfs(files)

        assert mock_add.call_count == 1  # Stopped after the first file's quota error
        assert "Stopped at a.pdf" in status
        assert "quota exhausted" in status.lower()


class TestRemoveDocument:
    """Tests for the remove_document() handler."""

    def test_no_selection_returns_prompt(self, monkeypatch):
        monkeypatch.setattr(app, "list_documents", lambda db: [])

        status, rows, _dropdown = app.remove_document(None)

        assert status == "Select a document to remove."

    def test_no_store_returns_message(self, monkeypatch):
        monkeypatch.setattr(app, "_vector_db", None)
        monkeypatch.setattr(app, "list_documents", lambda db: [])

        status, rows, _dropdown = app.remove_document("deadbeef")

        assert status == "No documents are indexed."

    @patch.object(app, "build_retriever")
    @patch.object(app, "build_rag_chain")
    @patch.object(app, "remove_pdf")
    def test_removes_selected_document(
        self, mock_remove, mock_build_chain, mock_build_retriever, monkeypatch
    ):
        db = MagicMock()
        monkeypatch.setattr(app, "_vector_db", db)
        monkeypatch.setattr(
            app,
            "list_documents",
            lambda _db: [{"doc_id": "deadbeef", "source": "handbook.pdf", "chunks": 4}],
        )

        status, rows, _dropdown = app.remove_document("deadbeef")

        mock_remove.assert_called_once_with(db, "deadbeef")
        assert status == "Removed handbook.pdf."
