"""Unit tests for app.py Gradio handlers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

        assert result == "RAG pipeline is not initialised."

    def test_delegates_to_rag_chain(self, mock_rag_chain, monkeypatch):
        monkeypatch.setattr(app, "_rag_chain", mock_rag_chain)

        result = app.answer("What is PTO?", history=[])

        mock_rag_chain.invoke.assert_called_once_with("What is PTO?")
        assert result == "Mocked answer."


class TestIngestPdf:
    """Tests for the ingest_pdf() upload handler."""

    def test_no_file_returns_message(self):
        assert app.ingest_pdf(None) == "No file uploaded."

    @patch.object(app, "build_rag_chain")
    @patch.object(app, "reindex_vector_store")
    def test_reindexes_uploaded_pdf(
        self,
        mock_reindex,
        mock_build_chain,
        monkeypatch,
    ):
        existing_db = MagicMock()
        monkeypatch.setattr(app, "_vector_db", existing_db)
        new_db = MagicMock()
        mock_reindex.return_value = new_db
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain
        uploaded = SimpleNamespace(path="/tmp/uploaded.pdf", orig_name="invoice.pdf")

        result = app.ingest_pdf(uploaded)

        mock_reindex.assert_called_once_with(existing_db, "/tmp/uploaded.pdf")
        mock_build_chain.assert_called_once_with(new_db)
        assert app._vector_db is new_db
        assert app._rag_chain is mock_chain
        assert result == "Indexed: invoice.pdf"
