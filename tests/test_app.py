"""Unit tests for app.py Gradio handlers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app


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
    @patch.object(app, "build_vector_store")
    @patch.object(app.shutil, "rmtree")
    @patch.object(app.os.path, "exists", return_value=True)
    def test_reindexes_uploaded_pdf(
        self, mock_exists, mock_rmtree, mock_build_store, mock_build_chain, monkeypatch
    ):
        monkeypatch.setattr(app, "CHROMA_DIR", "./chroma_db")
        mock_db = MagicMock()
        mock_chain = MagicMock()
        mock_build_store.return_value = mock_db
        mock_build_chain.return_value = mock_chain
        uploaded = SimpleNamespace(name="/tmp/uploaded.pdf")

        result = app.ingest_pdf(uploaded)

        mock_rmtree.assert_called_once_with("./chroma_db")
        mock_build_store.assert_called_once_with("/tmp/uploaded.pdf")
        mock_build_chain.assert_called_once_with(mock_db)
        assert app._vector_db is mock_db
        assert app._rag_chain is mock_chain
        assert result == "Indexed: uploaded.pdf"
