"""Unit tests for rag.ingest."""

from unittest.mock import MagicMock, patch

import rag.ingest as ingest


class TestLoadVectorStore:
    """Tests for load_vector_store branching logic."""

    @patch.object(ingest, "build_vector_store")
    @patch.object(ingest, "Chroma")
    @patch.object(ingest.os.path, "exists", return_value=True)
    @patch.object(ingest.os, "listdir", return_value=["chroma.sqlite3"])
    def test_loads_existing_store_when_directory_populated(
        self, mock_listdir, mock_exists, mock_chroma, mock_build, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ingest, "CHROMA_DIR", str(tmp_path))
        mock_chroma.return_value = MagicMock()

        result = ingest.load_vector_store("dummy.pdf")

        mock_build.assert_not_called()
        mock_chroma.assert_called_once()
        assert result is mock_chroma.return_value

    @patch.object(ingest, "build_vector_store")
    @patch.object(ingest.os.path, "exists", return_value=False)
    def test_builds_store_when_directory_missing(
        self, mock_exists, mock_build, monkeypatch
    ):
        monkeypatch.setattr(ingest, "CHROMA_DIR", "./missing_chroma")
        mock_build.return_value = MagicMock()

        result = ingest.load_vector_store("dummy.pdf")

        mock_build.assert_called_once()
        assert result is mock_build.return_value


class TestBuildVectorStore:
    """Tests for build_vector_store document processing."""

    @patch.object(ingest, "Chroma")
    @patch.object(ingest, "PyPDFLoader")
    def test_loads_splits_and_persists_chunks(
        self, mock_loader_cls, mock_chroma, sample_documents, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(ingest, "CHROMA_DIR", str(tmp_path))
        mock_loader_cls.return_value.load.return_value = sample_documents
        mock_chroma.from_documents.return_value = MagicMock()
        mock_embeddings = MagicMock()

        ingest.build_vector_store("test.pdf", embeddings=mock_embeddings)

        mock_loader_cls.assert_called_once_with("test.pdf")
        mock_chroma.from_documents.assert_called_once()
        call_kwargs = mock_chroma.from_documents.call_args.kwargs
        assert call_kwargs["embedding"] is mock_embeddings
        assert call_kwargs["persist_directory"] == str(tmp_path)
        assert len(call_kwargs["documents"]) >= 1
