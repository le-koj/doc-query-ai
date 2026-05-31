"""Unit tests for rag.ingest."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import rag.ingest as ingest


@pytest.fixture(autouse=True)
def mock_build_embeddings(monkeypatch):
    """Prevent loading local Hugging Face embedding model during unit tests."""
    monkeypatch.setattr(ingest, "_build_embeddings", MagicMock())



class TestParseRetrySeconds:
    """Tests for _parse_retry_seconds."""

    def test_parses_human_readable_retry(self):
        assert ingest._parse_retry_seconds("Please retry in 17.9s.") == pytest.approx(18.9)

    def test_parses_retry_delay_field(self):
        assert ingest._parse_retry_seconds("'retryDelay': '32s'") == pytest.approx(33.0)

    def test_caps_excessive_delay(self):
        assert ingest._parse_retry_seconds("retry in 999s") == ingest.EMBED_MAX_RETRY_SECONDS

    def test_falls_back_to_default_when_absent(self):
        assert ingest._parse_retry_seconds("no hint here") == pytest.approx(
            ingest.EMBED_DEFAULT_RETRY_SECONDS + 1.0
        )


class TestResilientEmbeddings:
    """Tests for the rate-limit-resilient embeddings wrapper."""

    def test_retries_rate_limited_batch_then_succeeds(self):
        base = MagicMock()
        base.embed_documents.side_effect = [
            RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 1s."),
            [[0.1], [0.2]],
        ]
        sleeps: list[float] = []
        wrapper = ingest.ResilientEmbeddings(base, sleep=sleeps.append)

        result = wrapper.embed_documents(["a", "b"])

        assert result == [[0.1], [0.2]]
        assert base.embed_documents.call_count == 2
        assert sleeps  # Slept at least once before retrying

    def test_raises_quota_error_after_max_retries(self):
        base = MagicMock()
        base.embed_documents.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        wrapper = ingest.ResilientEmbeddings(base, max_retries=3, sleep=lambda _s: None)

        with pytest.raises(ingest.EmbeddingQuotaError):
            wrapper.embed_documents(["a"])

        assert base.embed_documents.call_count == 3

    def test_non_rate_limit_error_propagates_immediately(self):
        base = MagicMock()
        base.embed_documents.side_effect = ValueError("bad input")
        wrapper = ingest.ResilientEmbeddings(base, sleep=lambda _s: None)

        with pytest.raises(ValueError, match="bad input"):
            wrapper.embed_documents(["a"])

        assert base.embed_documents.call_count == 1

    def test_batches_documents(self):
        base = MagicMock()
        base.embed_documents.side_effect = lambda batch: [[0.0]] * len(batch)
        wrapper = ingest.ResilientEmbeddings(base, batch_size=2, sleep=lambda _s: None)

        result = wrapper.embed_documents(["a", "b", "c"])

        assert len(result) == 3
        assert base.embed_documents.call_count == 2  # 2 + 1


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
    @patch.object(ingest.os.path, "isfile", return_value=True)
    @patch.object(ingest.os.path, "exists", return_value=False)
    def test_builds_store_when_directory_missing(
        self, mock_exists, mock_isfile, mock_build, monkeypatch
    ):
        monkeypatch.setattr(ingest, "CHROMA_DIR", "./missing_chroma")
        mock_build.return_value = MagicMock()

        result = ingest.load_vector_store("dummy.pdf")

        mock_build.assert_called_once()
        assert result is mock_build.return_value

    @patch.object(ingest, "build_vector_store")
    @patch.object(ingest.os.path, "isfile", return_value=False)
    @patch.object(ingest.os.path, "exists", return_value=False)
    def test_returns_none_when_store_and_pdf_missing(
        self, mock_exists, mock_isfile, mock_build, monkeypatch
    ):
        monkeypatch.setattr(ingest, "CHROMA_DIR", "./missing_chroma")

        result = ingest.load_vector_store("missing.pdf")

        mock_build.assert_not_called()
        assert result is None

    @patch.object(ingest, "build_vector_store")
    @patch.object(ingest.os.path, "exists", return_value=False)
    def test_returns_none_without_seed_pdf(self, mock_exists, mock_build, monkeypatch):
        monkeypatch.setattr(ingest, "CHROMA_DIR", "./missing_chroma")

        result = ingest.load_vector_store()

        mock_build.assert_not_called()
        assert result is None


class TestLoadChunks:
    """Tests for _load_chunks validation and metadata tagging."""

    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "PyPDFLoader")
    def test_tags_chunks_with_source_metadata(self, mock_loader_cls, mock_doc_id):
        mock_loader_cls.return_value.load.return_value = [
            Document(page_content="Some readable policy text.", metadata={"page": 0}),
        ]

        chunks = ingest._load_chunks("documents/handbook.pdf")

        assert chunks  # At least one chunk produced
        for index, chunk in enumerate(chunks):
            assert chunk.metadata["doc_id"] == "deadbeef"
            assert chunk.metadata["source"] == "handbook.pdf"
            assert chunk.metadata["chunk_index"] == index

    @patch.object(ingest, "PyPDFLoader")
    def test_raises_when_pdf_has_no_text(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = [Document(page_content="   ")]

        with pytest.raises(ingest.PdfIngestError, match="No extractable text"):
            ingest._load_chunks("blank.pdf")

    @patch.object(ingest, "PyPDFLoader")
    def test_raises_when_splitting_produces_no_chunks(self, mock_loader_cls):
        mock_loader_cls.return_value.load.return_value = [Document(page_content="Hi")]

        with patch.object(
            ingest.RecursiveCharacterTextSplitter,
            "split_documents",
            return_value=[],
        ):
            with pytest.raises(ingest.PdfIngestError, match="no text chunks"):
                ingest._load_chunks("tiny.pdf")


class TestBuildVectorStore:
    """Tests for build_vector_store document processing."""

    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "Chroma")
    @patch.object(ingest, "PyPDFLoader")
    def test_loads_splits_and_persists_chunks(
        self, mock_loader_cls, mock_chroma, mock_doc_id, sample_documents, monkeypatch, tmp_path
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


class TestAddPdf:
    """Tests for add_pdf incremental indexing."""

    @patch.object(ingest, "build_vector_store")
    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "PyPDFLoader")
    def test_builds_fresh_store_when_none(
        self, mock_loader_cls, mock_doc_id, mock_build, sample_documents
    ):
        mock_loader_cls.return_value.load.return_value = sample_documents
        mock_build.return_value = MagicMock()

        summary = ingest.add_pdf(None, "handbook.pdf")

        mock_build.assert_called_once_with("handbook.pdf")
        assert summary["doc_id"] == "deadbeef"
        assert summary["source"] == "handbook.pdf"
        assert summary["replaced"] is False
        assert summary["vector_db"] is mock_build.return_value

    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "PyPDFLoader")
    def test_appends_to_existing_store(
        self, mock_loader_cls, mock_doc_id, sample_documents
    ):
        mock_loader_cls.return_value.load.return_value = sample_documents
        mock_db = MagicMock()
        mock_db.get.return_value = {"ids": []}  # No prior copy of this document

        summary = ingest.add_pdf(mock_db, "handbook.pdf")

        mock_db.delete.assert_not_called()
        mock_db.add_documents.assert_called_once()
        assert summary["replaced"] is False
        assert summary["vector_db"] is mock_db

    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "PyPDFLoader")
    def test_replaces_existing_document(
        self, mock_loader_cls, mock_doc_id, sample_documents
    ):
        mock_loader_cls.return_value.load.return_value = sample_documents
        mock_db = MagicMock()
        mock_db.get.return_value = {"ids": ["old-1", "old-2"]}  # Prior copy present

        summary = ingest.add_pdf(mock_db, "handbook.pdf")

        mock_db.delete.assert_called_once_with(where={"doc_id": "deadbeef"})
        mock_db.add_documents.assert_called_once()
        assert summary["replaced"] is True

    @patch.object(ingest, "compute_doc_id", return_value="deadbeef")
    @patch.object(ingest, "PyPDFLoader")
    def test_does_not_mutate_store_when_pdf_is_empty(self, mock_loader_cls, mock_doc_id):
        mock_loader_cls.return_value.load.return_value = [Document(page_content="")]
        mock_db = MagicMock()

        with pytest.raises(ingest.PdfIngestError):
            ingest.add_pdf(mock_db, "blank.pdf")

        mock_db.delete.assert_not_called()
        mock_db.add_documents.assert_not_called()


class TestRemovePdf:
    """Tests for remove_pdf."""

    def test_deletes_only_matching_doc_id(self):
        mock_db = MagicMock()

        ingest.remove_pdf(mock_db, "deadbeef")

        mock_db.delete.assert_called_once_with(where={"doc_id": "deadbeef"})


class TestListDocuments:
    """Tests for list_documents aggregation."""

    def test_returns_empty_list_when_store_is_none(self):
        assert ingest.list_documents(None) == []

    def test_aggregates_chunks_per_document(self):
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "metadatas": [
                {"doc_id": "a", "source": "b.pdf"},
                {"doc_id": "a", "source": "b.pdf"},
                {"doc_id": "c", "source": "a.pdf"},
            ]
        }

        result = ingest.list_documents(mock_db)

        assert result == [
            {"doc_id": "c", "source": "a.pdf", "chunks": 1},
            {"doc_id": "a", "source": "b.pdf", "chunks": 2},
        ]
