"""Unit tests for rag.chain."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import rag.chain as chain


@pytest.fixture(autouse=True)
def disable_rerank_by_default(monkeypatch):
    """Keep retriever unit tests on the plain MMR path unless a test opts in."""
    monkeypatch.setattr(chain, "RERANK_ENABLED", False)


class TestFormatDocLabel:
    """Tests for the _format_doc_label helper."""

    def test_includes_source_and_one_indexed_page(self):
        doc = Document(page_content="x", metadata={"source": "a.pdf", "page": 2})
        assert chain._format_doc_label(doc) == "[a.pdf, p. 3]"

    def test_source_only_when_page_missing(self):
        doc = Document(page_content="x", metadata={"source": "a.pdf"})
        assert chain._format_doc_label(doc) == "[a.pdf]"

    def test_empty_when_no_source(self):
        assert chain._format_doc_label(Document(page_content="x")) == ""


class TestFormatDocs:
    """Tests for the _format_docs helper."""

    def test_labels_chunks_with_source_and_page(self, sample_documents):
        result = chain._format_docs(sample_documents)

        assert result == (
            "[handbook.pdf, p. 1]\nFirst chunk about vacation policy.\n\n"
            "[handbook.pdf, p. 2]\nSecond chunk about code of conduct."
        )

    def test_falls_back_to_plain_content_without_metadata(self):
        docs = [Document(page_content="No metadata here.")]
        assert chain._format_docs(docs) == "No metadata here."

    def test_empty_list_returns_empty_string(self):
        assert chain._format_docs([]) == ""


class TestRetrievalStatusLine:
    """Tests for retrieval_status_line."""

    def test_plain_mmr_status(self, monkeypatch):
        monkeypatch.setattr(chain, "RERANK_ENABLED", False)

        status = chain.retrieval_status_line()

        assert f"k={chain.RETRIEVER_K}" in status
        assert "rerank" not in status

    def test_rerank_status(self, monkeypatch):
        monkeypatch.setattr(chain, "RERANK_ENABLED", True)

        status = chain.retrieval_status_line()

        assert "rerank" in status
        assert chain.RERANK_MODEL in status


class TestBuildRetriever:
    """Tests for build_retriever search configuration."""

    def test_uses_mmr_with_diversity_settings(self):
        db = MagicMock()

        chain.build_retriever(db)

        db.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": chain.RETRIEVER_K,
                "fetch_k": chain.RETRIEVER_FETCH_K,
                "lambda_mult": chain.MMR_LAMBDA,
            },
        )

    def test_adds_doc_id_filter_when_scoped(self):
        db = MagicMock()

        chain.build_retriever(db, doc_ids=["a", "b"])

        _, kwargs = db.as_retriever.call_args
        assert kwargs["search_kwargs"]["filter"] == {"doc_id": {"$in": ["a", "b"]}}

    def test_no_filter_when_doc_ids_empty(self):
        db = MagicMock()

        chain.build_retriever(db, doc_ids=[])

        _, kwargs = db.as_retriever.call_args
        assert "filter" not in kwargs["search_kwargs"]

    def test_wraps_with_reranker_when_enabled(self, monkeypatch):
        monkeypatch.setattr(chain, "RERANK_ENABLED", True)
        db = MagicMock()
        base = MagicMock()
        db.as_retriever.return_value = base
        wrapped = MagicMock()

        with patch.object(chain, "_wrap_with_reranker", return_value=wrapped) as mock_wrap:
            result = chain.build_retriever(db)

        mock_wrap.assert_called_once_with(base)
        assert result is wrapped
        _, kwargs = db.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == chain.RERANK_CANDIDATE_K


class TestWrapWithReranker:
    """Tests for reranker wiring."""

    def test_builds_contextual_compression_retriever(self):
        base = MagicMock()

        with patch.object(chain, "ContextualCompressionRetriever") as mock_cls:
            chain._wrap_with_reranker(base)

        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["base_retriever"] is base
        assert mock_cls.call_args.kwargs["base_compressor"].top_n == chain.RERANK_TOP_N


class TestLazyCrossEncoder:
    """Tests for lazy cross-encoder loading."""

    def test_scores_via_cached_model(self, monkeypatch):
        chain.reset_reranker_cache()
        mock_model = MagicMock()
        mock_model.score.return_value = [0.9, 0.1]
        monkeypatch.setattr(chain, "_get_cross_encoder", lambda: mock_model)

        scores = chain._LazyCrossEncoder().score([("q", "a"), ("q", "b")])

        assert scores == [0.9, 0.1]
        mock_model.score.assert_called_once_with([("q", "a"), ("q", "b")])


class TestBuildRagChain:
    """Tests for build_rag_chain."""

    def test_returns_invokable_runnable(self, mock_vector_db):
        runnable = chain.build_rag_chain(mock_vector_db)

        assert hasattr(runnable, "invoke")
        mock_vector_db.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": chain.RETRIEVER_K,
                "fetch_k": chain.RETRIEVER_FETCH_K,
                "lambda_mult": chain.MMR_LAMBDA,
            },
        )

    def test_passes_doc_id_scope_to_retriever(self, mock_vector_db):
        chain.build_rag_chain(mock_vector_db, doc_ids=["x"])

        _, kwargs = mock_vector_db.as_retriever.call_args
        assert kwargs["search_kwargs"]["filter"] == {"doc_id": {"$in": ["x"]}}
