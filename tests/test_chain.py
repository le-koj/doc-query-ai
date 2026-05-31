"""Unit tests for rag.chain."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag.chain import (
    MMR_LAMBDA,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    _format_doc_label,
    _format_docs,
    build_rag_chain,
    build_retriever,
)


class TestFormatDocLabel:
    """Tests for the _format_doc_label helper."""

    def test_includes_source_and_one_indexed_page(self):
        doc = Document(page_content="x", metadata={"source": "a.pdf", "page": 2})
        assert _format_doc_label(doc) == "[a.pdf, p. 3]"

    def test_source_only_when_page_missing(self):
        doc = Document(page_content="x", metadata={"source": "a.pdf"})
        assert _format_doc_label(doc) == "[a.pdf]"

    def test_empty_when_no_source(self):
        assert _format_doc_label(Document(page_content="x")) == ""


class TestFormatDocs:
    """Tests for the _format_docs helper."""

    def test_labels_chunks_with_source_and_page(self, sample_documents):
        result = _format_docs(sample_documents)

        assert result == (
            "[handbook.pdf, p. 1]\nFirst chunk about vacation policy.\n\n"
            "[handbook.pdf, p. 2]\nSecond chunk about code of conduct."
        )

    def test_falls_back_to_plain_content_without_metadata(self):
        docs = [Document(page_content="No metadata here.")]
        assert _format_docs(docs) == "No metadata here."

    def test_empty_list_returns_empty_string(self):
        assert _format_docs([]) == ""


class TestBuildRetriever:
    """Tests for build_retriever search configuration."""

    def test_uses_mmr_with_diversity_settings(self):
        db = MagicMock()

        build_retriever(db)

        db.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

    def test_adds_doc_id_filter_when_scoped(self):
        db = MagicMock()

        build_retriever(db, doc_ids=["a", "b"])

        _, kwargs = db.as_retriever.call_args
        assert kwargs["search_kwargs"]["filter"] == {"doc_id": {"$in": ["a", "b"]}}

    def test_no_filter_when_doc_ids_empty(self):
        db = MagicMock()

        build_retriever(db, doc_ids=[])

        _, kwargs = db.as_retriever.call_args
        assert "filter" not in kwargs["search_kwargs"]


class TestBuildRagChain:
    """Tests for build_rag_chain."""

    def test_returns_invokable_runnable(self, mock_vector_db):
        chain = build_rag_chain(mock_vector_db)

        assert hasattr(chain, "invoke")
        mock_vector_db.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

    def test_passes_doc_id_scope_to_retriever(self, mock_vector_db):
        build_rag_chain(mock_vector_db, doc_ids=["x"])

        _, kwargs = mock_vector_db.as_retriever.call_args
        assert kwargs["search_kwargs"]["filter"] == {"doc_id": {"$in": ["x"]}}
