"""Unit tests for rag.chain."""

from rag.chain import _format_docs, build_rag_chain


class TestFormatDocs:
    """Tests for the _format_docs helper."""

    def test_joins_chunks_with_double_newline(self, sample_documents):
        result = _format_docs(sample_documents)

        assert result == (
            "First chunk about vacation policy.\n\n"
            "Second chunk about code of conduct."
        )

    def test_empty_list_returns_empty_string(self):
        assert _format_docs([]) == ""


class TestBuildRagChain:
    """Tests for build_rag_chain."""

    def test_returns_invokable_runnable(self, mock_vector_db):
        chain = build_rag_chain(mock_vector_db)

        assert hasattr(chain, "invoke")
        mock_vector_db.as_retriever.assert_called_once_with(search_kwargs={"k": 2})
