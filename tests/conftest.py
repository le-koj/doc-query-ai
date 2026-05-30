"""Shared pytest fixtures for the Doc Query AI test suite."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return a small list of source-tagged Document objects for unit tests."""
    return [
        Document(
            page_content="First chunk about vacation policy.",
            metadata={"source": "handbook.pdf", "page": 0, "doc_id": "abc123"},
        ),
        Document(
            page_content="Second chunk about code of conduct.",
            metadata={"source": "handbook.pdf", "page": 1, "doc_id": "abc123"},
        ),
    ]


@pytest.fixture
def mock_rag_chain():
    """Return a mock chain whose invoke() returns a fixed answer string."""
    chain = MagicMock()
    chain.invoke.return_value = "Mocked answer."
    return chain


@pytest.fixture
def mock_vector_db():
    """Return a mock Chroma vector store with a retriever stub."""
    db = MagicMock()
    db.as_retriever.return_value = MagicMock()
    return db
