"""RAG pipeline package for document ingestion and question answering.

This package provides modules for loading PDF documents into a ChromaDB
vector store and building a LangChain retrieval-augmented generation chain.

Modules:
    ingest: PDF loading, chunking, embedding, and library management.
    chain: MMR retrieval, optional reranking, and Gemini answer generation.
"""
