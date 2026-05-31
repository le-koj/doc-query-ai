"""RAG chain construction for document question answering.

Builds a LangChain Expression Language (LCEL) pipeline that retrieves
relevant document chunks from ChromaDB and generates answers with Google
Gemini. Retrieved context is labelled with its source document so answers can
cite where information came from across a multi-PDF library.

When reranking is enabled, vector search + MMR produce a candidate pool and a
local cross-encoder reranker reorders them before the LLM sees the context.
"""

import os  # Read reranking toggles from the environment

from langchain_chroma import Chroma  # Vector store used as the document retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document  # Typed document chunks
from langchain_core.output_parsers import StrOutputParser  # Extract plain text from LLM responses
from langchain_core.prompts import PromptTemplate  # Format prompts sent to the LLM
from langchain_core.runnables import Runnable, RunnablePassthrough  # Compose and pass data through the chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model

LLM_MODEL = "gemini-3.1-flash-lite"  # Google Gemini model used for answer generation
RETRIEVER_K = 6  # Number of document chunks returned to the LLM when reranking is off
RETRIEVER_FETCH_K = 20  # Candidate pool size MMR selects from before diversifying
MMR_LAMBDA = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity; 0.5 balances both

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "30"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", str(RETRIEVER_K)))

_cross_encoder: HuggingFaceCrossEncoder | None = None

_PROMPT_TEMPLATE = """\
You are answering questions about a library of documents. Use only the
retrieved context below to answer the question. Each context block is labelled
with its source document and page.

If the answer is not in the context, say you don't know. If different documents
disagree, point out the disagreement and cite each source. Keep the answer
concise (three sentences maximum) and mention the source document(s) you used.

Context:
{context}

Question: {question}

Answer:"""


class _LazyCrossEncoder(BaseCrossEncoder):
    """Load the HuggingFace cross-encoder only on the first rerank call."""

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return _get_cross_encoder().score(text_pairs)


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Return a cached HuggingFace cross-encoder for reranking."""
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading reranker model: {RERANK_MODEL}")
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=RERANK_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _cross_encoder


def reset_reranker_cache() -> None:
    """Clear the cached cross-encoder (used in tests)."""
    global _cross_encoder
    _cross_encoder = None


def retrieval_status_line() -> str:
    """Return a one-line summary of the active retrieval configuration."""
    if RERANK_ENABLED:
        return (
            f"MMR (\u03bb={MMR_LAMBDA}, candidates={RERANK_CANDIDATE_K}) \u2192 "
            f"rerank `{RERANK_MODEL}` (top_n={RERANK_TOP_N})"
        )
    return f"MMR (k={RETRIEVER_K}, fetch_k={RETRIEVER_FETCH_K}, \u03bb={MMR_LAMBDA})"


def _format_doc_label(doc: Document) -> str:
    """Build a ``[source, p. N]`` label from a chunk's metadata."""
    metadata = doc.metadata or {}
    source = metadata.get("source")
    if not source:
        return ""
    page = metadata.get("page")
    if isinstance(page, int):
        return f"[{source}, p. {page + 1}]"  # PyPDF pages are 0-indexed
    return f"[{source}]"


def _format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single, source-labelled context block."""
    blocks = []
    for doc in docs:
        label = _format_doc_label(doc)
        blocks.append(f"{label}\n{doc.page_content}" if label else doc.page_content)
    return "\n\n".join(blocks)


def _mmr_search_kwargs(doc_ids: list[str] | None, *, rerank: bool) -> dict:
    """Build MMR search kwargs for plain or rerank-first retrieval."""
    if rerank:
        search_kwargs = {
            "k": RERANK_CANDIDATE_K,
            "fetch_k": max(RERANK_CANDIDATE_K, RETRIEVER_FETCH_K),
            "lambda_mult": MMR_LAMBDA,
        }
    else:
        search_kwargs = {
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        }
    if doc_ids:
        search_kwargs["filter"] = {"doc_id": {"$in": list(doc_ids)}}
    return search_kwargs


def _wrap_with_reranker(base_retriever):
    """Wrap a base retriever with a cross-encoder reranker."""
    compressor = CrossEncoderReranker(model=_LazyCrossEncoder(), top_n=RERANK_TOP_N)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_retriever(vector_db: Chroma, doc_ids: list[str] | None = None):
    """Create a retriever over the document library, optionally scoped and reranked.

    With reranking enabled, MMR returns ``RERANK_CANDIDATE_K`` diverse candidates
    and a local cross-encoder keeps the best ``RERANK_TOP_N`` for the LLM.
    """
    base = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs=_mmr_search_kwargs(doc_ids, rerank=RERANK_ENABLED),
    )
    if RERANK_ENABLED:
        return _wrap_with_reranker(base)
    return base


def build_rag_chain(vector_db: Chroma, doc_ids: list[str] | None = None) -> Runnable:
    """Wire retriever, prompt, LLM, and parser into a single LCEL chain."""
    retriever = build_retriever(vector_db, doc_ids)
    prompt = PromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
