"""RAG chain construction for document question answering.

Builds a LangChain Expression Language (LCEL) pipeline that retrieves
relevant document chunks from ChromaDB and generates answers with Google
Gemini. Retrieved context is labelled with its source document so answers can
cite where information came from across a multi-PDF library.
"""

from langchain_chroma import Chroma  # Vector store used as the document retriever
from langchain_core.documents import Document  # Typed document chunks
from langchain_core.output_parsers import StrOutputParser  # Extract plain text from LLM responses
from langchain_core.prompts import PromptTemplate  # Format prompts sent to the LLM
from langchain_core.runnables import Runnable, RunnablePassthrough  # Compose and pass data through the chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model

LLM_MODEL = "gemini-2.5-flash"  # Google Gemini model used for answer generation
RETRIEVER_K = 6  # Number of document chunks retrieved per query across the library

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


def _format_doc_label(doc: Document) -> str:
    """Build a ``[source, p. N]`` label from a chunk's metadata.

    Args:
        doc: A retrieved LangChain Document.

    Returns:
        str: A bracketed source label, or an empty string when no source
            metadata is present.
    """
    metadata = doc.metadata or {}
    source = metadata.get("source")
    if not source:
        return ""
    page = metadata.get("page")
    if isinstance(page, int):
        return f"[{source}, p. {page + 1}]"  # PyPDF pages are 0-indexed
    return f"[{source}]"


def _format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single, source-labelled context block.

    Args:
        docs: List of LangChain Document objects returned by the retriever.

    Returns:
        str: All chunk texts joined with blank lines, each prefixed with its
            source label when available.
    """
    blocks = []
    for doc in docs:
        label = _format_doc_label(doc)
        blocks.append(f"{label}\n{doc.page_content}" if label else doc.page_content)
    return "\n\n".join(blocks)


def build_retriever(vector_db: Chroma):
    """Create a top-k retriever over the document library.

    Args:
        vector_db: A Chroma vector store containing embedded document chunks.

    Returns:
        A retriever that returns the ``RETRIEVER_K`` most similar chunks.
    """
    return vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})


def build_rag_chain(vector_db: Chroma) -> Runnable:
    """Wire retriever, prompt, LLM, and parser into a single LCEL chain.

    The resulting chain accepts a plain-text question string and returns a
    plain-text answer grounded in the most relevant document chunks.

    Args:
        vector_db: A Chroma vector store containing embedded document chunks.

    Returns:
        Runnable: An invokable LangChain chain with the signature
            ``question: str -> answer: str``.
    """
    retriever = build_retriever(vector_db)  # Fetch top-k similar chunks across the library
    prompt = PromptTemplate.from_template(_PROMPT_TEMPLATE)  # Build the RAG prompt template
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)  # Initialise Gemini with deterministic output

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )  # Retrieve context, fill the prompt, generate an answer, return plain text
