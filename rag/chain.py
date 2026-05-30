"""RAG chain construction for document question answering.

Builds a LangChain Expression Language (LCEL) pipeline that retrieves
relevant document chunks from ChromaDB and generates answers with Google
Gemini.
"""

from langchain_chroma import Chroma  # Vector store used as the document retriever
from langchain_core.output_parsers import StrOutputParser  # Extract plain text from LLM responses
from langchain_core.prompts import PromptTemplate  # Format prompts sent to the LLM
from langchain_core.runnables import Runnable, RunnablePassthrough  # Compose and pass data through the chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model

LLM_MODEL = "gemini-3.5-flash"  # Google Gemini model used for answer generation
RETRIEVER_K = 2  # Number of document chunks retrieved per query

_PROMPT_TEMPLATE = """\
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""


def _format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context block.

    Args:
        docs: List of LangChain Document objects returned by the retriever.

    Returns:
        str: All chunk texts joined with blank lines.
    """
    return "\n\n".join(doc.page_content for doc in docs)  # Merge chunk texts into one string


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
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})  # Fetch top-k similar chunks
    prompt = PromptTemplate.from_template(_PROMPT_TEMPLATE)  # Build the RAG prompt template
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)  # Initialise Gemini with deterministic output

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )  # Retrieve context, fill the prompt, generate an answer, return plain text
