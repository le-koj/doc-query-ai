from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

LLM_MODEL = "gemini-3.5-flash"
RETRIEVER_K = 2

_PROMPT_TEMPLATE = """\
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""


def _format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_db: Chroma) -> Runnable:
    """Wire retriever → prompt → LLM → parser into a single LCEL chain."""
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    prompt = PromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
