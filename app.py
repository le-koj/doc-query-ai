import gc
import os
import shutil

import gradio as gr
from dotenv import load_dotenv

from rag.chain import build_rag_chain
from rag.ingest import build_vector_store, load_vector_store

load_dotenv()

PDF_PATH = "documents/TechCorp_Official_Employee_Handbook.pdf"

# Build the chain once at startup; reused for every user query
print("Initialising RAG pipeline...")
_vector_db = load_vector_store(PDF_PATH)
_rag_chain = build_rag_chain(_vector_db)
print("Ready.\n")


def answer(question: str, history: list) -> str:
    """Gradio chat handler — invokes the RAG chain and returns a plain-text answer."""
    if not question.strip():
        return "Please enter a question."
    return _rag_chain.invoke(question)


def ingest_pdf(file) -> str:
    """Re-index a newly uploaded PDF and rebuild the chain."""
    global _vector_db, _rag_chain

    if file is None:
        return "No file uploaded."

    # Release the ChromaDB connection before deleting the directory.
    # Without this the underlying SQLite file stays locked and the new
    # Chroma.from_documents() call fails with "readonly database".
    _vector_db = None
    _rag_chain = None
    gc.collect()

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    _vector_db = build_vector_store(file.name)
    _rag_chain = build_rag_chain(_vector_db)
    return f"Indexed: {os.path.basename(file.name)}"


with gr.Blocks(title="Doc Query AI") as demo:
    gr.Markdown(
        """
        # Doc Query AI
        Ask natural-language questions about your PDF documents, powered by
        **ChromaDB** vector search and **Google Gemini**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload a new document")
            upload = gr.File(label="PDF file", file_types=[".pdf"])
            upload_btn = gr.Button("Index document")
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(fn=ingest_pdf, inputs=upload, outputs=upload_status)

        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=answer,
                examples=[
                    "What is the vacation policy?",
                    "What are the code of conduct rules?",
                    "How does the performance review process work?",
                ],
            )

if __name__ == "__main__":
    demo.launch()
