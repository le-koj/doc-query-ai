"""CLI entry point for the Doc Query AI chatbot.

Runs an interactive terminal session where users can ask questions about
indexed PDF documents. Type ``exit``, ``quit``, or ``bye`` to close the
session.
"""

from dotenv import load_dotenv  # Load API keys from a .env file

from rag.chain import build_rag_chain  # Build the retrieval + LLM chain
from rag.ingest import load_vector_store  # Load or build the vector store

load_dotenv()  # Read GOOGLE_API_KEY and other secrets from .env

PDF_PATH = "documents/TechCorp_Official_Employee_Handbook.pdf"  # Default document to index

vector_db = load_vector_store(PDF_PATH)  # Load existing store or ingest the PDF on first run
if vector_db is None:
    print(
        f"\nNo indexed documents yet. Place a PDF at {PDF_PATH}, "
        "or run the web UI (`python app.py`) to upload one.\n"
    )
    raise SystemExit(1)

rag_chain = build_rag_chain(vector_db)  # Wire retriever → prompt → LLM → parser

print("\nChatbot ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()  # Read the user's question from stdin

    if question.lower() in {"exit", "quit", "bye"}:
        print("Goodbye!")  # Acknowledge the exit command
        break  # Leave the interactive loop

    if not question:
        continue  # Ignore blank input and wait for the next question

    answer = rag_chain.invoke(question)  # Send the question through the RAG chain
    print(f"\nAnswer: {answer}\n")  # Display the model's grounded response
