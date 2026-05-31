"""CLI entry point for the Doc Query AI chatbot.

Runs an interactive terminal session where users can ask questions about the
indexed document library. Type ``exit``, ``quit``, or ``bye`` to close the
session.

Usage:
    python rag_app.py
"""

from dotenv import load_dotenv  # Load API keys from a .env file

from rag.chain import build_rag_chain  # Build the retrieval + LLM chain
from rag.ingest import list_documents, load_vector_store  # Load the library vector store

load_dotenv()  # Read GOOGLE_API_KEY and other secrets from .env

SEED_PDF_PATH = "documents/TechCorp_Official_Employee_Handbook.pdf"  # Optional seed document


def main() -> None:
    """Load the library, print its contents, and run an interactive Q&A loop.

    Exits with status code 1 when no documents are indexed and no seed PDF is
    available.
    """
    vector_db = load_vector_store(SEED_PDF_PATH)  # Load existing library or seed it on first run
    if vector_db is None:
        print(
            f"\nNo indexed documents yet. Place a PDF at {SEED_PDF_PATH}, "
            "or run the web UI (`python app.py`) to upload one.\n"
        )
        raise SystemExit(1)

    documents = list_documents(vector_db)  # Report what the library currently contains
    print(f"\nLibrary contains {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['source']} ({doc['chunks']} chunks)")

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


if __name__ == "__main__":
    main()
