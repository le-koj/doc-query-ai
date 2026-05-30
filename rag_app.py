"""CLI entry point — runs the RAG chatbot in the terminal."""

from dotenv import load_dotenv  # Load API keys from .env

from rag.chain import build_rag_chain  # Build the retrieval + LLM chain
from rag.ingest import load_vector_store  # Load or build the vector store

load_dotenv()

PDF_PATH = "documents/TechCorp_Official_Employee_Handbook.pdf"

vector_db = load_vector_store(PDF_PATH)  # Load existing store or ingest the PDF
rag_chain = build_rag_chain(vector_db)  # Wire retriever → prompt → LLM → parser

print("\nChatbot ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit", "bye"}:
        print("Goodbye!")
        break

    if not question:
        continue

    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
