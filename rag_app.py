import os  # Access environment variables and file paths

from dotenv import load_dotenv  # Load API keys from a .env file
from langchain_community.document_loaders import PyPDFLoader  # Extract text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks for embedding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Gemini LLM and embedding models
from langchain_community.vectorstores import Chroma  # Persist and query document embeddings locally
from langchain_core.prompts import PromptTemplate  # Format prompts sent to the LLM
from langchain_core.runnables import RunnablePassthrough  # Pass inputs through the RAG chain unchanged

# Load environment variables from .env file
load_dotenv()

print("Loading PDF documents...")  # Status message while the PDF is being read
loader = PyPDFLoader("documents/TechCorp_Official_Employee_Handbook.pdf")  # Point the loader at the handbook PDF
document = loader.load()  # Parse the PDF into a list of page-level Document objects

#print(document[0].page_content)  # Print the extracted text from the first page

print("Splitting documents into chunks...")  # Status message while the document is being split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Split the document into chunks of 500 characters with 50 character overlap
chunks = text_splitter.split_documents(document)  # Split the document into chunks
print(f"Split into {len(chunks)} chunks")  # Print the number of chunks

print(chunks[0].page_content)

print("Embedding chunks and creating vector store...")  # Status message while the chunks are being embedded
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")  # Create an embedding model
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("\n", "Vector store created successfully")  # Status message when the vector store is created

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Define the hidden prompt structure for the LLM
template = """ 
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Test the RAG chain
user_question = input("Enter your question: ")
print("\n", "User question:", user_question)

response = rag_chain.invoke(user_question)
print(f"Answer: {response.content}")
