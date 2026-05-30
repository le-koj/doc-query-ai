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

print(document[0].page_content)  # Print the extracted text from the first page