import sys
import os
import torch

# Add parent directory to path so we can import rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("Testing BAAI/bge-m3 embedding generation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target hardware device: {device}")
    
    print("Initializing HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    test_text = "This is a quick verification of the local BGE-M3 embedding model."
    print(f"Generating embedding for text: '{test_text}'")
    vector = embeddings.embed_query(test_text)
    
    print("Embedding generated successfully!")
    print(f"Vector dimensions: {len(vector)} (Expected: 1024)")
    print(f"First 5 values: {vector[:5]}")
    
    if len(vector) == 1024:
        print("Verification SUCCESS.")
        sys.exit(0)
    else:
        print("Verification FAILURE: Dimension mismatch.")
        sys.exit(1)

if __name__ == "__main__":
    main()
