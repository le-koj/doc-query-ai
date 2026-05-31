"""Smoke-test script for local BGE-M3 embedding generation.

Verifies that ``HuggingFaceEmbeddings`` can load ``BAAI/bge-m3`` and produce
a 1024-dimensional vector. Run from the project root with the venv active:

    python scratch/validate_embeddings.py
"""

import os
import sys

import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Allow importing sibling ``rag`` package when run as a standalone script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EMBEDDING_MODEL = "BAAI/bge-m3"
EXPECTED_DIMENSIONS = 1024


def main() -> None:
    """Load BGE-M3, embed a sample string, and verify vector dimensionality.

    Prints diagnostic output and exits with code 0 on success or 1 on failure.
    """
    print("Testing BAAI/bge-m3 embedding generation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target hardware device: {device}")

    print("Initializing HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    test_text = "This is a quick verification of the local BGE-M3 embedding model."
    print(f"Generating embedding for text: '{test_text}'")
    vector = embeddings.embed_query(test_text)

    print("Embedding generated successfully!")
    print(f"Vector dimensions: {len(vector)} (Expected: {EXPECTED_DIMENSIONS})")
    print(f"First 5 values: {vector[:5]}")

    if len(vector) == EXPECTED_DIMENSIONS:
        print("Verification SUCCESS.")
        sys.exit(0)

    print("Verification FAILURE: Dimension mismatch.")
    sys.exit(1)


if __name__ == "__main__":
    main()
