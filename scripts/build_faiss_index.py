"""Build an exact inner-product FAISS index from data/embeddings.npy.

Usage: python -m scripts.build_faiss_index
"""
import numpy as np

from backend import config
from backend.semantic_search import build_faiss_index, save_faiss_index


def main():
    embeddings = np.load(config.EMBEDDINGS_PATH)
    print(f"Building IndexFlatIP for {embeddings.shape[0]:,} vectors of dim {embeddings.shape[1]}")
    index = build_faiss_index(embeddings)
    save_faiss_index(index, config.FAISS_INDEX_PATH)
    print(f"Saved FAISS index ({index.ntotal:,} vectors) to {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
