"""Encode every paper (title + abstract) with all-MiniLM-L6-v2.

Embeddings are normalized and saved to data/embeddings.npy so they only
have to be computed once. Row i of the array = row i of papers.csv.

Usage: python -m scripts.build_embeddings [--batch-size 64]
"""
import argparse

import numpy as np

from backend import config
from backend.data_utils import load_papers, paper_texts
from backend.semantic_search import encode_texts, load_embedding_model

CHUNK_SIZE = 1024  # how many papers to encode between progress messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    papers = load_papers(config.PAPERS_CSV)
    texts = paper_texts(papers)
    print(f"Loading {config.EMBEDDING_MODEL_NAME} ...")
    model = load_embedding_model(config.EMBEDDING_MODEL_NAME)
    print(f"Encoding {len(texts):,} papers on device: {model.device}")

    chunks = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start:start + CHUNK_SIZE]
        chunks.append(encode_texts(model, chunk, batch_size=args.batch_size))
        done = start + len(chunk)
        print(f"  encoded {done:,}/{len(texts):,} ({100 * done / len(texts):.1f}%)")

    embeddings = np.vstack(chunks)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {config.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
