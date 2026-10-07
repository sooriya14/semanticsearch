"""Dense retrieval: sentence-transformer embeddings + FAISS.

Embeddings are L2-normalized, so the inner product of two vectors equals
their cosine similarity. That lets us use FAISS's exact IndexFlatIP.
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


def encode_texts(model, texts, batch_size=64):
    """Encode texts into normalized float32 vectors."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def encode_query(model, query):
    """Encode a single query into a (1, dim) normalized float32 array."""
    return encode_texts(model, [query])


def build_faiss_index(embeddings):
    """Exact inner-product index. Row i in the index = row i in papers.csv."""
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index, path):
    faiss.write_index(index, str(path))


def load_faiss_index(path):
    return faiss.read_index(str(path))


def search_faiss(query_embedding, index, top_k=20, mask=None):
    """Return a list of (row_index, similarity) for the top_k nearest papers.

    If a category mask is given we search the whole index (cheap for a
    flat index of ~30k vectors) and keep the first top_k matching papers.
    """
    search_k = index.ntotal if mask is not None else min(top_k, index.ntotal)
    scores, rows = index.search(np.asarray(query_embedding, dtype="float32"), search_k)

    results = []
    for row, score in zip(rows[0], scores[0]):
        if row == -1:
            continue
        if mask is not None and not mask[row]:
            continue
        results.append((int(row), float(score)))
        if len(results) == top_k:
            break
    return results
