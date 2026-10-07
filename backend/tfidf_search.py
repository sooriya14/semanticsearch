"""TF-IDF keyword-search baseline.

Each paper (title + abstract) becomes a sparse vector of word weights.
A query is turned into a vector with the same vectorizer and compared to
every paper with cosine similarity.
"""
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(texts):
    """Fit a TF-IDF vectorizer on the corpus and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_tfidf(vectorizer, matrix, vectorizer_path, matrix_path):
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, matrix)


def load_tfidf(vectorizer_path, matrix_path):
    return joblib.load(vectorizer_path), sparse.load_npz(matrix_path)


def search_tfidf(query, vectorizer, matrix, top_k=5, mask=None):
    """Return a list of (row_index, score) for the top_k most similar papers.

    mask: optional boolean array; papers where mask is False are excluded.
    """
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    if mask is not None:
        scores = np.where(mask, scores, -1.0)

    top_rows = np.argsort(-scores)[:top_k]
    # Drop excluded papers and papers that share no words with the query
    return [(int(row), float(scores[row])) for row in top_rows if scores[row] > 0]
