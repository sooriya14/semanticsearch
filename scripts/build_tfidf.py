"""Fit the TF-IDF baseline on papers.csv and save it to disk.

Usage: python -m scripts.build_tfidf
"""
from backend import config
from backend.data_utils import load_papers, paper_texts
from backend.tfidf_search import build_tfidf, save_tfidf


def main():
    papers = load_papers(config.PAPERS_CSV)
    print(f"Fitting TF-IDF on {len(papers):,} papers...")
    vectorizer, matrix = build_tfidf(paper_texts(papers))
    save_tfidf(vectorizer, matrix, config.TFIDF_VECTORIZER_PATH, config.TFIDF_MATRIX_PATH)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}; matrix shape: {matrix.shape}")
    print(f"Saved {config.TFIDF_VECTORIZER_PATH} and {config.TFIDF_MATRIX_PATH}")


if __name__ == "__main__":
    main()
