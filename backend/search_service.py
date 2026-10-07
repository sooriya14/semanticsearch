"""Ties the three search modes together.

Used by both the FastAPI app and the evaluation scripts so they always run
exactly the same search code.
"""
from backend import config
from backend.data_utils import category_mask, load_papers, paper_texts
from backend.reranker import load_cross_encoder, rerank
from backend.semantic_search import (
    encode_query,
    load_embedding_model,
    load_faiss_index,
    search_faiss,
)
from backend.tfidf_search import load_tfidf, search_tfidf

MODES = ("tfidf", "semantic", "reranked")


def load_resources():
    """Load the dataset, artifacts and models once. Returns a plain dict."""
    print("Loading papers...")
    papers = load_papers(config.PAPERS_CSV)

    print("Loading TF-IDF artifacts...")
    vectorizer, tfidf_matrix = load_tfidf(config.TFIDF_VECTORIZER_PATH, config.TFIDF_MATRIX_PATH)

    print("Loading FAISS index...")
    faiss_index = load_faiss_index(config.FAISS_INDEX_PATH)

    if not (len(papers) == tfidf_matrix.shape[0] == faiss_index.ntotal):
        raise RuntimeError(
            "papers.csv, the TF-IDF matrix and the FAISS index have different sizes. "
            "Rebuild the artifacts after changing papers.csv."
        )

    print("Loading sentence-transformer and cross-encoder models...")
    return {
        "papers": papers,
        "texts": paper_texts(papers),
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "faiss_index": faiss_index,
        "embedding_model": load_embedding_model(config.EMBEDDING_MODEL_NAME),
        "cross_encoder": load_cross_encoder(config.CROSS_ENCODER_MODEL_NAME),
    }


def _paper_result(papers, row):
    """Metadata for one dataframe row as a dict."""
    paper = papers.iloc[row]
    return {
        "row": row,
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "categories": paper["categories"],
        "update_date": paper["update_date"],
        "url": paper["url"],
        "tfidf_score": None,
        "faiss_score": None,
        "rerank_score": None,
    }


def search(query, mode, resources, category=None, top_k=config.TOP_K,
           num_candidates=config.FAISS_CANDIDATES):
    """Run one search and return a ranked list of result dicts."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    papers = resources["papers"]
    mask = category_mask(papers, category) if category else None

    if mode == "tfidf":
        hits = search_tfidf(query, resources["vectorizer"], resources["tfidf_matrix"], top_k, mask)
        results = []
        for row, score in hits:
            result = _paper_result(papers, row)
            result["tfidf_score"] = score
            results.append(result)

    else:
        query_embedding = encode_query(resources["embedding_model"], query)
        k = top_k if mode == "semantic" else num_candidates
        hits = search_faiss(query_embedding, resources["faiss_index"], k, mask)

        results = []
        for row, score in hits:
            result = _paper_result(papers, row)
            result["faiss_score"] = score
            results.append(result)

        if mode == "reranked":
            for result in results:
                result["text"] = resources["texts"][result["row"]]
            results = rerank(query, results, resources["cross_encoder"], top_k)
            for result in results:
                del result["text"]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results
