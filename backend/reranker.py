"""Cross-encoder reranking.

A bi-encoder (the sentence-transformer) embeds query and paper separately,
which is fast but loses word-level interaction. A cross-encoder reads the
query and the paper *together* and outputs one relevance score. It is too
slow to run on 30k papers, so we only run it on FAISS's top 20 candidates.
"""
from sentence_transformers import CrossEncoder


def load_cross_encoder(model_name):
    return CrossEncoder(model_name)


def rerank(query, candidates, cross_encoder, top_k=5):
    """Score (query, paper text) pairs and return the top_k candidates.

    candidates: list of dicts that each have a "text" key.
    Returns new dicts with an added "rerank_score", sorted high -> low.
    Note: these are raw ranking scores (logits), not probabilities or accuracy.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    scored = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]
