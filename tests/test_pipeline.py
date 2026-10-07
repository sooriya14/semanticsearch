"""Small, fast tests on a tiny synthetic dataset (no model downloads needed)."""
import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.data_utils import category_mask, clean_text, is_cs_paper, parse_paper
from backend.reranker import rerank
from backend.semantic_search import build_faiss_index, search_faiss
from backend.tfidf_search import build_tfidf, search_tfidf
from evaluation.evaluate import evaluate, precision_at_k
from scripts.prepare_data import extract_papers

TINY_PAPERS = pd.DataFrame({
    "arxiv_id": ["0001", "0002", "0003", "0004"],
    "title": ["Graph neural networks", "Image segmentation", "Machine translation", "Speech recognition"],
    "abstract": [
        "We study message passing graph neural networks for molecules.",
        "A convolutional network for medical image segmentation.",
        "Neural machine translation with attention between languages.",
        "End-to-end speech recognition with transformers.",
    ],
    "categories": ["cs.LG", "cs.CV eess.IV", "cs.CL", "cs.CL eess.AS"],
    "update_date": ["2023-01-01"] * 4,
    "url": ["https://arxiv.org/abs/000%d" % i for i in range(1, 5)],
})


# ---------- dataset cleaning ----------

def test_clean_text_collapses_whitespace():
    assert clean_text("  Deep\n  learning \t rocks ") == "Deep learning rocks"
    assert clean_text(None) == ""


def test_parse_paper_skips_missing_fields():
    assert parse_paper({"id": "1", "title": "  ", "abstract": "x", "categories": "cs.AI"}) is None
    paper = parse_paper({"id": "2101.00001", "title": "A\n title", "abstract": " An abstract ",
                         "categories": "cs.AI", "update_date": "2021-01-05"})
    assert paper["title"] == "A title"
    assert paper["url"] == "https://arxiv.org/abs/2101.00001"


def test_is_cs_paper():
    assert is_cs_paper("math.ST cs.LG")
    assert not is_cs_paper("physics.optics")


def test_extract_papers_filters_and_dedupes(tmp_path):
    records = [
        {"id": "a", "title": "T1", "abstract": "A1", "categories": "cs.AI", "update_date": "2020-01-01"},
        {"id": "a", "title": "T1 dup", "abstract": "A1", "categories": "cs.AI", "update_date": "2020-01-01"},
        {"id": "b", "title": "", "abstract": "A2", "categories": "cs.AI", "update_date": "2021-01-01"},
        {"id": "c", "title": "T3", "abstract": "A3", "categories": "hep-th", "update_date": "2022-01-01"},
        {"id": "d", "title": "T4", "abstract": "A4", "categories": "cs.CL", "update_date": "2023-01-01"},
    ]
    path = tmp_path / "snapshot.json"
    path.write_text("\n".join(json.dumps(r) for r in records))

    papers, stats = extract_papers(path, num_papers=10)
    assert [p["arxiv_id"] for p in papers] == ["d", "a"]  # newest first
    assert stats["duplicates"] == 1 and stats["invalid"] == 1 and stats["not_cs"] == 1


def test_category_mask():
    assert category_mask(TINY_PAPERS, "cs.CL").tolist() == [False, False, True, True]
    assert category_mask(TINY_PAPERS, "eess").tolist() == [False, True, False, True]


# ---------- TF-IDF ----------

def test_tfidf_search_returns_relevant_result():
    texts = (TINY_PAPERS["title"] + ". " + TINY_PAPERS["abstract"]).tolist()
    vectorizer, matrix = build_tfidf(texts)
    results = search_tfidf("graph neural networks", vectorizer, matrix, top_k=5)
    assert len(results) > 0
    assert results[0][0] == 0  # row of the graph paper


# ---------- FAISS ----------

def test_faiss_returns_requested_number_of_candidates():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(50, 16)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_faiss_index(embeddings)

    results = search_faiss(embeddings[7:8], index, top_k=20)
    assert len(results) == 20
    assert results[0][0] == 7                        # a vector is closest to itself
    assert abs(results[0][1] - 1.0) < 1e-5           # cosine similarity of 1


def test_faiss_respects_category_mask():
    embeddings = np.eye(4, dtype="float32")
    index = build_faiss_index(embeddings)
    mask = np.array([False, True, False, True])
    results = search_faiss(embeddings[0:1], index, top_k=20, mask=mask)
    assert {row for row, _ in results} == {1, 3}


# ---------- reranker ----------

class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text."""
    def predict(self, pairs):
        return [sum(w in text.lower() for w in query.lower().split()) for query, text in pairs]


def test_reranker_sorts_descending():
    candidates = [{"arxiv_id": i, "text": t} for i, t in enumerate(TINY_PAPERS["abstract"])]
    results = rerank("neural machine translation attention", candidates, FakeCrossEncoder(), top_k=3)
    scores = [r["rerank_score"] for r in results]
    assert len(results) == 3
    assert scores == sorted(scores, reverse=True)
    assert results[0]["arxiv_id"] == 2


# ---------- evaluation ----------

def test_precision_at_5():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "c"}) == 0.4
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], {"f"}) == 0.0  # only top 5 count
    assert precision_at_k(["a"], {"a"}) == 0.2                          # divide by k, not len


def test_evaluate_skips_unlabeled_queries():
    system_results = pd.DataFrame({
        "query_id": ["q1"] * 3 + ["q2"],
        "system": ["tfidf", "semantic", "reranked", "tfidf"],
        "rank": [1, 1, 1, 1],
        "arxiv_id": ["p1", "p2", "p2", "p9"],
        "latency_ms": [1.0, 2.0, 3.0, 1.0],
    })
    labels = pd.DataFrame({"query_id": ["q1", "q1"], "arxiv_id": ["p1", "p2"], "relevant": ["0", "1"]})
    summary = evaluate(system_results, labels)
    assert summary["num_evaluated_queries"] == 1
    assert summary["skipped_query_ids"] == ["q2"]
    assert summary["mean_precision_at_5"] == {"tfidf": 0.0, "semantic": 0.2, "reranked": 0.2}


# ---------- API ----------

def test_health_endpoint():
    from backend import main
    # Not using `with TestClient(...)` skips the startup model loading.
    main.STATE["papers"] = TINY_PAPERS
    try:
        response = TestClient(main.app).get("/health")
    finally:
        main.STATE.clear()
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "num_papers": 4}
