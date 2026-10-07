"""Compute Precision@5 for TF-IDF, semantic (FAISS) and reranked search.

Reads:
    evaluation/system_results.csv   (top-5 rankings from annotate.py)
    evaluation/relevance_labels.csv (your 1/0 labels)
Writes:
    evaluation/results.json

A query is evaluated only if EVERY paper returned by every system for that
query has a label. Unlabeled queries are excluded, never guessed.

Usage: python -m evaluation.evaluate
"""
import json
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_CSV = EVAL_DIR / "queries.csv"
SYSTEM_RESULTS_CSV = EVAL_DIR / "system_results.csv"
LABELS_CSV = EVAL_DIR / "relevance_labels.csv"
RESULTS_JSON = EVAL_DIR / "results.json"

SYSTEMS = {"tfidf": "TF-IDF baseline", "semantic": "Semantic (FAISS)", "reranked": "FAISS + cross-encoder"}


def precision_at_k(ranked_ids, relevant_ids, k=5):
    """Fraction of the top k results that are relevant.

    We always divide by k, so returning fewer than k results is penalised.
    """
    top_k = list(ranked_ids)[:k]
    return sum(1 for pid in top_k if pid in relevant_ids) / k


def evaluate(system_results, labels, k=5):
    labels = labels.astype({"relevant": int})
    label_map = {(q, p): rel for q, p, rel in zip(labels["query_id"], labels["arxiv_id"], labels["relevant"])}

    per_query = []
    skipped = []
    for query_id, group in system_results.groupby("query_id", sort=True):
        if not all((query_id, pid) in label_map for pid in group["arxiv_id"]):
            skipped.append(query_id)
            continue

        relevant = {pid for pid in group["arxiv_id"] if label_map[(query_id, pid)] == 1}
        row = {"query_id": query_id}
        for system in SYSTEMS:
            ranked = group[group["system"] == system].sort_values("rank")["arxiv_id"]
            row[system] = precision_at_k(ranked, relevant, k)
        per_query.append(row)

    summary = {
        "num_evaluated_queries": len(per_query),
        "num_skipped_queries": len(skipped),
        "skipped_query_ids": skipped,
        f"mean_precision_at_{k}": {
            s: (sum(r[s] for r in per_query) / len(per_query) if per_query else None) for s in SYSTEMS
        },
        "mean_latency_ms": {
            s: float(system_results[system_results["system"] == s]
                     .groupby("query_id")["latency_ms"].first().mean())
            for s in SYSTEMS if "latency_ms" in system_results
        },
        "per_query": per_query,
    }
    return summary


def main():
    if not SYSTEM_RESULTS_CSV.exists() or not LABELS_CSV.exists():
        print("Missing system_results.csv or relevance_labels.csv. Run: python -m evaluation.annotate")
        return

    system_results = pd.read_csv(SYSTEM_RESULTS_CSV, dtype={"query_id": str, "arxiv_id": str})
    labels = pd.read_csv(LABELS_CSV, dtype=str)
    summary = evaluate(system_results, labels)

    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluated queries: {summary['num_evaluated_queries']} "
          f"(skipped {summary['num_skipped_queries']} not fully labeled)")
    if summary["num_evaluated_queries"] == 0:
        print("No fully labeled queries yet.")
        return
    print(f"\n{'System':<25}{'Precision@5':>12}{'Avg latency':>14}")
    for system, name in SYSTEMS.items():
        p = summary["mean_precision_at_5"][system]
        lat = summary["mean_latency_ms"].get(system)
        lat_str = f"{lat:.1f} ms" if lat is not None else "n/a"
        print(f"{name:<25}{p:>11.1%}{lat_str:>14}")
    print(f"\nSaved {RESULTS_JSON}")


if __name__ == "__main__":
    main()
