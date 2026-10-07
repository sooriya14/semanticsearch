"""Terminal tool for labeling query-paper relevance.

Step 1 (once): run all three systems on every query in queries.csv and save
        each system's top 5 to evaluation/system_results.csv (with latency).
Step 2: for each query, pool the papers from all systems, remove duplicates,
        shuffle them (so you can't tell which system found which paper), and
        ask you to label each one 1 = relevant, 0 = not relevant.

Labels are appended to evaluation/relevance_labels.csv after every answer,
so you can quit at any time and resume later.

Usage (from the project root):
    python -m evaluation.annotate
    python -m evaluation.annotate --rerun-retrieval   # redo step 1
"""
import argparse
import random
import textwrap
import time
from pathlib import Path

import pandas as pd

from backend.search_service import MODES, load_resources, search

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_CSV = EVAL_DIR / "queries.csv"
SYSTEM_RESULTS_CSV = EVAL_DIR / "system_results.csv"
LABELS_CSV = EVAL_DIR / "relevance_labels.csv"


def run_retrieval(queries, resources, top_k=5):
    """Run every system on every query. Returns a dataframe of rankings."""
    search("warm up", "reranked", resources)  # first call is slower; don't time it

    rows = []
    for _, q in queries.iterrows():
        for mode in MODES:
            start = time.perf_counter()
            results = search(q["query"], mode, resources, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            for r in results:
                rows.append({
                    "query_id": q["query_id"], "system": mode, "rank": r["rank"],
                    "arxiv_id": r["arxiv_id"], "latency_ms": round(latency_ms, 2),
                })
        print(f"  retrieved results for {q['query_id']}")
    return pd.DataFrame(rows)


def load_labels():
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV, dtype=str)
    return pd.DataFrame(columns=["query_id", "arxiv_id", "relevant"])


def append_label(query_id, arxiv_id, relevant):
    write_header = not LABELS_CSV.exists()
    pd.DataFrame([{"query_id": query_id, "arxiv_id": arxiv_id, "relevant": relevant}]).to_csv(
        LABELS_CSV, mode="a", header=write_header, index=False)


def ask_label():
    while True:
        answer = input("Relevant? [1 = yes, 0 = no, s = skip, q = quit]: ").strip().lower()
        if answer in {"1", "0", "s", "q"}:
            return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-retrieval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    queries = pd.read_csv(QUERIES_CSV, dtype=str)
    resources = load_resources()

    if args.rerun_retrieval or not SYSTEM_RESULTS_CSV.exists():
        print("Running all systems on all queries...")
        run_retrieval(queries, resources).to_csv(SYSTEM_RESULTS_CSV, index=False)
        print(f"Saved rankings to {SYSTEM_RESULTS_CSV}")
    system_results = pd.read_csv(SYSTEM_RESULTS_CSV, dtype=str)

    labels = load_labels()
    labeled = set(zip(labels["query_id"], labels["arxiv_id"]))
    papers = resources["papers"].set_index("arxiv_id")
    rng = random.Random(args.seed)

    total_pairs = len(system_results[["query_id", "arxiv_id"]].drop_duplicates())
    print(f"\nLabeled so far: {len(labeled)} of {total_pairs} query-paper pairs.")

    for _, q in queries.iterrows():
        # Pool the papers from all systems and remove duplicates
        pool = sorted(set(system_results.loc[system_results["query_id"] == q["query_id"], "arxiv_id"]))
        todo = [pid for pid in pool if (q["query_id"], pid) not in labeled]
        if not todo:
            continue
        rng.shuffle(todo)  # hide which system produced which paper

        for i, arxiv_id in enumerate(todo, start=1):
            paper = papers.loc[arxiv_id]
            print("\n" + "=" * 80)
            print(f"QUERY [{q['query_id']}]: {q['query']}    (paper {i}/{len(todo)} for this query)")
            print("-" * 80)
            print(f"TITLE: {paper['title']}\n")
            print(textwrap.fill(paper["abstract"], width=80))
            print("-" * 80)

            answer = ask_label()
            if answer == "q":
                print("Progress saved. Run the script again to continue.")
                return
            if answer == "s":
                continue
            append_label(q["query_id"], arxiv_id, int(answer))
            labeled.add((q["query_id"], arxiv_id))
            print(f"Saved. Overall progress: {len(labeled)}/{total_pairs}")

    print("\nAll pooled query-paper pairs are labeled. Now run: python -m evaluation.evaluate")


if __name__ == "__main__":
    main()
