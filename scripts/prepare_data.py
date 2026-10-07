"""Extract a clean computer-science subset of the arXiv metadata snapshot.

Input : data/arxiv-metadata-oai-snapshot.json  (JSON Lines, ~4 GB)
Output: data/papers.csv

The file is streamed line by line. We keep the N most recently updated
valid cs.* papers using a fixed-size min-heap, so memory stays small no
matter how big the input file is.

Usage (from the project root):
    python -m scripts.prepare_data
    python -m scripts.prepare_data --num-papers 32000
"""
import argparse
import heapq

import pandas as pd

from backend import config
from backend.data_utils import PAPER_COLUMNS, is_cs_paper, iter_jsonl, parse_paper


def extract_papers(input_path, num_papers):
    """Stream the snapshot and return up to num_papers clean cs papers,
    newest first (by update_date)."""
    heap = []          # min-heap of (update_date, arxiv_id, paper)
    seen_ids = set()   # for skipping duplicate arXiv ids
    stats = {"lines": 0, "invalid": 0, "not_cs": 0, "duplicates": 0}

    for record in iter_jsonl(input_path):
        stats["lines"] += 1
        if stats["lines"] % 250_000 == 0:
            print(f"  read {stats['lines']:,} lines...")

        paper = parse_paper(record)
        if paper is None:
            stats["invalid"] += 1
            continue
        if not is_cs_paper(paper["categories"]):
            stats["not_cs"] += 1
            continue
        if paper["arxiv_id"] in seen_ids:
            stats["duplicates"] += 1
            continue
        seen_ids.add(paper["arxiv_id"])

        item = (paper["update_date"], paper["arxiv_id"], paper)
        if len(heap) < num_papers:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)  # drop the oldest paper we were keeping

    papers = [item[2] for item in sorted(heap, key=lambda x: x[:2], reverse=True)]
    return papers, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(config.RAW_ARXIV_PATH))
    parser.add_argument("--output", default=str(config.PAPERS_CSV))
    parser.add_argument("--num-papers", type=int, default=32_000)
    args = parser.parse_args()

    print(f"Streaming {args.input} ...")
    papers, stats = extract_papers(args.input, args.num_papers)
    if not papers:
        raise SystemExit("ERROR: no valid cs.* papers found. Is the input the arXiv metadata snapshot?")

    df = pd.DataFrame(papers, columns=PAPER_COLUMNS)
    df.to_csv(args.output, index=False)

    print(f"Lines read: {stats['lines']:,} | skipped invalid: {stats['invalid']:,} | "
          f"non-cs: {stats['not_cs']:,} | duplicates: {stats['duplicates']:,}")
    print(f"Saved {len(df):,} valid papers to {args.output}")
    if len(df) < 30_000:
        print("WARNING: fewer than 30,000 papers were saved.")


if __name__ == "__main__":
    main()
