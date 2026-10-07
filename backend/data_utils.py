"""Helpers for cleaning arXiv records and loading the paper dataset."""
import json
import re

import numpy as np
import pandas as pd

PAPER_COLUMNS = ["arxiv_id", "title", "abstract", "categories", "update_date", "url"]


def clean_text(text):
    """Collapse newlines / repeated spaces into single spaces and strip the ends."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_cs_paper(categories):
    """True if at least one category starts with 'cs.' (e.g. 'cs.CL math.ST')."""
    return any(cat.startswith("cs.") for cat in categories.split())


def parse_paper(record):
    """Turn one raw arXiv metadata record into a clean paper dict.

    Returns None if the record is missing an id, title or abstract.
    """
    arxiv_id = clean_text(record.get("id"))
    title = clean_text(record.get("title"))
    abstract = clean_text(record.get("abstract"))
    categories = clean_text(record.get("categories"))

    if not arxiv_id or not title or not abstract:
        return None

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "categories": categories,
        "update_date": clean_text(record.get("update_date")),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def iter_jsonl(path):
    """Yield one parsed JSON object per line, streaming (never loads the whole file)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupted lines


def load_papers(path):
    """Load papers.csv. Every column is read as a string so ids like
    '0704.0001' are not turned into floats."""
    papers = pd.read_csv(path, dtype=str, keep_default_na=False)
    return papers.reset_index(drop=True)


def paper_text(title, abstract):
    """The text we index for every paper: title + abstract."""
    return f"{title}. {abstract}"


def paper_texts(papers):
    """paper_text() for every row of the dataframe."""
    return [paper_text(t, a) for t, a in zip(papers["title"], papers["abstract"])]


def category_mask(papers, category):
    """Boolean array marking papers that match a category filter.

    'cs.CL' matches papers tagged cs.CL; 'cs' matches every cs.* paper.
    """
    category = category.strip()

    def matches(categories):
        return any(c == category or c.startswith(category + ".") for c in categories.split())

    return np.array([matches(c) for c in papers["categories"]], dtype=bool)
