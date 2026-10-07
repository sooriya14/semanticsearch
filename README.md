# Semantic Paper Search Engine

Search ~32,000 computer-science papers from arXiv by *meaning*, not just keywords.

- **Semantic retrieval** with a Hugging Face sentence-transformer (`all-MiniLM-L6-v2`) + a FAISS vector index
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) of the top 20 candidates
- **TF-IDF keyword baseline** (scikit-learn) for comparison
- **RAG relevance explanations**: a local Qwen 3 model (via Ollama) explains why a paper matches your query, using only that paper's abstract
- **FastAPI** backend + **React (Vite)** frontend with a category filter
- **Evaluation**: Precision@5 of all three systems, measured from your own relevance labels

No database, auth, Docker, or cloud: just local files and three Python models.

---

## Architecture

**Semantic + reranking pipeline (main system)**

```
User query
 → sentence transformer (all-MiniLM-L6-v2)
 → query embedding (384-dim, normalized)
 → FAISS retrieves 20 candidates (exact inner product = cosine similarity)
 → cross-encoder reranks the 20 candidates
 → top 5 papers returned
```

**Optional explanation (RAG)**

```
selected paper's title + abstract (retrieved from papers.csv)
 + original query
 → Qwen 3 through Ollama (prompt: "use only this abstract, don't invent claims")
 → grounded 2–3 sentence relevance explanation
```

**Baseline**

```
User query
 → TF-IDF vector
 → cosine similarity against every paper's TF-IDF vector
 → top 5 results
```

**Offline build steps (run once)**

```
arxiv-metadata-oai-snapshot.json ─► prepare_data ─► papers.csv ─┬─► build_tfidf ─► tfidf_vectorizer.joblib + tfidf_matrix.npz
                                                                └─► build_embeddings ─► embeddings.npy ─► build_faiss_index ─► faiss.index
```

Row `i` of `papers.csv` = row `i` of the TF-IDF matrix = vector `i` in FAISS. That is how a search hit is mapped back to the paper's metadata.

## Repository structure

```
semantic-paper-search/
  backend/
    config.py            # file paths, model names, Ollama settings
    data_utils.py        # cleaning arXiv records, loading papers.csv, category filter
    tfidf_search.py      # TF-IDF baseline: build / save / load / search
    semantic_search.py   # embeddings + FAISS: encode / build index / search
    reranker.py          # cross-encoder reranking
    rag.py               # prompt + Ollama call for relevance explanations
    search_service.py    # loads everything once; runs tfidf / semantic / reranked search
    main.py              # FastAPI app: /health, /search, /explain
  scripts/
    prepare_data.py      # stream the arXiv snapshot → data/papers.csv
    build_tfidf.py       # fit TF-IDF → data/tfidf_*.{joblib,npz}
    build_embeddings.py  # encode papers → data/embeddings.npy
    build_faiss_index.py # embeddings → data/faiss.index
  evaluation/
    queries.csv          # 30 research queries
    annotate.py          # terminal labeling tool (blind, pooled)
    evaluate.py          # Precision@5 → results.json
  frontend/              # React + Vite + plain CSS
  tests/test_pipeline.py
  data/                  # put the arXiv snapshot here; generated artifacts go here too
  requirements.txt
```

## Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, streaming JSON Lines |
| Baseline | scikit-learn `TfidfVectorizer`, cosine similarity |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), PyTorch |
| Vector search | FAISS `IndexFlatIP` |
| Reranking | sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) |
| RAG | Ollama + Qwen 3 (`qwen3:4b` by default), `ollama` Python package, no LangChain |
| API | FastAPI, Pydantic, Uvicorn |
| UI | React 18, Vite, plain CSS |

## How each piece works

### TF-IDF (baseline)
Each paper's `title + abstract` becomes a sparse vector with one weight per vocabulary word.
The weight is high when a word is frequent in *this* paper (TF) but rare across the corpus (IDF).
The query is transformed with the same fitted vectorizer and compared to every paper with cosine similarity.
It is fast and exact, but it only matches **shared words**: "LLM" and "large language model" look unrelated.

### Sentence-transformer embeddings
`all-MiniLM-L6-v2` is a small BERT-style model trained so that texts with similar meaning get nearby vectors.
Each paper becomes one dense 384-dimensional vector. Vectors are L2-normalized, so the dot product of two
vectors equals their cosine similarity. Papers are encoded once (`build_embeddings.py`) and saved; at search
time only the query is encoded.

### FAISS
FAISS is a library for fast nearest-neighbour search over vectors. With ~32k vectors, an exact
`IndexFlatIP` (brute-force inner product) is fast enough (milliseconds), so no approximate index is needed.
Given the query vector, it returns the row indexes and similarity scores of the closest papers.

### Cross-encoder reranking
The sentence-transformer is a *bi-encoder*: it embeds the query and paper separately, which is fast but coarse.
A *cross-encoder* reads the query and the paper **together** in one forward pass and outputs a relevance score,
so it can model word-level interactions, which makes it more accurate. It is too slow to run on every paper, so it only reranks
FAISS's top 20. Its scores are unbounded ranking scores (logits), **not** probabilities or accuracy.

### RAG (retrieval-augmented generation)
1. **Retrieve**: look up the selected paper's real title and abstract in `papers.csv`.
2. **Augment**: put them in the prompt next to the user's query.
3. **Generate**: Qwen 3 writes a 2–3 sentence explanation. The prompt says to use only the provided text,
   not to invent claims, and to say so if relevance is weak. Grounding the LLM in retrieved text keeps it
   from hallucinating about papers it has never seen.

---

## Setup and run

All commands are run from the `semantic-paper-search/` folder unless noted.

### 0. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 1. Download the arXiv metadata snapshot

The official snapshot is hosted on Kaggle: <https://www.kaggle.com/datasets/Cornell-University/arxiv>
(about 1.5 GB zipped, ~5.6 GB unzipped, JSON Lines, one paper per line).

Either download it in the browser and unzip it, or use the Kaggle CLI (needs a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv -p data --unzip
```

The file must end up at:

```
data/arxiv-metadata-oai-snapshot.json
```

### 2. Extract ~32,000 papers

```bash
python -m scripts.prepare_data                  # default: 32,000 papers
python -m scripts.prepare_data --num-papers 40000
```

This streams the file line by line (never loading it all into memory), keeps papers with at least one
`cs.*` category, skips papers missing a title/abstract, skips duplicate IDs, cleans whitespace, and keeps the
**most recently updated** N papers (using a fixed-size heap). Output: `data/papers.csv`.

### 3. Build the TF-IDF baseline

```bash
python -m scripts.build_tfidf
```

Output: `data/tfidf_vectorizer.joblib`, `data/tfidf_matrix.npz`.

### 4. Generate embeddings

```bash
python -m scripts.build_embeddings              # optional: --batch-size 128
```

Downloads `all-MiniLM-L6-v2` on first run (~90 MB), prints progress, and saves `data/embeddings.npy`.
Uses Apple Silicon (MPS) or CUDA automatically if available. Expect several minutes on a laptop for 32k papers.

### 5. Build the FAISS index

```bash
python -m scripts.build_faiss_index
```

Output: `data/faiss.index`.

> If you ever change `papers.csv`, rerun steps 3–5. The API refuses to start if the three artifacts have different sizes.

### 6. Start Ollama and pull Qwen 3

Install Ollama from <https://ollama.com/download>, then:

```bash
ollama serve                 # skip if the Ollama desktop app is already running
ollama pull qwen3:4b
```

To use a different model (for example one you already have):

```bash
export OLLAMA_MODEL=qwen3:8b
```

If Ollama is not running, `/explain` returns HTTP 503 with a clear message. Search keeps working.

### 7. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

All models and indexes are loaded **once** at startup. Try it:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/search?q=graph+neural+networks+for+molecules&mode=reranked"
curl "http://localhost:8000/search?q=speech+recognition&mode=semantic&category=cs.CL"
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" \
     -d '{"query": "graph neural networks for molecules", "arxiv_id": "<an id from a search result>"}'
```

Interactive API docs: <http://localhost:8000/docs>

| Endpoint | Description |
|---|---|
| `GET /health` | `{status, num_papers}` |
| `GET /search?q=...&mode=tfidf\|semantic\|reranked&category=cs.CL` | top 5 papers with scores |
| `POST /explain` `{query, arxiv_id}` | Qwen 3 relevance explanation grounded in the abstract |

The `category` filter matches exact categories (`cs.CL`) or prefixes (`cs`, `eess`).

### 8. Run the React frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. (Set `VITE_API_URL` if the API is not on `http://localhost:8000`.)

### 9. Run the tests

```bash
pytest
```

The tests use a tiny synthetic dataset and fake/random models, so they need no downloads.

---

## Evaluation (Precision@5)

**Precision@5** = (number of relevant papers among the top 5) / 5, averaged over queries.

Relevance is judged by **you**, using *pooling* (the standard approach in IR evaluation, e.g. TREC):
for each query, the top 5 from all three systems are merged and de-duplicated, and you label each
pooled paper once. You don't see which system returned which paper, so the labels aren't biased.

### Step 1: label

```bash
python -m evaluation.annotate
```

- The first run executes all 3 systems on all 30 queries in `evaluation/queries.csv` and saves their
  rankings plus latency to `evaluation/system_results.csv`.
- Then it shows each pooled paper (shuffled) and asks: `1` = relevant, `0` = not relevant, `s` = skip, `q` = quit.
- Every answer is saved immediately to `evaluation/relevance_labels.csv`. Quit anytime and rerun to resume.
- Expect roughly 8–12 unique papers per query (~300 labels for 30 queries).
- `--rerun-retrieval` recomputes rankings (do this if you rebuild the indexes).

### Step 2: compute metrics

```bash
python -m evaluation.evaluate
```

This prints mean Precision@5 and average latency for TF-IDF, semantic FAISS, and FAISS + cross-encoder,
and saves everything (including per-query scores) to `evaluation/results.json`. Queries with any unlabeled
pooled paper are **excluded**, never guessed.

Report the numbers this script prints. The repository contains no pre-computed metrics.
