"""Central place for file paths and model names.

Everything is a plain constant so it is easy to find and change.
DATA_DIR can be overridden with an environment variable (useful for tests
or for trying the pipeline on a small sample without touching data/).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

# Raw input (downloaded manually from Kaggle) and the cleaned dataset
RAW_ARXIV_PATH = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
PAPERS_CSV = DATA_DIR / "papers.csv"

# TF-IDF baseline artifacts
TFIDF_VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.npz"

# Semantic search artifacts
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local LLM used for RAG explanations
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Search sizes
TOP_K = 5              # results shown to the user
FAISS_CANDIDATES = 20  # candidates passed to the cross-encoder
