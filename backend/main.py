"""FastAPI backend for the Semantic Paper Search Engine.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import config
from backend.rag import OllamaUnavailableError, explain_relevance
from backend.search_service import load_resources, search

# Everything loaded at startup lives here (models are NOT reloaded per request).
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE.update(load_resources())
    print(f"Ready: {len(STATE['papers'])} papers indexed.")
    yield
    STATE.clear()


app = FastAPI(title="Semantic Paper Search", lifespan=lifespan)

# Allow the local Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------

class HealthResponse(BaseModel):
    status: str
    num_papers: int


class PaperResult(BaseModel):
    rank: int
    arxiv_id: str
    title: str
    abstract: str
    categories: str
    update_date: str
    url: str
    tfidf_score: Optional[float] = None
    faiss_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    category: Optional[str]
    results: List[PaperResult]


class ExplainRequest(BaseModel):
    query: str
    arxiv_id: str


class ExplainResponse(BaseModel):
    arxiv_id: str
    title: str
    explanation: str
    model: str


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    papers = STATE.get("papers")
    return HealthResponse(status="ok", num_papers=0 if papers is None else len(papers))


@app.get("/search", response_model=SearchResponse)
def search_papers(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: Literal["tfidf", "semantic", "reranked"] = "reranked",
    category: Optional[str] = Query(None, description="e.g. cs.CL or cs"),
):
    category = category.strip() if category and category.strip() else None
    results = search(q.strip(), mode, STATE, category=category)
    return SearchResponse(query=q, mode=mode, category=category, results=results)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    papers = STATE["papers"]
    matches = papers[papers["arxiv_id"] == request.arxiv_id.strip()]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Paper {request.arxiv_id} not found.")

    paper = matches.iloc[0]
    try:
        explanation = explain_relevance(
            request.query, paper["title"], paper["abstract"],
            model=config.OLLAMA_MODEL, host=config.OLLAMA_HOST,
        )
    except OllamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ExplainResponse(
        arxiv_id=paper["arxiv_id"], title=paper["title"],
        explanation=explanation, model=config.OLLAMA_MODEL,
    )
