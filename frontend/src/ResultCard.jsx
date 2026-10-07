import { useState } from "react";
import { explainPaper } from "./api.js";

// Show whichever score the chosen search mode produced.
function scoreLabel(paper) {
  if (paper.rerank_score !== null) {
    return `Rerank score: ${paper.rerank_score.toFixed(2)} · FAISS similarity: ${paper.faiss_score.toFixed(3)}`;
  }
  if (paper.faiss_score !== null) return `FAISS cosine similarity: ${paper.faiss_score.toFixed(3)}`;
  if (paper.tfidf_score !== null) return `TF-IDF cosine similarity: ${paper.tfidf_score.toFixed(3)}`;
  return "";
}

export default function ResultCard({ paper, query }) {
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleExplain() {
    setLoading(true);
    setError("");
    try {
      const data = await explainPaper(query, paper.arxiv_id);
      setExplanation(data.explanation);
    } catch (err) {
      setError(err.message || "Could not generate an explanation.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="rank">#{paper.rank}</span>
        <h2>
          <a href={paper.url} target="_blank" rel="noreferrer">{paper.title}</a>
        </h2>
      </div>

      <div className="meta">
        <span>{paper.categories}</span>
        <span>Updated {paper.update_date}</span>
        <a href={paper.url} target="_blank" rel="noreferrer">arXiv:{paper.arxiv_id}</a>
      </div>

      <p className="abstract">{paper.abstract}</p>
      <div className="score">{scoreLabel(paper)}</div>

      <button className="explain-button" onClick={handleExplain} disabled={loading}>
        {loading ? "Asking Qwen..." : "Why is this relevant?"}
      </button>

      {error && <div className="error">{error}</div>}
      {explanation && (
        <div className="explanation">
          <strong>Explanation (based only on the abstract):</strong>
          <p>{explanation}</p>
        </div>
      )}
    </div>
  );
}
