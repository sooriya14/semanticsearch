import { useState } from "react";
import { searchPapers } from "./api.js";
import ResultCard from "./ResultCard.jsx";

const MODES = [
  { value: "tfidf", label: "TF-IDF Baseline" },
  { value: "semantic", label: "Semantic Search" },
  { value: "reranked", label: "Semantic + Reranking" },
];

const CATEGORIES = [
  { value: "", label: "All categories" },
  { value: "cs.AI", label: "cs.AI – Artificial Intelligence" },
  { value: "cs.CL", label: "cs.CL – Computation and Language" },
  { value: "cs.CR", label: "cs.CR – Cryptography and Security" },
  { value: "cs.CV", label: "cs.CV – Computer Vision" },
  { value: "cs.IR", label: "cs.IR – Information Retrieval" },
  { value: "cs.LG", label: "cs.LG – Machine Learning" },
  { value: "cs.RO", label: "cs.RO – Robotics" },
  { value: "cs.SE", label: "cs.SE – Software Engineering" },
];

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("reranked");
  const [category, setCategory] = useState("");
  const [results, setResults] = useState([]);
  const [searchedQuery, setSearchedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    try {
      const data = await searchPapers(query.trim(), mode, category);
      setResults(data.results);
      setSearchedQuery(query.trim());
    } catch (err) {
      setError(err.message || "Search failed. Is the backend running?");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>Semantic Paper Search</h1>
      <p className="subtitle">Search computer science papers from arXiv</p>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="e.g. reducing hallucinations in large language models"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>

        <div className="options">
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </label>
          <label>
            Category
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              {CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </label>
        </div>
      </form>

      {error && <div className="error">{error}</div>}
      {loading && <div className="status">Loading results...</div>}
      {!loading && !error && searchedQuery && results.length === 0 && (
        <div className="status">No papers found.</div>
      )}

      <div className="results">
        {results.map((paper) => (
          <ResultCard key={paper.arxiv_id} paper={paper} query={searchedQuery} />
        ))}
      </div>
    </div>
  );
}
