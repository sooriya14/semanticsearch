// Small wrappers around the FastAPI backend.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

export async function searchPapers(query, mode, category) {
  const params = new URLSearchParams({ q: query, mode });
  if (category) params.set("category", category);
  const response = await fetch(`${API_URL}/search?${params}`);
  return handleResponse(response);
}

export async function explainPaper(query, arxivId) {
  const response = await fetch(`${API_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, arxiv_id: arxivId }),
  });
  return handleResponse(response);
}
