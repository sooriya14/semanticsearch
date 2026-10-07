"""RAG relevance explanation with a local Qwen 3 model served by Ollama.

Retrieval-Augmented Generation here means:
  1. Retrieve: look up the real title + abstract of the chosen paper.
  2. Augment:  put that text into the prompt next to the user's query.
  3. Generate: ask the LLM to explain relevance using ONLY that text.
"""
import re

import ollama


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not running or the model is not available."""


def build_prompt(query, title, abstract):
    return f"""You are helping a researcher decide whether a paper matches their search.

Search query: {query}

Paper title: {title}

Paper abstract: {abstract}

In 2 to 3 sentences, explain why this paper is or is not relevant to the search query.
Rules:
- Use ONLY the title and abstract provided above.
- Do not invent methods, results, or claims that are not supported by the abstract.
- If the relevance is weak or unclear, say so plainly."""


def strip_thinking(text):
    """Qwen 3 may emit <think>...</think> reasoning; keep only the final answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def explain_relevance(query, title, abstract, model, host):
    """Ask the local LLM for a grounded explanation. Returns plain text."""
    prompt = build_prompt(query, title, abstract)
    try:
        client = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            think=False,                   # skip Qwen 3's long reasoning mode
            options={"temperature": 0.2},  # keep answers focused and repeatable
        )
    except ollama.ResponseError as e:
        # e.g. model not pulled yet
        raise OllamaUnavailableError(
            f"Ollama returned an error for model '{model}': {e.error}. "
            f"Try: ollama pull {model}"
        ) from e
    except Exception as e:  # connection refused, timeout, ...
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {host}. Is it running? Start it with: ollama serve"
        ) from e

    return strip_thinking(response.message.content)
