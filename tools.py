from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing TAVILY_API_KEY. Add it to your .env file before running web search."
        )
    return TavilyClient(api_key=api_key)


def _is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs, and snippets."""
    query = (query or "").strip()
    if not query:
        return "Search query is empty. Please provide a valid topic."

    try:
        tavily = _get_tavily_client()
        results = tavily.search(query=query, max_results=5)
    except Exception as exc:
        return f"Web search failed: {exc}"

    out = []

    for r in results.get("results", []):
        out.append(
            f"Title: {r.get('title', 'N/A')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Snippet: {str(r.get('content', ''))[:300]}\n"
        )

    if not out:
        return "No search results found."

    return "\n----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    url = (url or "").strip()
    if not _is_valid_http_url(url):
        return "Could not scrape URL: invalid URL format."

    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return "Could not scrape URL: page contains no readable text."
        return text[:3000]
    except requests.Timeout:
        return "Could not scrape URL: request timed out."
    except requests.HTTPError as exc:
        return f"Could not scrape URL: HTTP error ({exc.response.status_code})."
    except requests.RequestException as exc:
        return f"Could not scrape URL: network error ({exc})."
    except Exception as e:
        return f"Could not scrape URL: unexpected error ({e})."
