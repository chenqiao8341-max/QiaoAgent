from __future__ import annotations

import os
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


USER_AGENT = "agent-project/0.1 (+https://example.local)"


def _env_bool(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _clamp_timeout(timeout_seconds: int) -> int:
    return max(1, min(timeout_seconds, 30))


def _fetch_url(url: str, timeout_seconds: int = 10) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported.")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=_clamp_timeout(timeout_seconds)) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")

    encoding = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1).strip('"')

    return raw.decode(encoding, errors="replace"), content_type


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        class_names = set(
            (attr_map.get("class") or "")
            .replace(chr(39), " ")
            .replace(chr(34), " ")
            .split()
        )
        if not ({"result-link", "result__a"} & class_names):
            return
        href = attr_map.get("href")
        if not href:
            return
        self._href = href
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return

        title = _normalize_space("".join(self._text_parts))
        href = _clean_duckduckgo_url(self._href)
        self._href = None
        self._text_parts = []

        if not title or not href:
            return
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"}:
            return
        if parsed.netloc.endswith("duckduckgo.com"):
            return
        if (title, href) not in self.results:
            self.results.append((title, href))


def _clean_duckduckgo_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/"):
        href = urljoin("https://duckduckgo.com", href)

    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return href


@tool
def web_search(query: str, max_results: int = 5, timeout_seconds: int = 10) -> str:
    """Search the web and return result titles and URLs."""
    emit_progress(f"searching web: {query}")
    if not _env_bool("AGENT_ENABLE_WEB_SEARCH", True):
        emit_progress("web search skipped: disabled by AGENT_ENABLE_WEB_SEARCH")
        return "Web search is disabled by AGENT_ENABLE_WEB_SEARCH."

    max_results = max(1, min(max_results, 10))
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    try:
        html, _content_type = _fetch_url(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        emit_progress(f"web search failed: {exc}")
        return f"Web search error: {exc}"

    if "anomaly.js" in html or "challenge-form" in html:
        emit_progress("web search blocked by provider anti-bot challenge")
        return "Web search error: search provider returned an anti-bot challenge."

    parser = _SearchParser()
    parser.feed(html)
    results = parser.results[:max_results]
    if not results:
        emit_progress(f"web search complete: {query} (0 results)")
        return "No web search results found."

    emit_progress(f"web search complete: {query} ({len(results)} results)")
    for title, href in results[:3]:
        emit_progress(f"search result: {title} -> {href}")

    lines = [f"Search results for: {query}"]
    for index, (title, href) in enumerate(results, start=1):
        lines.append(f"{index}. {title}\n   {href}")
    return "\n".join(lines)
