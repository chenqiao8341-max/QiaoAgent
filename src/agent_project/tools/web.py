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
_SEARCH_CACHE: dict[str, str] = {}
_SEARCH_COUNT = 0
_SEEN_SEARCH_URLS: set[str] = set()


def _env_bool(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _clamp_timeout(timeout_seconds: int) -> int:
    return max(1, min(timeout_seconds, 30))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.casefold()).strip()


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or parsed.path
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        fragment="",
    ).geturl()


def _research_budget_note() -> str:
    soft_limit = max(1, _env_int("AGENT_RESEARCH_SEARCH_SOFT_LIMIT", 6))
    if _SEARCH_COUNT < soft_limit:
        return ""
    return (
        f"Research note: {soft_limit}+ web searches have been run in this process. "
        "If you already have several credible sources, stop searching and start "
        "opening, synthesizing, or writing the requested report.\n\n"
    )


def _fetch_url_bytes(url: str, timeout_seconds: int = 10) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported.")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=_clamp_timeout(timeout_seconds)) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")

    return raw, content_type


def _fetch_url(url: str, timeout_seconds: int = 10) -> tuple[str, str]:
    raw, content_type = _fetch_url_bytes(url, timeout_seconds=timeout_seconds)
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
    global _SEARCH_COUNT
    emit_progress(f"searching web: {query}")
    if not _env_bool("AGENT_ENABLE_WEB_SEARCH", True):
        emit_progress("web search skipped: disabled by AGENT_ENABLE_WEB_SEARCH")
        return "Web search is disabled by AGENT_ENABLE_WEB_SEARCH."

    normalized_query = _normalize_query(query)
    if normalized_query in _SEARCH_CACHE:
        emit_progress(f"web search cache hit: {query}")
        return (
            "Duplicate search query. Reusing cached result; prefer opening a source "
            "or writing with the information already gathered.\n\n"
            + _SEARCH_CACHE[normalized_query]
        )

    hard_limit = max(1, _env_int("AGENT_RESEARCH_SEARCH_HARD_LIMIT", 12))
    if _SEARCH_COUNT >= hard_limit:
        emit_progress(f"web search blocked by research hard limit: {query}")
        return (
            f"Research search budget reached ({hard_limit} unique searches in this "
            "process). Do not run more web_search calls for this task. Open or reuse "
            "the best sources already found, synthesize the findings, write the "
            "requested report, or ask the user to explicitly allow more searching."
        )

    _SEARCH_COUNT += 1
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

    new_count = 0
    seen_count = 0
    lines = [_research_budget_note() + f"Search results for: {query}"]
    for index, (title, href) in enumerate(results, start=1):
        normalized_url = _normalize_url(href)
        if normalized_url in _SEEN_SEARCH_URLS:
            marker = "seen"
            seen_count += 1
        else:
            marker = "new"
            new_count += 1
            _SEEN_SEARCH_URLS.add(normalized_url)
        lines.append(f"{index}. [{marker}] {title}\n   {href}")
    if seen_count and not new_count:
        lines.append(
            "\nResearch note: this search produced no new URLs beyond earlier searches. "
            "Use the sources already found or change strategy instead of repeating searches."
        )
    elif seen_count:
        lines.append(
            f"\nResearch note: {seen_count} result URL(s) were already seen; prioritize "
            "the new sources or start synthesis."
        )
    output = "\n".join(lines)
    _SEARCH_CACHE[normalized_query] = output
    return output
