from __future__ import annotations

import os
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool

from agent_project.tools.web import _fetch_url


def _env_bool(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._in_title = False
        self._href: str | None = None
        self._link_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            attr_map = dict(attrs)
            href = attr_map.get("href")
            if href:
                self._href = urljoin(self.base_url, href)
                self._link_text_parts = []
                self.text_parts.append(" ")
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "li", "br", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._href is not None:
            self._link_text_parts.append(data)
        self.text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._href is not None:
            label = _normalize_space("".join(self._link_text_parts)) or self._href
            parsed = urlparse(self._href)
            if parsed.scheme in {"http", "https"} and (label, self._href) not in self.links:
                self.links.append((label, self._href))
            self.text_parts.append(" ")
            self._href = None
            self._link_text_parts = []

    def page_text(self) -> str:
        text = "\n".join(_normalize_space(part) for part in "".join(self.text_parts).splitlines())
        return re.sub(r"\n{3,}", "\n\n", text).strip()


@tool
def open_web_page(url: str, max_chars: int = 20000, timeout_seconds: int = 15) -> str:
    """Open a web page with a lightweight text browser and return readable text."""
    if not _env_bool("AGENT_ENABLE_BROWSER_TOOLS", True):
        return "Browser tools are disabled by AGENT_ENABLE_BROWSER_TOOLS."

    try:
        html, content_type = _fetch_url(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        return f"Browser open error: {exc}"

    if "html" not in content_type.lower() and content_type:
        return f"Fetched non-HTML content from {url}. Content-Type: {content_type}"

    parser = _PageParser(url)
    parser.feed(html)
    title = _normalize_space(parser.title) or "(no title)"
    text = parser.page_text()
    output = f"URL: {url}\nTitle: {title}\n\n{text}" if text else f"URL: {url}\nTitle: {title}\n\n(no readable text found)"

    if max_chars > 0 and len(output) > max_chars:
        return output[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return output


@tool
def list_web_page_links(url: str, max_links: int = 30, timeout_seconds: int = 15) -> str:
    """Open a web page and list links found in the HTML."""
    if not _env_bool("AGENT_ENABLE_BROWSER_TOOLS", True):
        return "Browser tools are disabled by AGENT_ENABLE_BROWSER_TOOLS."

    try:
        html, _content_type = _fetch_url(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        return f"Browser link extraction error: {exc}"

    parser = _PageParser(url)
    parser.feed(html)
    links = parser.links[: max(1, min(max_links, 100))]
    if not links:
        return f"No links found on {url}."

    lines = [f"Links found on {url}:"]
    for index, (label, href) in enumerate(links, start=1):
        lines.append(f"{index}. {label}\n   {href}")
    return "\n".join(lines)
