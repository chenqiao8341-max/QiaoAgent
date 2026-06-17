from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.research_budget import normalize_url, try_consume_page_open
from agent_project.tools.web import _fetch_url, _fetch_url_bytes


_OPENED_URL_COUNTS: dict[str, int] = {}
_PAGE_CACHE: dict[str, str] = {}


def _env_bool(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _record_opened_url(url: str) -> int:
    normalized = normalize_url(url)
    count = _OPENED_URL_COUNTS.get(normalized, 0) + 1
    _OPENED_URL_COUNTS[normalized] = count
    return count


def _truncate_output(output: str, max_chars: int) -> str:
    if max_chars > 0 and len(output) > max_chars:
        return output[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return output


def _pdf_text(raw: bytes, timeout_seconds: int) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(raw)
        pdf_file.flush()
        completed = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", pdf_file.name, "-"],
            text=True,
            capture_output=True,
            timeout=max(1, min(timeout_seconds, 30)),
            check=False,
        )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


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
        if tag in {
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "li",
            "br",
            "h1",
            "h2",
            "h3",
        }:
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
    emit_progress(f"opening web page: {url}")
    if not _env_bool("AGENT_ENABLE_BROWSER_TOOLS", True):
        emit_progress("browser open skipped: disabled by AGENT_ENABLE_BROWSER_TOOLS")
        return "Browser tools are disabled by AGENT_ENABLE_BROWSER_TOOLS."

    normalized_url = normalize_url(url)
    open_count = _record_opened_url(url)
    repeated_note = ""
    if open_count > 1:
        repeated_note = (
            f"Note: this URL has already been opened {open_count - 1} time(s) "
            "in this process. For research tasks, prefer using the earlier "
            "result or opening a different source unless this repeat is necessary.\n\n"
        )
    if normalized_url in _PAGE_CACHE:
        emit_progress(f"web page cache hit: {url}")
        return _truncate_output(repeated_note + _PAGE_CACHE[normalized_url], max_chars)

    allowed, budget_message = try_consume_page_open(url)
    if not allowed:
        emit_progress(f"browser open blocked by research hard limit: {url}")
        return repeated_note + budget_message

    try:
        html, content_type = _fetch_url(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        emit_progress(f"browser open failed: {exc}")
        output = (
            f"Browser open error for {url}: {exc}\n\n"
            "Research note: do not retry this exact URL in the same task. Prefer an "
            "accessible mirror such as arXiv, INSPIRE, PDG, HFLAV, CERN CDS, or an "
            "official collaboration page."
        )
        _PAGE_CACHE[normalized_url] = output
        return _truncate_output(repeated_note + output, max_chars)

    if "pdf" in content_type.lower() or url.lower().split("?", 1)[0].endswith(".pdf"):
        try:
            raw, _pdf_content_type = _fetch_url_bytes(url, timeout_seconds=timeout_seconds)
            text = _pdf_text(raw, timeout_seconds=timeout_seconds)
        except Exception as exc:
            emit_progress(f"PDF text extraction failed: {exc}")
            text = ""

        if not text:
            emit_progress(f"browser fetched PDF but could not extract text: {url}")
            return (
                repeated_note
                + f"Fetched PDF content from {url}, but text extraction failed. "
                "The local pdftotext command may be unavailable or the PDF may be scanned."
            )

        emit_progress(f"PDF read complete: {url} ({len(text)} text chars)")
        output = f"URL: {url}\nContent-Type: {content_type}\n\n{text}"
        _PAGE_CACHE[normalized_url] = output
        return _truncate_output(repeated_note + output, max_chars)

    if "html" not in content_type.lower() and content_type:
        emit_progress(f"browser fetched non-HTML content: {url} ({content_type})")
        return (
            repeated_note
            + f"Fetched non-HTML content from {url}. Content-Type: {content_type}"
        )

    parser = _PageParser(url)
    parser.feed(html)
    title = _normalize_space(parser.title) or "(no title)"
    text = parser.page_text()
    emit_progress(
        f"web page read complete: {title} ({len(text)} text chars, {len(parser.links)} links)"
    )
    if text:
        output = f"URL: {url}\nTitle: {title}\n\n{text}"
    else:
        output = (
            f"URL: {url}\nTitle: {title}\n\n(no readable text found)\n\n"
            "Research note: this page did not expose readable text to the lightweight "
            "browser. Do not spend more calls on this exact page; use an accessible "
            "PDF, arXiv, INSPIRE metadata, PDG/HFLAV, or another readable source."
        )

    if 0 < len(text) < 1000:
        output += (
            "\n\nResearch note: this page exposed very little readable text. Treat it "
            "as weak evidence and prefer fuller sources before citing details."
        )

    _PAGE_CACHE[normalized_url] = output
    return _truncate_output(repeated_note + output, max_chars)


@tool
def list_web_page_links(url: str, max_links: int = 30, timeout_seconds: int = 15) -> str:
    """Open a web page and list links found in the HTML."""
    emit_progress(f"listing links on web page: {url}")
    if not _env_bool("AGENT_ENABLE_BROWSER_TOOLS", True):
        emit_progress("link listing skipped: disabled by AGENT_ENABLE_BROWSER_TOOLS")
        return "Browser tools are disabled by AGENT_ENABLE_BROWSER_TOOLS."

    normalized_url = normalize_url(url)
    if normalized_url not in _PAGE_CACHE:
        allowed, budget_message = try_consume_page_open(url)
        if not allowed:
            emit_progress(f"link listing blocked by research hard limit: {url}")
            return budget_message

    try:
        html, _content_type = _fetch_url(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        emit_progress(f"link extraction failed: {exc}")
        return f"Browser link extraction error: {exc}"

    parser = _PageParser(url)
    parser.feed(html)
    links = parser.links[: max(1, min(max_links, 100))]
    if not links:
        emit_progress(f"link listing complete: {url} (0 links)")
        return f"No links found on {url}."

    emit_progress(f"link listing complete: {url} ({len(links)} links shown)")
    lines = [f"Links found on {url}:"]
    for index, (label, href) in enumerate(links, start=1):
        lines.append(f"{index}. {label}\n   {href}")
    return "\n".join(lines)
