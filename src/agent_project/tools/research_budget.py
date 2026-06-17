from __future__ import annotations

import os
import re
from urllib.parse import urlparse


_SEARCH_COUNT = 0
_PAGE_OPEN_URLS: set[str] = set()
_ARXIV_ABS_IDS: set[str] = set()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or parsed.path
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        fragment="",
    ).geturl()


def search_budget_note() -> str:
    soft_limit = max(1, env_int("AGENT_RESEARCH_SEARCH_SOFT_LIMIT", 6))
    if _SEARCH_COUNT < soft_limit:
        return ""
    return (
        f"Research note: {soft_limit}+ web searches have been run in this process. "
        "If you already have several credible sources, stop searching and start "
        "opening, synthesizing, or writing the requested report.\n\n"
    )


def try_consume_search(query: str) -> tuple[bool, str]:
    global _SEARCH_COUNT
    hard_limit = max(1, env_int("AGENT_RESEARCH_SEARCH_HARD_LIMIT", 12))
    if _SEARCH_COUNT >= hard_limit:
        return (
            False,
            f"Research search budget reached ({hard_limit} unique searches in this "
            "process). Do not run more web_search calls for this task. Open or reuse "
            "the best sources already found, synthesize the findings, write the "
            "requested report, or ask the user to explicitly allow more searching.",
        )
    _SEARCH_COUNT += 1
    return True, ""


def _arxiv_article_id(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.netloc.lower().endswith("arxiv.org"):
        return ""
    match = re.match(
        r"^/(?:abs|html|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?(?:\.pdf)?$",
        parsed.path,
    )
    return match.group(1) if match else ""


def try_consume_page_open(url: str) -> tuple[bool, str]:
    normalized_url = normalize_url(url)
    if normalized_url in _PAGE_OPEN_URLS:
        return True, ""

    page_limit = max(1, env_int("AGENT_RESEARCH_PAGE_HARD_LIMIT", 20))
    if len(_PAGE_OPEN_URLS) >= page_limit:
        return (
            False,
            f"Research page-open budget reached ({page_limit} unique pages in this "
            "process). Stop opening new pages for this task; use the sources already "
            "read, synthesize the findings, write the requested report, or ask the "
            "user to explicitly allow more browsing.",
        )

    arxiv_id = _arxiv_article_id(url)
    if arxiv_id and arxiv_id not in _ARXIV_ABS_IDS:
        arxiv_limit = max(1, env_int("AGENT_RESEARCH_ARXIV_ABS_HARD_LIMIT", 8))
        if len(_ARXIV_ABS_IDS) >= arxiv_limit:
            return (
                False,
                f"Research arXiv article budget reached ({arxiv_limit} distinct "
                "arXiv article pages in this process). Stop enumerating nearby arXiv "
                "IDs one by one. Use targeted searches, the sources already found, "
                "and write the requested report, or ask the user to explicitly allow "
                "more arXiv browsing.",
            )
        _ARXIV_ABS_IDS.add(arxiv_id)

    _PAGE_OPEN_URLS.add(normalized_url)
    return True, ""
