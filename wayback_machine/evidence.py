"""VENDORED FROZEN COPY of ``src/website_evidence.py``.

This is a deliberate copy, not an import. The whole research design rests on the
2023 evidence being cleaned EXACTLY the way today's evidence was, so any measured
change is real and not a tooling artifact. Vendoring keeps this sub-project
free of ``src`` imports (so it can be lifted out), while
``tests/test_evidence_golden.py`` asserts this function produces output
identical to the live ``src`` version on sample inputs — failing loudly the
moment the two drift apart.

Do NOT "improve" this file. If the live cleaner changes, re-vendor it verbatim
and re-run the golden test.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_MAX_EVIDENCE_CHARS = None
DEFAULT_MAX_PAGE_CHARS = None

# If cleaned evidence is shorter than this, omit it entirely. Thin fragments
# are worse than no website evidence because they can mislead the classifier.
MIN_USEFUL_EVIDENCE_CHARS = 100

_BOILERPLATE_EXACT = {
    "top of page",
    "bottom of page",
    "skip to content",
    "skip to main content",
    "scroll to top",
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "terms & conditions",
    "accept cookies",
    "book a demo",
    "book a call",
    "blog",
    "booking faq",
    "other links",
    "rental links",
    "social media",
    "get in touch",
    "contact us",
    "learn more",
    "read more",
}

_BOILERPLATE_CONTAINS = (
    "something went wrong",
    "all rights reserved",
    "thank you for joining",
    "i wish to subscribe",
    "please leave your email",
    "our website uses cookies",
    "copyright ",
    "© ",
)

_SIGNAL_TERMS = (
    "about",
    "ai",
    "api",
    "automation",
    "case stud",
    "customer",
    "data",
    "docs",
    "how it works",
    "industry",
    "integration",
    "machine learning",
    "model",
    "platform",
    "pricing",
    "product",
    "research",
    "service",
    "solution",
    "technical",
    "use case",
    "workflow",
)


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _is_image_or_asset_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", stripped):
        return True
    if lower.startswith("!["):
        return True
    return bool(re.search(r"\.(png|jpe?g|gif|webp|svg|avif)(\)|\?|$)", lower))


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower().strip("[]()#*:_-. ")
    if not lower:
        return True
    if "{{" in lower and "}}" in lower:
        return True
    if lower in _BOILERPLATE_EXACT:
        return True
    if len(lower) <= 3 and not any(ch.isalpha() for ch in lower):
        return True
    if any(marker in lower for marker in _BOILERPLATE_CONTAINS):
        return True
    if lower.startswith(("tel:", "mailto:")):
        return True
    if re.fullmatch(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", lower):
        return True
    if re.fullmatch(r"[\d\s().+-]{7,}", lower):
        return True
    if lower in {"home", "contact", "login", "sign in", "sign up", "close menu"}:
        return True
    return False


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = re.sub(r"\W+", "", line).lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return out


def _strip_boilerplate_prefix(line: str) -> str:
    return re.sub(
        r"^(social media|other links|rental links)\s+\[\.\.\.\]\s*", "", line, flags=re.I
    ).strip()


def _shape_signal_text(text: str) -> str:
    """Remove obvious page chrome while preserving classifier-relevant claims."""
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    kept: list[str] = []
    for line in cleaned.split("\n"):
        stripped = _strip_boilerplate_prefix(line.strip())
        if _is_image_or_asset_line(stripped) or _is_boilerplate_line(stripped):
            continue
        if stripped.startswith("[") and "](" in stripped and len(stripped) < 80:
            continue
        kept.append(stripped)

    deduped = _dedupe_lines(kept)
    if len("\n".join(deduped)) <= 20_000:
        return "\n".join(deduped).strip()

    signal_lines: list[str] = []
    regular_lines: list[str] = []
    for line in deduped:
        lower = line.lower()
        if line.startswith("#") or any(term in lower for term in _SIGNAL_TERMS):
            signal_lines.append(line)
        else:
            regular_lines.append(line)

    shaped: list[str] = []
    total = 0
    for line in signal_lines + regular_lines:
        projected = total + len(line) + 1
        if projected > 20_000:
            break
        shaped.append(line)
        total = projected
    return "\n".join(shaped).strip()


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def _page_kind(url: str) -> str:
    """Label pages by URL route instead of inferred content category."""
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return "homepage"
    return path.rsplit("/", maxsplit=1)[-1] or "homepage"


def compact_tavily_response(
    response: dict[str, Any],
    max_evidence_chars: int | None = DEFAULT_MAX_EVIDENCE_CHARS,
    max_page_chars: int | None = DEFAULT_MAX_PAGE_CHARS,
) -> tuple[str, str]:
    """Return (`website_pages_used`, `website_evidence`) from one Tavily response."""
    pages = response.get("results")
    if not isinstance(pages, list):
        return "", ""

    evidence_blocks: list[str] = []
    urls: list[str] = []
    for idx, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        url = str(page.get("url", "")).strip()
        raw_content = _shape_signal_text(str(page.get("raw_content", "")))
        if not url or not raw_content:
            continue
        urls.append(url)
        content = _truncate(raw_content, max_page_chars)
        evidence_blocks.append(f"[Page {idx}: {_page_kind(url)}]\nURL: {url}\n{content}")

    combined = "\n\n".join(evidence_blocks)
    if len(combined) < MIN_USEFUL_EVIDENCE_CHARS:
        return "", ""

    return " | ".join(urls), _truncate(combined, max_evidence_chars)
