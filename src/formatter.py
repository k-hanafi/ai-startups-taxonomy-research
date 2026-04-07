"""Format CSV rows into user messages for the classification prompt.

Each row becomes a structured text block matching the INPUT FORMAT section
of the system prompt. The formatter maps raw CSV column names to the
prompt's expected field names and handles missing fields gracefully.
"""

from __future__ import annotations

from typing import Any

# Truncation guard: 10K chars ~ 2,500 tokens. Well within context limits
# even with the ~2,400-token system prompt + schema prefix.
MAX_USER_MESSAGE_CHARS: int = 10_000


def _extract_year(date_str: str) -> str:
    """Pull a 4-digit year from Crunchbase date strings like '01nov2016'."""
    cleaned = str(date_str).strip()
    if not cleaned or cleaned.lower() in ("nan", "none", "nat"):
        return "Unknown"
    for i in range(len(cleaned) - 3):
        chunk = cleaned[i : i + 4]
        if chunk.isdigit() and 1900 <= int(chunk) <= 2100:
            return chunk
    return cleaned


def _clean(value: Any) -> str:
    """Convert a value to a stripped string. Treat NaN/None as blank."""
    s = str(value).strip() if value is not None else ""
    if s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def _merge_keywords(row: dict[str, Any]) -> str:
    """Combine category_list and category_groups_list into one Keywords field.

    The model benefits from both the specific tags (category_list) and the
    broader groupings (category_groups_list) to make classification decisions.
    """
    cats = _clean(row.get("category_list", ""))
    groups = _clean(row.get("category_groups_list", ""))
    if cats and groups:
        return f"{cats}, {groups}"
    return cats or groups


def format_user_message(row: dict[str, Any]) -> str:
    """Convert one CSV row into the user message string.

    Args:
        row: Dictionary whose keys are raw CSV column names
             (org_uuid, name, short_description, Long description,
              category_list, category_groups_list, founded_date).

    Returns:
        Formatted text block matching the prompt's INPUT FORMAT section.
    """
    cid = _clean(row.get("org_uuid", ""))
    cname = _clean(row.get("name", ""))
    short = _clean(row.get("short_description", ""))
    long = _clean(row.get("Long description", ""))
    keywords = _merge_keywords(row)
    year = _extract_year(row.get("founded_date", ""))

    parts = [
        f"CompanyID: {cid}",
        f"CompanyName: {cname}",
        f"Short Description: {short}",
    ]

    if not long:
        parts.append("Long Description: [not available]")
    else:
        parts.append(f"Long Description: {long}")

    parts.append(f"Keywords: {keywords}")
    parts.append(f"YearFounded: {year}")

    message = "\n".join(parts)

    if len(message) > MAX_USER_MESSAGE_CHARS:
        message = message[:MAX_USER_MESSAGE_CHARS] + "\n[truncated]"

    return message


def build_custom_id(org_uuid: str) -> str:
    """Create a deterministic custom_id for batch result matching.

    The custom_id is the sole key for matching async batch results to
    their original inputs. Batch output order is not guaranteed.
    """
    sanitized = _clean(org_uuid).replace(" ", "-")
    if not sanitized:
        raise ValueError("Cannot build custom_id from blank org_uuid")
    return f"startup-{sanitized}"
