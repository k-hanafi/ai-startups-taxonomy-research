"""Numbered archive of eval dashboard instances.

`eval_dashboard.html` is a working file: the next build overwrites it. A paid
sweep is worth keeping, so every dashboard built from real scored runs is also
written as a numbered page (`eval_instance_01.html`, `02`, ...) under
`eval_instances/`, alongside an index that lists them all. Mock builds are
archived only when explicitly asked for, otherwise development rebuilds would
bury the real runs.

An instance is identified by the runs behind it, not by when the page was
rendered. Rebuilding the same runs after a styling fix overwrites that
instance instead of allocating a new number. Runs that never recorded a start
time cannot be identified, so those always take a fresh number rather than
risk overwriting a different sweep.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.paths import EVAL_INSTANCES_DIR

REGISTRY_NAME = "instances.json"
INDEX_NAME = "index.html"
_FILENAME_RE = re.compile(r"^eval_instance_(\d+)\.html$")


# --------------------------------------------------------------------------
# Run-instance text, shared by the suite header card and the index rows
# --------------------------------------------------------------------------

def local_time(iso: str | None) -> datetime.datetime | None:
    """Recorded UTC instant in the reading machine's local timezone."""
    if not iso:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone()


def format_clock(moment: datetime.datetime) -> str:
    """2:47 PM. Built by hand: %-I is not portable across platforms."""
    hour = moment.hour % 12 or 12
    meridiem = "AM" if moment.hour < 12 else "PM"
    return f"{hour}:{moment.minute:02d} {meridiem}"


def format_day(moment: datetime.datetime) -> str:
    return f"{moment.strftime('%b')} {moment.day}, {moment.year}"


def format_run_headline(run: dict[str, Any]) -> str:
    """Which eval instance this is, in one line."""
    n_runs = int(run.get("n_runs") or 0)
    lead = "Eval run" if n_runs == 1 else f"{n_runs} eval runs"
    first = local_time(run.get("started_first"))
    last = local_time(run.get("started_last"))
    if first is None or last is None:
        return f"{lead}, start time not recorded"
    if (last - first).total_seconds() < 60:
        return f"{lead}, {format_day(first)} at {format_clock(first)}"
    if first.date() == last.date():
        return f"{lead}, {format_day(first)}, {format_clock(first)} to {format_clock(last)}"
    return (
        f"{lead}, {format_day(first)} {format_clock(first)} "
        f"to {format_day(last)} {format_clock(last)}"
    )


def run_meta_bits(run: dict[str, Any]) -> list[str]:
    """Secondary provenance, plain text: models, golden size, commit."""
    bits: list[str] = []
    groups = [g for g in (run.get("model_groups") or []) if g]
    if groups:
        bits.append(", ".join(groups))
    n_gold = run.get("n_gold")
    if n_gold:
        bits.append(f"{int(n_gold)} golden companies")
    commit = run.get("git_commit")
    if commit:
        bits.append(f"commit {commit}")
    return bits


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def instance_filename(number: int) -> str:
    return f"eval_instance_{number:02d}.html"


def load_registry(directory: Path = EVAL_INSTANCES_DIR) -> list[dict[str, Any]]:
    """Archived instances, oldest first. Missing or damaged registry reads empty."""
    try:
        data = json.loads((directory / REGISTRY_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("instances") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    clean = [e for e in entries if isinstance(e, dict) and isinstance(e.get("n"), int)]
    return sorted(clean, key=lambda e: e["n"])


def _write_registry(directory: Path, entries: list[dict[str, Any]]) -> None:
    payload = {"instances": sorted(entries, key=lambda e: e["n"])}
    (directory / REGISTRY_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def next_instance_number(directory: Path, entries: list[dict[str, Any]]) -> int:
    """One past the highest number in the registry or on disk.

    Filenames are consulted too so a lost registry cannot hand out a number
    that would overwrite a page still sitting in the folder.
    """
    used = [int(e["n"]) for e in entries]
    if directory.exists():
        used += [
            int(m.group(1))
            for m in (_FILENAME_RE.match(p.name) for p in directory.iterdir())
            if m
        ]
    return max(used, default=0) + 1


def _identity(metrics: dict[str, Any]) -> str | None:
    """Stable fingerprint of the runs behind a page, None when unidentifiable.

    Synthetic/mock builds key on their source path so ``--save-instance`` can
    replace the preview instead of minting a new number every rebuild. Real
    runs without a recorded start time stay unidentifiable (fresh number)
    rather than risk overwriting a different sweep.
    """
    if metrics.get("synthetic") or (metrics.get("run_instance") or {}).get("synthetic"):
        source = str(metrics.get("source") or "fixture")
        parts = [
            "synthetic",
            source,
            "|".join(sorted(str(c) for c in (metrics.get("config_ids") or []))),
        ]
        return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]

    run = metrics.get("run_instance") or {}
    first, last = run.get("started_first"), run.get("started_last")
    if not first or not last:
        return None
    parts = [
        first,
        last,
        str(run.get("n_runs") or 0),
        "|".join(sorted(str(c) for c in (metrics.get("config_ids") or []))),
    ]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ArchivedInstance:
    """Where the page landed and whether it replaced an earlier build of it."""

    path: Path
    index_path: Path
    number: int
    replaced: bool


def archive_instance(
    page_html: str,
    metrics: dict[str, Any],
    *,
    directory: Path = EVAL_INSTANCES_DIR,
    now: datetime.datetime | None = None,
) -> ArchivedInstance:
    """Write the page as a numbered instance and refresh the index."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
    entries = load_registry(directory)
    identity = _identity(metrics)

    existing = next(
        (e for e in entries if identity and e.get("identity") == identity), None
    )
    if existing is None:
        entry: dict[str, Any] = {
            "n": next_instance_number(directory, entries),
            "archived_utc": stamp,
        }
        entries.append(entry)
    else:
        entry = existing
        entry["rebuilt_utc"] = stamp

    entry.update(
        {
            "file": instance_filename(int(entry["n"])),
            "identity": identity,
            "synthetic": bool(metrics.get("synthetic")),
            "n_configs": int(metrics.get("n_configs") or 0),
            "source": str(metrics.get("source") or ""),
            "run": metrics.get("run_instance") or {},
        }
    )

    path = directory / str(entry["file"])
    path.write_text(page_html, encoding="utf-8")
    _write_registry(directory, entries)
    index_path = directory / INDEX_NAME
    index_path.write_text(render_index(entries), encoding="utf-8")
    return ArchivedInstance(
        path=path,
        index_path=index_path,
        number=int(entry["n"]),
        replaced=existing is not None,
    )


# --------------------------------------------------------------------------
# Index page
# --------------------------------------------------------------------------

INDEX_STYLE = """
:root {
  --bg: #0a0a0a;
  --surface: #111111;
  --border: #2a2a2a;
  --text: #e8e8ea;
  --text2: #a8abb0;
  --muted: #7a7e85;
  --accent: #5b8fc4;
  --sans: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.appbar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  border-bottom: 1px solid var(--border);
  padding: 18px 36px;
}
.brand { font-size: 17px; font-weight: 600; letter-spacing: -0.015em; }
.brand small { margin-left: 12px; font-size: 12px; font-weight: 400; color: var(--muted); }
.appbar-meta { font-size: 12px; font-family: var(--mono); color: var(--muted); }
main { padding: 28px 36px 48px; max-width: 1100px; }
.lede { color: var(--text2); max-width: 70ch; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); }
th, td { text-align: left; padding: 11px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
tr:last-child td { border-bottom: none; }
td.num { font-family: var(--mono); color: var(--text2); white-space: nowrap; }
td.meta { color: var(--text2); }
.tag {
  display: inline-block;
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--border);
  padding: 1px 6px;
  margin-left: 8px;
}
.empty { border: 1px solid var(--border); background: var(--surface); padding: 20px; color: var(--text2); }
footer { margin-top: 28px; font-size: 12px; color: var(--muted); }
code { font-family: var(--mono); font-size: 0.92em; }
"""


def _index_row(entry: dict[str, Any]) -> str:
    run = entry.get("run") or {}
    headline = format_run_headline(run) if run else "Run details not recorded"
    if entry.get("synthetic"):
        headline = "Synthetic data (mock matrix)"
    meta = " &middot; ".join(html.escape(b) for b in run_meta_bits(run))
    archived = local_time(entry.get("rebuilt_utc") or entry.get("archived_utc"))
    when = f"{format_day(archived)}, {format_clock(archived)}" if archived else "unknown"
    rebuilt = ' <span class="tag">rebuilt</span>' if entry.get("rebuilt_utc") else ""
    file_name = html.escape(str(entry.get("file") or ""))
    return f"""
    <tr>
      <td class="num">{int(entry['n']):02d}</td>
      <td>
        <a href="{file_name}">{html.escape(headline)}</a>{rebuilt}
        {f'<div class="meta">{meta}</div>' if meta else ''}
      </td>
      <td class="num">{int(entry.get('n_configs') or 0)}</td>
      <td class="meta">{html.escape(when)}</td>
      <td class="num">{file_name}</td>
    </tr>"""


def render_index(entries: list[dict[str, Any]]) -> str:
    """Newest-first table of archived instances."""
    rows = "".join(_index_row(e) for e in sorted(entries, key=lambda e: -int(e["n"])))
    body = (
        f"""
  <table>
    <thead>
      <tr><th>#</th><th>Eval instance</th><th>Configs</th><th>Archived</th><th>File</th></tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>"""
        if entries
        else """
  <div class="empty">No instances archived yet. Building the dashboard from real
  scored runs archives one automatically.</div>"""
    )
    today = datetime.date.today().isoformat()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval instances</title>
<style>{INDEX_STYLE}</style>
</head>
<body>
<header class="appbar">
  <div class="brand">Classifier Eval Suite<small>instance archive</small></div>
  <div class="appbar-meta">{len(entries)} archived</div>
</header>
<main>
  <p class="lede">Each row is one saved build of the eval suite, kept so a scored
  run stays viewable after later builds. The working page
  (<a href="../eval_dashboard.html">eval_dashboard.html</a>) is overwritten every
  time the dashboard is rebuilt.</p>{body}
  <footer>Index rewritten {today}. Times shown in this machine's local timezone.</footer>
</main>
</body>
</html>
"""
