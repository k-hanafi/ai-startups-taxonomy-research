"""Stage 2: gold-labeling workspace + human review page.

Flow:
1. export_labeling_workspace() dumps each golden-set company as the exact
   formatted user message the classifier models receive (via the production
   src.formatter), with no production prediction attached, so the drafter
   judges the same evidence the models see and is not anchored by nano's
   answer.
2. The drafter (Fable, in-session) fills draft_* columns via apply_drafts().
3. render_review_page() writes a local HTML page (evidence beside draft
   label + rationale) for the human verdict pass. The page embeds scraped
   website text, so it is git-ignored; only the CSV labels are committed.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

from src.formatter import format_user_message

from evals.paths import (
    CLASSIFIER_INPUT_CSV,
    GOLDEN_SET_CSV,
    LABELING_WORKSPACE_DIR,
    REVIEW_PAGE_HTML,
)

logger = logging.getLogger(__name__)

DRAFT_COLUMNS = [
    "draft_ai_native",
    "draft_subclass",
    "draft_rad",
    "draft_rationale",
    "ambiguity_flag",
]

VALID_SUBCLASSES = {"1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C"}
VALID_RAD = {"RAD-H", "RAD-M", "RAD-L", "RAD-NA"}


def _s(value: object) -> str:
    """String-ify a merged cell, mapping pandas NaN/None to empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _load_golden_with_evidence() -> pd.DataFrame:
    golden = pd.read_csv(GOLDEN_SET_CSV, dtype=str, keep_default_na=False)
    classifier_input = pd.read_csv(CLASSIFIER_INPUT_CSV)
    merged = golden.merge(
        classifier_input, on="org_uuid", how="left", suffixes=("", "_ci"), indicator=True
    )
    if len(merged) != len(golden):
        raise AssertionError("golden_set rows did not join 1:1 with classifier_input")
    unmatched = merged.loc[merged["_merge"] != "both", "org_uuid"].tolist()
    if unmatched:
        raise AssertionError(
            f"{len(unmatched)} golden rows missing from classifier_input: {unmatched[:5]}"
        )
    return merged.drop(columns="_merge")


def export_labeling_workspace() -> list[Path]:
    """Write one prompt-view text file per golden company (git-ignored)."""
    merged = _load_golden_with_evidence()
    LABELING_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for seq, row in enumerate(merged.to_dict(orient="records"), start=1):
        row["name"] = _s(row.get("name_ci")) or _s(row.get("name"))
        body = format_user_message(row)
        path = LABELING_WORKSPACE_DIR / f"{seq:03d}_{row['org_uuid'][:8]}.txt"
        path.write_text(body, encoding="utf-8")
        paths.append(path)

    logger.info("Wrote %d labeling files to %s", len(paths), LABELING_WORKSPACE_DIR)
    return paths


def apply_drafts(drafts_json: Path | str) -> int:
    """Merge a batch of draft labels into golden_set.csv.

    The JSON maps org_uuid -> {draft_ai_native, draft_subclass, draft_rad,
    draft_rationale, ambiguity_flag}. Values are validated against the
    taxonomy before anything is written.
    """
    drafts: dict[str, dict] = json.loads(Path(drafts_json).read_text(encoding="utf-8"))
    golden = pd.read_csv(GOLDEN_SET_CSV, dtype=str, keep_default_na=False)
    known = set(golden["org_uuid"])

    for uuid, fields in drafts.items():
        if uuid not in known:
            raise KeyError(f"org_uuid {uuid!r} is not in the golden set")
        subclass = fields["draft_subclass"]
        rad = fields["draft_rad"]
        native = str(fields["draft_ai_native"])
        if subclass not in VALID_SUBCLASSES:
            raise ValueError(f"{uuid}: invalid subclass {subclass!r}")
        if rad not in VALID_RAD:
            raise ValueError(f"{uuid}: invalid rad {rad!r}")
        if native not in {"0", "1"}:
            raise ValueError(f"{uuid}: draft_ai_native must be 0 or 1")
        if (native == "1") != subclass.startswith("1"):
            raise ValueError(f"{uuid}: ai_native={native} contradicts subclass {subclass}")
        if (native == "0") != (rad == "RAD-NA"):
            raise ValueError(f"{uuid}: rad {rad} contradicts ai_native={native}")
        if not str(fields["draft_rationale"]).strip():
            raise ValueError(f"{uuid}: empty draft_rationale")

        mask = golden["org_uuid"] == uuid
        for col in DRAFT_COLUMNS:
            golden.loc[mask, col] = str(fields[col])

    # Atomic write: never leave the committed golden set truncated mid-write.
    tmp = GOLDEN_SET_CSV.with_suffix(".csv.tmp")
    golden.to_csv(tmp, index=False)
    tmp.replace(GOLDEN_SET_CSV)
    done = (golden["draft_subclass"] != "").sum()
    logger.info("Applied %d drafts (%d/%d labeled)", len(drafts), done, len(golden))
    return int(done)


def render_review_page() -> Path:
    """Write the human-review HTML page (git-ignored: embeds evidence text)."""
    merged = _load_golden_with_evidence()

    cards: list[str] = []
    for seq, row in enumerate(merged.to_dict(orient="records"), start=1):
        flag = _s(row.get("ambiguity_flag")).strip()
        flag_html = (
            f'<span class="flag">AMBIGUOUS: {html.escape(flag)}</span>' if flag else ""
        )
        cards.append(f"""
<details class="card" id="{row['org_uuid']}">
  <summary>
    <span class="seq">{seq:03d}</span>
    <strong>{html.escape(_s(row.get('name_ci')) or _s(row.get('name')))}</strong>
    <span class="label">{html.escape(_s(row.get('draft_subclass')) or '?')}
      / {html.escape(_s(row.get('draft_rad')) or '?')}</span>
    {flag_html}
  </summary>
  <p class="meta">org_uuid: <code>{row['org_uuid']}</code>
     | predicted (nano): {html.escape(_s(row.get('predicted_subclass')))}
     | evidence: {html.escape(_s(row.get('evidence_chars')))} chars
     ({html.escape(_s(row.get('evidence_tercile')))})</p>
  <p class="rationale"><b>Draft rationale:</b>
     {html.escape(_s(row.get('draft_rationale')))}</p>
  <h4>Descriptions</h4>
  <p>{html.escape(_s(row.get('short_description')))}</p>
  <p>{html.escape(_s(row.get('Long description')))}</p>
  <h4>Website evidence</h4>
  <pre>{html.escape(_s(row.get('website_evidence')))}</pre>
</details>""")

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Golden-set review</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; margin: .6rem 0; padding: .4rem .8rem; }}
  summary {{ cursor: pointer; display: flex; gap: .7rem; align-items: baseline; }}
  .seq {{ color: #999; font-family: monospace; }}
  .label {{ background: #eef; border-radius: 4px; padding: 0 .4rem; font-family: monospace; }}
  .flag {{ background: #fde68a; border-radius: 4px; padding: 0 .4rem; font-size: .85em; }}
  .meta {{ color: #666; font-size: .85em; }}
  pre {{ white-space: pre-wrap; background: #fafafa; padding: .8rem; border-radius: 6px;
         max-height: 30rem; overflow-y: auto; }}
</style></head><body>
<h1>Golden-set review ({len(merged)} companies)</h1>
<p>Review each draft label. Record your verdict in <code>evals/golden/golden_set.csv</code>:
set <code>gold_verdict</code> to <code>accept</code> or <code>override</code>, and fill
<code>gold_ai_native / gold_subclass / gold_rad</code> with the final values.
This page is local-only (git-ignored) because it embeds scraped website text.</p>
{''.join(cards)}
</body></html>"""

    REVIEW_PAGE_HTML.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PAGE_HTML.write_text(page, encoding="utf-8")
    logger.info("Wrote review page: %s", REVIEW_PAGE_HTML)
    return REVIEW_PAGE_HTML
