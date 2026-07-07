"""Capture anonymized logprob fixtures from a banked effort=none run.

The banked raw responses (evals/runs/*/raw/, git-ignored) hold real logprob
arrays but also company names, descriptions, and scraped evidence text —
none of which may be committed (public repo). This script distills each
selected response into a Pass-A-shaped payload built ONLY from the
decision-relevant token entries, all copied verbatim from the API:

    '{"'  'ai'  '_native'  '":'  <digit>  '}'

The composed text ('{"ai_native":<digit>}') is exactly what Pass A's
BinaryResult schema produces, and every entry keeps its real bytes, logprob,
and full top_logprobs list — including the -100.0 grammar-mask sentinels on
the key tokens and the true {0,1} candidate spread on the decision token.
No name, uuid, or evidence text survives, by construction (gate Q5), while
the tokenization the extractor depends on stays real (gate Q2).

Usage (offline, no keys):
    python -m evals.tests.capture_fixtures

Rewrites evals/tests/fixtures/*.json from FIXTURE_SOURCES. Committed so the
fixture provenance is reproducible while the source run stays local-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.logprob_extract import (
    extract_binary_confidence,
    locate_int_value_span,
    output_text_content,
    verify_reconstruction,
)
from evals.paths import RUNS_DIR

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SOURCE_RUN = "2026-07-06_gpt-5.4-nano_none_r1"

# custom_id -> anonymous fixture name. Chosen from the banked run to span the
# confidence range: a saturated 0, a saturated 1, a near-coin-flip (the model
# split 0.50/0.50 and sampled 1), and a row where the sampled token holds the
# MINORITY of the renormalized mass (chosen '1' at p_one~0.28) — the case
# where argmax-of-logprobs and the emitted verdict disagree.
FIXTURE_SOURCES: dict[str, str] = {
    "startup-e0eddfa2-b0aa-479d-8cdc-0d104d374f09": "confident_zero",
    "startup-d82fc661-0b0e-4592-b201-d7aea735fb84": "confident_one",
    "startup-50db1254-1c56-4394-809b-e730075ed708": "coin_flip",
    "startup-dc08d6a3-f877-49c8-a75b-845ea1fb3432": "chose_minority",
}


def _decision_span_entries(
    text: str, logprobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Entries exactly tiling the bytes of `ai_native":<digits>` in the text.

    The key's opening quote belongs to the preceding token (typically '","'
    mid-object or '{"' at the start), so the tiled span starts at the key
    name itself. Raises if the tokenization fuses across the span boundary,
    in which case pick a different source row.
    """
    value_start, value_end = locate_int_value_span(text, "ai_native")
    key_start = text.rindex('"ai_native"', 0, value_start) + 1  # past the quote
    span_start = len(text[:key_start].encode("utf-8"))
    span_end = len(text[:value_end].encode("utf-8"))

    entries: list[dict[str, Any]] = []
    pos = 0
    for entry in logprobs:
        width = len(bytes(entry["bytes"]))
        if pos >= span_start and pos + width <= span_end:
            entries.append(entry)
        elif pos < span_end and pos + width > span_start:
            raise AssertionError(
                f"token {entry['token']!r} straddles the decision span; "
                "choose a different source row"
            )
        pos += width
    return entries


def compose_fixture(response: dict[str, Any], name: str) -> dict[str, Any]:
    """A Pass-A-shaped payload from only the decision-relevant real entries."""
    content = output_text_content(response)
    text: str = content["text"]
    logprobs: list[dict[str, Any]] = content["logprobs"]
    verify_reconstruction(text, logprobs)

    opener, closer = logprobs[0], logprobs[-1]
    if opener["token"] != '{"' or closer["token"] != "}":
        raise AssertionError(
            f"unexpected frame tokens {opener['token']!r}...{closer['token']!r}; "
            "choose a different source row"
        )

    entries = [opener, *_decision_span_entries(text, logprobs), closer]
    fixture_text = "".join(bytes(e["bytes"]).decode("utf-8") for e in entries)

    fixture = {
        "fixture_name": name,
        "source_run": SOURCE_RUN,
        "anonymization": (
            "composed from the '{\"', ai_native key/value, and '}' token "
            "entries of one real response; no company or evidence text"
        ),
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": fixture_text,
                        "logprobs": entries,
                    }
                ],
            }
        ],
    }

    # The fixture must carry the same decision metrics as its source: the
    # extractor only reads the decision entry, so composition is lossless.
    original = extract_binary_confidence(response)
    composed = extract_binary_confidence(fixture)
    if composed != original.__class__(
        **{**original.as_dict(), "decision_token_index": composed.decision_token_index}
    ):
        raise AssertionError(f"fixture {name} diverged from its source metrics")

    fixture["expected"] = composed.as_dict()
    return fixture


def main() -> None:
    raw_dir = RUNS_DIR / SOURCE_RUN / "raw"
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for custom_id, name in FIXTURE_SOURCES.items():
        response = json.loads(
            (raw_dir / f"{custom_id}.json").read_text(encoding="utf-8")
        )
        fixture = compose_fixture(response, name)
        out = FIXTURES_DIR / f"{name}.json"
        out.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
        print(f"wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
