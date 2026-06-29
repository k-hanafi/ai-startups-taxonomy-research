#!/usr/bin/env python3
"""Smoke-test logprobs + strict json_schema on the production request shape.

Runs two probes against POST /v1/responses:
  1. Minimal schema (community repro from OpenAI forum #1371927)
  2. Full ClassificationResult schema (exact production classifier shape)

Usage:
  python scripts/smoke_test_logprobs.py
  python scripts/smoke_test_logprobs.py --model gpt-5.4-nano --top-logprobs 5
  python scripts/smoke_test_logprobs.py --company-name Stripe

Requires OPENAI_API_KEY (or keys/openai.env).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.builder import (  # noqa: E402
    _openai_strict_schema,
    load_system_prompt,
    responses_text_format_json_schema,
)
from src.config import DEFAULT_MODEL, MAX_OUTPUT_TOKENS, PROMPT_CACHE_KEY  # noqa: E402
from src.formatter import format_user_message  # noqa: E402
from src.submitter import get_client  # noqa: E402


def _minimal_schema() -> dict:
    return {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }


def _logprob_count(response) -> tuple[int, int]:
    """Return (num_logprob_entries, num_with_top_logprobs)."""
    total = 0
    with_top = 0
    for item in response.output or []:
        content = getattr(item, "content", None) or []
        for block in content:
            logprobs = getattr(block, "logprobs", None) or []
            total += len(logprobs)
            for entry in logprobs:
                top = getattr(entry, "top_logprobs", None) or []
                if top:
                    with_top += 1
    return total, with_top


def _run_probe(
    *,
    label: str,
    model: str,
    instructions: str | None,
    user_input: str,
    schema: dict,
    top_logprobs: int,
    reasoning_none: bool,
) -> dict:
    client = get_client()
    kwargs: dict = {
        "model": model,
        "input": user_input,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": top_logprobs,
        "text": responses_text_format_json_schema(schema),
    }
    if instructions:
        kwargs["instructions"] = instructions
        kwargs["prompt_cache_key"] = PROMPT_CACHE_KEY
    if reasoning_none:
        kwargs["reasoning"] = {"effort": "none"}

    response = client.responses.create(**kwargs)
    count, with_top = _logprob_count(response)
    return {
        "label": label,
        "model": model,
        "top_logprobs": top_logprobs,
        "reasoning_none": reasoning_none,
        "response_id": response.id,
        "output_text_preview": (response.output_text or "")[:120],
        "logprob_entries": count,
        "entries_with_top_logprobs": with_top,
        "verdict": "PASS" if count > 0 else "FAIL (empty logprobs[])",
    }


def _resolve_user_message(args: argparse.Namespace) -> tuple[str, str | None]:
    if not args.company_id and not args.company_name:
        return "Return JSON with x set to hello.", None

    import pandas as pd

    from src.tavily_crawl import DEFAULT_CLASSIFIER_INPUT_CSV

    data_path = Path(args.data) if args.data else DEFAULT_CLASSIFIER_INPUT_CSV
    df = pd.read_csv(data_path)
    if args.company_id:
        match = df[df["org_uuid"] == args.company_id]
    else:
        match = df[df["name"].str.contains(args.company_name, case=False, na=False)]
    if match.empty:
        raise SystemExit("No matching company found in classifier input CSV.")
    row = match.iloc[0].to_dict()
    return format_user_message(row), load_system_prompt()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--company-id", help="org_uuid from classifier input CSV")
    parser.add_argument("--company-name", help="Substring match on company name")
    parser.add_argument(
        "--data",
        help="Classifier input CSV (default: outputs/tavilycrawl/processed/classifier_input.csv)",
    )
    parser.add_argument(
        "--skip-minimal",
        action="store_true",
        help="Only run the full ClassificationResult schema probe",
    )
    args = parser.parse_args()

    user_msg, system_prompt = _resolve_user_message(args)
    full_schema = _openai_strict_schema()

    probes: list[dict] = []
    if not args.skip_minimal:
        probes.append(
            _run_probe(
                label="minimal_schema",
                model=args.model,
                instructions=None,
                user_input="Hello!",
                schema=_minimal_schema(),
                top_logprobs=args.top_logprobs,
                reasoning_none=True,
            )
        )

    probes.append(
        _run_probe(
            label="classification_result_schema",
            model=args.model,
            instructions=system_prompt,
            user_input=user_msg,
            schema=full_schema,
            top_logprobs=args.top_logprobs,
            reasoning_none=True,
        )
    )

    print(json.dumps({"probes": probes}, indent=2))

    if any(p["logprob_entries"] == 0 for p in probes):
        print(
            "\nResult: logprobs appear blocked for at least one probe. "
            "Under strict json_schema + GPT-5+, confidence must use a non-logprob method.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nResult: logprobs populated for all probes. Safe to prototype logprob-based confidence.")


if __name__ == "__main__":
    main()
