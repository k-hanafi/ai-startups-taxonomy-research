"""Stage 6: logprob -> binary-confidence extraction for Pass A responses.

Pass A of the classification runner answers {"ai_native": 0|1} in ~6 output
tokens with logprobs on. This module turns one raw Responses API payload into
a calibrated per-row confidence record: p(ai_native=1), top-1 probability,
margin, entropy, and valid_mass (the probability mass the model put on the
two legal answers before renormalization — the gate-Q3 evidence).

Hard-won payload facts this module is built around (verified against the
banked 2026-07-06 gpt-5.4-nano effort=none run):

- The decision rides ON a single '0' or '1' token, but its POSITION varies
  per response (the preceding fields tokenize differently row to row), and
  the first token is near-deterministic '{"'. So the value token is located
  STRUCTURALLY: parse the output text, find the char span of the ai_native
  value, map it to a byte offset, and walk the token byte spans to the entry
  covering it. Never index by token position.
- Tokens are byte sequences; concatenating every entry's `bytes` must equal
  the UTF-8 encoding of the output text. Reconstruction is verified before
  any extraction so a shape drift in the API fails loudly, not silently.
- top_logprobs lists contain grammar-masked entries at exactly -100.0 where
  the JSON schema forced the token (e.g. the 'ai' / '_native' key tokens
  carry 14 masked fillers). These are sentinels, not probabilities, and are
  excluded from the candidate pool.
- The chosen token is not guaranteed to appear in its own top_logprobs list;
  the entry's own {token, logprob} is merged into the pool first.

Deliberately dependency-free (stdlib only): nothing here may import single_pass_classifier.*
(OPENAI_API_KEY at import time) — extraction must run fully offline. This is
the prototype of the future single_pass_classifier/logprobs.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals import config as cfg

# Logprob value the API uses for grammar-masked top_logprobs fillers: tokens
# the JSON-schema decoder forbade, listed with a placeholder "probability".
# Exactly -100.0, per the banked responses (real logprobs near it, e.g.
# -26.5, do occur and must be kept).
MASKED_SENTINEL_LOGPROB: float = -100.0

# The JSON key whose value token carries the binary decision.
DECISION_KEY: str = "ai_native"


class LogprobExtractionError(ValueError):
    """A response payload does not match the shape this extractor relies on."""


class BinaryConfidenceUnavailable(LogprobExtractionError):
    """The decision token carries too little evidence to bound confidence.

    Raised instead of inventing a fake p=0/p=1 when the opposing digit is
    missing AND the unreported residual is too wide to bound it usefully.
    """


@dataclass(frozen=True)
class BinaryConfidence:
    """Per-row confidence metrics for the binary ai_native decision.

    Probabilities are renormalized over the {0, 1} candidates only;
    valid_mass records how much raw probability that pool held, so the
    renormalization is auditable after the fact.
    """

    ai_native: int              # the decoded verdict (from the chosen token)
    p_one: float                # renormalized P(ai_native = 1)
    top1_prob: float            # renormalized probability of the winning value
    margin: float               # |P(1) - P(0)| after renormalization
    entropy_bits: float         # binary entropy of {P(0), P(1)}, in bits
    valid_mass: float           # raw (pre-renormalization) mass on {0, 1}
    decision_token: str         # the token that carried the decision
    decision_token_index: int   # its position in the logprobs array
    censored: bool = False      # opposing digit absent; mass bounded, not read
    p_other_max: float = 0.0    # upper bound used for the absent digit's mass
    interval_width: float = 0.0  # confidence uncertainty implied by that bound

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Payload navigation + byte reconstruction
# ---------------------------------------------------------------------------

def output_text_content(response: dict[str, Any]) -> dict[str, Any]:
    """The output_text content block (text + logprobs) of a raw response dict."""
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                if not content.get("logprobs"):
                    raise LogprobExtractionError(
                        "output_text block has no logprobs — was the run made "
                        "with include=['message.output_text.logprobs']?"
                    )
                return content
    raise LogprobExtractionError("response has no message/output_text block")


def _entry_bytes(entry: dict[str, Any]) -> bytes:
    raw = entry.get("bytes")
    if raw is not None:
        return bytes(raw)
    return str(entry.get("token", "")).encode("utf-8")


def verify_reconstruction(text: str, logprobs: list[dict[str, Any]]) -> bytes:
    """Assert the token bytes concatenate to exactly the output text.

    Returns the reconstructed byte string (== text.encode()). Any mismatch
    means the char->byte->token span mapping below would be built on sand,
    so extraction refuses to continue.
    """
    reconstructed = b"".join(_entry_bytes(e) for e in logprobs)
    expected = text.encode("utf-8")
    if reconstructed != expected:
        raise LogprobExtractionError(
            f"token bytes ({len(reconstructed)}B) do not reconstruct the "
            f"output text ({len(expected)}B); refusing to map spans"
        )
    return reconstructed


# ---------------------------------------------------------------------------
# Structural location of the decision token
# ---------------------------------------------------------------------------

def locate_int_value_span(text: str, key: str) -> tuple[int, int]:
    """Char span [start, end) of the bare-integer value of a top-level key.

    A tiny string-aware JSON scanner: tracks nesting depth and skips string
    contents (including escapes), so a key-lookalike inside a string value
    (e.g. evidence text quoting '"ai_native":') can never match. Only
    top-level keys of the root object are considered.
    """
    i, n, depth = 0, len(text), 0
    while i < n:
        ch = text[i]
        if ch == '"':
            start = i + 1
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    break
                i += 1
            if i >= n:
                raise LogprobExtractionError("unterminated string in output text")
            raw_string = text[start:i]
            i += 1  # past the closing quote
            j = i
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] == ":" and depth == 1 and raw_string == key:
                j += 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                k = j
                while k < n and (text[k].isdigit() or text[k] == "-"):
                    k += 1
                if k == j:
                    raise LogprobExtractionError(
                        f"value of {key!r} is not a bare integer"
                    )
                return j, k
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    raise LogprobExtractionError(f"key {key!r} not found at the top level")


def find_decision_entry(
    text: str, logprobs: list[dict[str, Any]], key: str = DECISION_KEY
) -> tuple[int, dict[str, Any]]:
    """(index, entry) of the token whose byte span covers the key's value."""
    verify_reconstruction(text, logprobs)
    char_start, _ = locate_int_value_span(text, key)
    byte_start = len(text[:char_start].encode("utf-8"))

    pos = 0
    for index, entry in enumerate(logprobs):
        width = len(_entry_bytes(entry))
        if pos <= byte_start < pos + width:
            token_text = _entry_bytes(entry).decode("utf-8")
            if text[char_start] not in token_text:
                raise LogprobExtractionError(
                    f"token {token_text!r} at index {index} covers the "
                    f"{key!r} value span but does not contain the digit"
                )
            return index, entry
        pos += width
    raise LogprobExtractionError(
        f"no token covers byte offset {byte_start} of the {key!r} value"
    )


# ---------------------------------------------------------------------------
# Candidate pool + renormalization
# ---------------------------------------------------------------------------

def candidate_value(token: str) -> int | None:
    """Map a candidate token to the binary value it would decode to, if any.

    Whitespace is JSON-insignificant and trailing punctuation can fuse into
    the same token (e.g. '1,' or '0}'), so those forms count toward their
    digit. Everything else (pure whitespace, other digits, words) is not a
    legal value and contributes no mass.
    """
    core = token.strip().rstrip(",}").rstrip()
    if core == "0":
        return 0
    if core == "1":
        return 1
    return None


def binary_candidate_pool(entry: dict[str, Any]) -> dict[str, float]:
    """token -> logprob over the {0,1} candidates at the decision position.

    Merges the chosen token into its top_logprobs list (it may be absent),
    drops grammar-masked sentinels, and keeps only tokens that decode to a
    legal binary value.
    """
    pool: dict[str, float] = {}
    for candidate in entry.get("top_logprobs") or []:
        pool[str(candidate["token"])] = float(candidate["logprob"])
    pool.setdefault(str(entry["token"]), float(entry["logprob"]))

    return {
        token: logprob
        for token, logprob in pool.items()
        if candidate_value(token) is not None
        and logprob != MASKED_SENTINEL_LOGPROB
    }


def reported_probability_mass(entry: dict[str, Any]) -> float:
    """Total probability the API accounted for at this token position.

    Every unmasked candidate it listed, chosen token included, clamped to 1.
    Some models return logprobs a hair above 0 (observed +1.5e-5 on luna,
    i.e. p = 1.000015) from reduced-precision normalization, which would
    otherwise push the total past 1 and make the residual negative.
    """
    pool: dict[str, float] = {
        str(candidate["token"]): float(candidate["logprob"])
        for candidate in entry.get("top_logprobs") or []
    }
    pool.setdefault(str(entry["token"]), float(entry["logprob"]))
    total = sum(
        min(math.exp(logprob), 1.0)
        for logprob in pool.values()
        if logprob != MASKED_SENTINEL_LOGPROB
    )
    return min(total, 1.0)


def _binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------

def extract_binary_confidence(response: dict[str, Any]) -> BinaryConfidence:
    """One raw Responses API payload -> per-row binary confidence metrics."""
    content = output_text_content(response)
    text: str = content["text"]
    logprobs: list[dict[str, Any]] = content["logprobs"]

    index, entry = find_decision_entry(text, logprobs)

    chosen = candidate_value(str(entry["token"]))
    if chosen is None:
        raise LogprobExtractionError(
            f"decision token {entry['token']!r} does not decode to 0 or 1"
        )

    pool = binary_candidate_pool(entry)
    mass = {0: 0.0, 1: 0.0}
    for token, logprob in pool.items():
        mass[candidate_value(token)] += math.exp(logprob)  # type: ignore[index]

    if mass[chosen] <= 0.0:
        # Every legal candidate was grammar-masked, so the row carries no
        # usable evidence. Unavailable (skippable), not malformed: extract_run
        # re-raises plain LogprobExtractionError and would abort the whole run.
        raise BinaryConfidenceUnavailable(
            f"decision token {entry['token']!r} carries no unmasked "
            "probability mass"
        )

    censored, p_other_max, width = False, 0.0, 0.0
    other = 1 - chosen
    if mass[other] <= 0.0:
        # One-sided pool: the opposing digit was truncated, not observed at
        # zero. Its mass cannot exceed what the API left unaccounted for, so
        # bound it by the residual and take the midpoint of [0, bound]. This
        # assumes only that probabilities sum to at most 1. Renormalizing
        # against a hard zero instead would fabricate p = 1 exactly.
        censored = True
        p_other_max = max(0.0, 1.0 - reported_probability_mass(entry))
        p_chosen = min(mass[chosen], 1.0)
        width = p_other_max / (p_chosen + p_other_max)
        if width > cfg.MAX_CENSORED_INTERVAL_WIDTH:
            raise BinaryConfidenceUnavailable(
                f"binary confidence unavailable: opposing digit absent and the "
                f"unreported residual ({p_other_max:.4f}) only bounds it to "
                f"±{width:.4f}, wider than the "
                f"{cfg.MAX_CENSORED_INTERVAL_WIDTH} limit"
            )
        mass[other] = p_other_max / 2.0

    valid_mass = mass[0] + mass[1]
    p_one = mass[1] / valid_mass
    top1 = max(p_one, 1.0 - p_one)
    return BinaryConfidence(
        ai_native=chosen,
        p_one=p_one,
        top1_prob=top1,
        margin=abs(2.0 * p_one - 1.0),
        entropy_bits=_binary_entropy_bits(p_one),
        valid_mass=valid_mass,
        decision_token=str(entry["token"]),
        decision_token_index=index,
        censored=censored,
        p_other_max=p_other_max,
        interval_width=width,
    )


def extract_raw_file(path: Path) -> BinaryConfidence:
    """Extract from one banked raw response JSON file."""
    return extract_binary_confidence(json.loads(path.read_text(encoding="utf-8")))


def extract_run(raw_dir: Path) -> list[dict[str, Any]]:
    """Per-row metrics for every logprob-bearing raw response in a run dir.

    Two-pass runs bank Pass A as <custom_id>_a.json (preferred when present);
    single-pass effort=none runs bank <custom_id>.json. Returns one dict per
    extractable row, sorted by custom_id. Rows whose opposing digit is missing
    are omitted (confidence unavailable); callers that require full coverage
    must compare against the file count.
    """
    files = sorted(raw_dir.glob("*_a.json")) or sorted(raw_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        custom_id = path.stem.removesuffix("_a")
        try:
            rows.append({"custom_id": custom_id, **extract_raw_file(path).as_dict()})
        except BinaryConfidenceUnavailable:
            continue
        except LogprobExtractionError:
            raise
    return rows


# ---------------------------------------------------------------------------
# Scorer connector (pivot 6: confidence in the SAMPLED digit)
# ---------------------------------------------------------------------------

def chosen_confidence(row: dict[str, Any]) -> float:
    """Renormalized probability mass on the digit the model actually sampled.

    Pivot 6 (locked 2026-07-07): the sampled output is always the prediction,
    so calibration must measure how sure the model was about the digit it
    CHOSE, never the argmax. p_one is P(ai_native = 1), so a row that sampled
    the minority token (e.g. chose 1 at p_one = 0.28) gets confidence < 0.5.
    """
    return row["p_one"] if row["ai_native"] == 1 else 1.0 - row["p_one"]


def valid_mass_summary(
    rows: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    max_below_share: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-row valid_mass for the dashboard robustness panel."""
    cut = float(cfg.VALID_MASS_THRESHOLD if threshold is None else threshold)
    allow = float(
        cfg.VALID_MASS_MAX_BELOW_SHARE
        if max_below_share is None
        else max_below_share
    )
    masses = [float(r["valid_mass"]) for r in rows if r.get("valid_mass") is not None]
    if not masses:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "mean": None,
            "threshold": cut,
            "max_below_share": allow,
            "n_below_threshold": 0,
            "below_share": 0.0,
        }
    ordered = sorted(masses)
    n = len(ordered)
    n_below = sum(1 for m in ordered if m < cut)
    return {
        "n": n,
        "min": ordered[0],
        "p50": ordered[n // 2],
        "mean": sum(ordered) / n,
        "threshold": cut,
        "max_below_share": allow,
        "n_below_threshold": n_below,
        "below_share": n_below / n,
    }


def run_confidence(raw_dir: Path) -> dict[str, float]:
    """custom_id -> chosen-digit confidence for every raw response in a run.

    The exact shape the scorer's external-confidence seam accepts
    (resolve_confidence matches custom_id keys). Raises when the run has no
    raw responses at all, so an explicit --confidence-from-raw request never
    silently degrades into a calibration-free score.
    """
    rows = extract_confidence_rows(raw_dir)
    return {row["custom_id"]: chosen_confidence(row) for row in rows}


def extract_confidence_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Like ``extract_run``, but refuses when nothing is extractable.

    Shared by score ``--confidence-from-raw`` so confidence and valid_mass
    come from one pass over the banked raw responses.
    """
    raw_files = sorted(raw_dir.glob("*_a.json")) or sorted(raw_dir.glob("*.json"))
    rows = extract_run(raw_dir)
    if rows:
        return rows
    if not raw_dir.exists() or not raw_files:
        raise LogprobExtractionError(
            f"no raw response files under {raw_dir}; this run cannot supply "
            "logprob confidence (raw/ is git-ignored and machine-local)"
        )
    raise LogprobExtractionError(
        f"raw responses under {raw_dir} exist ({len(raw_files)} file(s)) "
        "but none yielded binary confidence: every decision token lacked "
        "the opposing digit and left too wide a residual to bound it "
        f"(limit {cfg.MAX_CENSORED_INTERVAL_WIDTH}). Raising "
        "PASS_A_TOP_LOGPROBS does not help models that truncate "
        "candidates at a probability floor."
    )
