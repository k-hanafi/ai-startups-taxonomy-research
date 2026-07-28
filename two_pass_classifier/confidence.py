"""Extract sampled-token Pass A confidence from Responses API logprobs.

The opposing digit can be omitted by mini and luna even when more top
logprobs are requested. In that censored case, this module bounds the missing
mass by the unreported residual and uses the validated midpoint estimate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .config import MAX_CENSORED_INTERVAL_WIDTH

MASKED_SENTINEL_LOGPROB = -100.0
DECISION_KEY = "ai_native"


class LogprobExtractionError(ValueError):
    """A response does not match the token shape required for extraction."""


class BinaryConfidenceUnavailable(LogprobExtractionError):
    """The response cannot support a sufficiently narrow confidence estimate."""


@dataclass(frozen=True)
class BinaryConfidence:
    ai_native: int
    p_one: float
    sampled_probability: float
    valid_mass: float
    decision_token: str
    decision_token_index: int
    censored: bool
    p_other_max: float
    interval_width: float

    @property
    def top1_prob(self) -> float:
        return max(self.p_one, 1.0 - self.p_one)

    @property
    def margin(self) -> float:
        return abs(2.0 * self.p_one - 1.0)

    @property
    def entropy_bits(self) -> float:
        if self.p_one <= 0.0 or self.p_one >= 1.0:
            return 0.0
        return -(
            self.p_one * math.log2(self.p_one)
            + (1.0 - self.p_one) * math.log2(1.0 - self.p_one)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "top1_prob": self.top1_prob,
            "margin": self.margin,
            "entropy_bits": self.entropy_bits,
        }


def extract_binary_confidence(response: dict[str, Any]) -> BinaryConfidence:
    """Extract binary probabilities and confidence in the sampled digit."""
    content = _output_text_content(response)
    text = str(content["text"])
    logprobs = content["logprobs"]
    index, entry = _find_decision_entry(text, logprobs)

    chosen = candidate_value(str(entry["token"]))
    if chosen is None:
        raise LogprobExtractionError(
            f"decision token {entry['token']!r} does not decode to 0 or 1"
        )

    mass = {0: 0.0, 1: 0.0}
    for token, logprob in _binary_candidate_pool(entry).items():
        value = candidate_value(token)
        if value is not None:
            mass[value] += math.exp(logprob)

    if mass[chosen] <= 0.0:
        raise BinaryConfidenceUnavailable(
            f"decision token {entry['token']!r} carries no usable probability mass"
        )

    censored = False
    p_other_max = 0.0
    interval_width = 0.0
    other = 1 - chosen
    if mass[other] <= 0.0:
        censored = True
        p_other_max = max(0.0, 1.0 - _reported_probability_mass(entry))
        p_chosen = min(mass[chosen], 1.0)
        interval_width = p_other_max / (p_chosen + p_other_max)
        if interval_width > MAX_CENSORED_INTERVAL_WIDTH:
            raise BinaryConfidenceUnavailable(
                "opposing digit is absent and its residual bound is too wide: "
                f"{interval_width:.4f} > {MAX_CENSORED_INTERVAL_WIDTH:.4f}"
            )
        mass[other] = p_other_max / 2.0

    valid_mass = mass[0] + mass[1]
    p_one = mass[1] / valid_mass
    sampled_probability = p_one if chosen == 1 else 1.0 - p_one
    return BinaryConfidence(
        ai_native=chosen,
        p_one=p_one,
        sampled_probability=sampled_probability,
        valid_mass=valid_mass,
        decision_token=str(entry["token"]),
        decision_token_index=index,
        censored=censored,
        p_other_max=p_other_max,
        interval_width=interval_width,
    )


def ai_native_confidence(response: dict[str, Any]) -> float | None:
    """Return sampled-token confidence, or None when its bound is unavailable."""
    try:
        return extract_binary_confidence(response).sampled_probability
    except BinaryConfidenceUnavailable:
        return None


def candidate_value(token: str) -> int | None:
    """Map whitespace or punctuation-fused token forms to binary values."""
    core = token.strip().rstrip(",}").rstrip()
    if core == "0":
        return 0
    if core == "1":
        return 1
    return None


def _output_text_content(response: dict[str, Any]) -> dict[str, Any]:
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                if not content.get("logprobs"):
                    raise LogprobExtractionError(
                        "output_text has no logprobs; Pass A must request them"
                    )
                return content
    raise LogprobExtractionError("response has no message/output_text content")


def _entry_bytes(entry: dict[str, Any]) -> bytes:
    raw = entry.get("bytes")
    if raw is not None:
        return bytes(raw)
    return str(entry.get("token", "")).encode("utf-8")


def _verify_reconstruction(
    text: str, logprobs: list[dict[str, Any]]
) -> None:
    reconstructed = b"".join(_entry_bytes(entry) for entry in logprobs)
    if reconstructed != text.encode("utf-8"):
        raise LogprobExtractionError(
            "token bytes do not reconstruct output text; refusing span mapping"
        )


def _locate_int_value_span(text: str, key: str) -> tuple[int, int]:
    index = 0
    depth = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            start = index + 1
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == '"':
                    break
                index += 1
            if index >= len(text):
                raise LogprobExtractionError("unterminated string in output text")
            raw_string = text[start:index]
            index += 1
            cursor = index
            while cursor < len(text) and text[cursor] in " \t\r\n":
                cursor += 1
            if (
                cursor < len(text)
                and text[cursor] == ":"
                and depth == 1
                and raw_string == key
            ):
                cursor += 1
                while cursor < len(text) and text[cursor] in " \t\r\n":
                    cursor += 1
                end = cursor
                while end < len(text) and (
                    text[end].isdigit() or text[end] == "-"
                ):
                    end += 1
                if end == cursor:
                    raise LogprobExtractionError(
                        f"value of {key!r} is not a bare integer"
                    )
                return cursor, end
            continue
        if character in "{[":
            depth += 1
        elif character in "}]":
            depth -= 1
        index += 1
    raise LogprobExtractionError(f"key {key!r} not found at the top level")


def _find_decision_entry(
    text: str,
    logprobs: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    _verify_reconstruction(text, logprobs)
    char_start, _ = _locate_int_value_span(text, DECISION_KEY)
    byte_start = len(text[:char_start].encode("utf-8"))

    position = 0
    for index, entry in enumerate(logprobs):
        width = len(_entry_bytes(entry))
        if position <= byte_start < position + width:
            token_text = _entry_bytes(entry).decode("utf-8")
            if text[char_start] not in token_text:
                raise LogprobExtractionError(
                    f"token {token_text!r} covers the decision span without its digit"
                )
            return index, entry
        position += width
    raise LogprobExtractionError("no logprob token covers the ai_native value")


def _binary_candidate_pool(entry: dict[str, Any]) -> dict[str, float]:
    pool = {
        str(candidate["token"]): float(candidate["logprob"])
        for candidate in entry.get("top_logprobs") or []
    }
    pool.setdefault(str(entry["token"]), float(entry["logprob"]))
    return {
        token: logprob
        for token, logprob in pool.items()
        if candidate_value(token) is not None
        and logprob != MASKED_SENTINEL_LOGPROB
    }


def _reported_probability_mass(entry: dict[str, Any]) -> float:
    pool = {
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
