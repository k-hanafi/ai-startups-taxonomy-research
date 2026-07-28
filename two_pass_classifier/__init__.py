"""Production two-pass classifier contracts and artifact builders.

This PR freezes taxonomy contracts only. The async runner, journal, CLI, and
workflow modules land in later stacked PRs.
"""

from .config import (
    DEFAULT_MODEL,
    DEFAULT_PASS_B_EFFORT,
    DEFAULT_PASS_B_REASONING_EFFORT,
    PASS_A_EFFORT,
    PASS_A_REASONING_EFFORT,
    PASS_A_TOP_LOGPROBS,
    SUPPORTED_MODELS,
)
from .request_builder import RequestSettings

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PASS_B_EFFORT",
    "DEFAULT_PASS_B_REASONING_EFFORT",
    "PASS_A_EFFORT",
    "PASS_A_REASONING_EFFORT",
    "PASS_A_TOP_LOGPROBS",
    "RequestSettings",
    "SUPPORTED_MODELS",
]
