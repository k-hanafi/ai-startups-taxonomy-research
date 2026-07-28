"""Load the production-owned two-pass prompt contracts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
BINARY_GATE_PROMPT = PROMPTS_DIR / "binary_gate_prompt.txt"
SUBCLASS_RAD_PROMPT = PROMPTS_DIR / "subclass_rad_prompt.txt"
FAMILY_BLOCK_AI = PROMPTS_DIR / "family_block_ai_native.txt"
FAMILY_BLOCK_NOT_AI = PROMPTS_DIR / "family_block_not_ai_native.txt"
FAMILY_BLOCK_PLACEHOLDER = "{family_block}"


@lru_cache(maxsize=1)
def load_pass_a_prompt() -> str:
    return BINARY_GATE_PROMPT.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=2)
def load_pass_b_prompt(family: int) -> str:
    """Load Pass B with exactly one fixed-family block inserted."""
    if family not in (0, 1):
        raise ValueError(f"family must be 0 or 1, got {family!r}")
    template = SUBCLASS_RAD_PROMPT.read_text(encoding="utf-8").strip()
    block_path = FAMILY_BLOCK_AI if family == 1 else FAMILY_BLOCK_NOT_AI
    block = block_path.read_text(encoding="utf-8").strip()
    if template.count(FAMILY_BLOCK_PLACEHOLDER) != 1:
        raise ValueError(
            f"{SUBCLASS_RAD_PROMPT.name} must contain exactly one "
            f"{FAMILY_BLOCK_PLACEHOLDER} placeholder"
        )
    return template.replace(FAMILY_BLOCK_PLACEHOLDER, block)
