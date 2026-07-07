"""Prompt contract checks for the two-pass classifier prompts."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def test_zero_family_sorting_keeps_counterfactual_guard():
    prompt = (PROMPTS_DIR / "family_block_not_ai_native.txt").read_text(encoding="utf-8")
    normalized = " ".join(prompt.split())

    assert "meaningfully augments the existing software value proposition" in normalized
    assert "traditional software/SaaS companies transitioning into AI-augmented products" in normalized
    assert "Do not use 0B when AI is the product mechanism" in normalized
    assert "Do not treat 0B as a home for AI-core products" in normalized
    assert "set boundary_disagreement = true" in normalized
