"""Tests for Tavily credit estimation."""

from __future__ import annotations

from wayback_machine.config import estimate_credits


def test_estimate_credits_basic() -> None:
    assert estimate_credits(0) == 0.0
    assert estimate_credits(5, extract_depth="basic") == 1.0
    assert estimate_credits(10, extract_depth="basic") == 2.0


def test_estimate_credits_advanced_costs_twice_basic() -> None:
    assert estimate_credits(5, extract_depth="advanced") == 2.0
    assert estimate_credits(10, extract_depth="advanced") == 4.0
    assert (
        estimate_credits(10, extract_depth="advanced")
        == estimate_credits(10, extract_depth="basic") * 2
    )
