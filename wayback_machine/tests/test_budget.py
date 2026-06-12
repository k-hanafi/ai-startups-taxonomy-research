"""Tests for pessimistic in-flight budget accounting."""

from __future__ import annotations

from wayback_machine.config import estimate_credits


def test_budget_blocks_when_in_flight_would_exceed() -> None:
    """Four in-flight rows at 4 successes should block the next pop at budget=1."""
    budget_credits = 1.0
    state_successful = 4
    in_flight_rows = 4
    assert (
        estimate_credits(state_successful + in_flight_rows, extract_depth="basic")
        >= budget_credits
    )
