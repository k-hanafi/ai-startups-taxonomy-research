from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from two_pass_classifier import config
from two_pass_classifier.rate_control import (
    AdaptiveConcurrencyController,
    CacheRouteWarmer,
    DualRateAdmissionController,
)
from two_pass_classifier.runner import categorize_error


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


def test_production_rate_targets_are_exactly_eighty_percent():
    controller = DualRateAdmissionController()

    for model in config.SUPPORTED_MODELS:
        assert controller.target_limits(model) == (24_000, 144_000_000)


@pytest.mark.asyncio
async def test_shared_rpm_and_tpm_admission_waits_for_window():
    clock = _FakeClock()
    controller = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(10, 100)},
        target_fraction=0.8,
        clock=clock,
        sleep=clock.sleep,
    )

    for _ in range(8):
        reservation = await controller.acquire(
            "test",
            estimated_input_tokens=5,
            output_allowance=5,
        )
        await controller.release(reservation)
    await controller.acquire(
        "test",
        estimated_input_tokens=5,
        output_allowance=5,
    )

    assert clock.sleeps == [60.0]
    assert controller.target_limits("test") == (8, 80)

    tpm_clock = _FakeClock()
    tpm = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(100, 100)},
        target_fraction=0.8,
        clock=tpm_clock,
        sleep=tpm_clock.sleep,
    )
    for _ in range(2):
        reservation = await tpm.acquire(
            "test",
            estimated_input_tokens=20,
            output_allowance=20,
        )
        await tpm.release(reservation)
    await tpm.acquire(
        "test",
        estimated_input_tokens=1,
        output_allowance=1,
    )
    assert tpm_clock.sleeps == [60.0]


@pytest.mark.asyncio
async def test_actual_usage_reconciles_reservation_and_headers_revise_limits():
    controller = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(10, 100)}
    )
    reservation = await controller.acquire(
        "test",
        estimated_input_tokens=5,
        output_allowance=5,
    )
    await controller.reconcile(
        reservation,
        actual_input_tokens=60,
        actual_output_tokens=10,
    )
    assert await controller.utilization("test") == pytest.approx(70 / 80)

    await controller.observe_headers(
        "test",
        {
            "x-ratelimit-limit-requests": "20",
            "x-ratelimit-limit-tokens": "200",
        },
    )
    assert controller.target_limits("test") == (16, 160)


@pytest.mark.asyncio
async def test_tiny_provider_headers_keep_nonzero_rate_targets():
    controller = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(10, 100)},
        target_fraction=0.8,
    )
    await controller.observe_headers(
        "test",
        {
            "x-ratelimit-limit-requests": "1",
            "x-ratelimit-limit-tokens": "1",
        },
    )
    assert controller.target_limits("test") == (1, 1)
    reservation = await controller.acquire(
        "test",
        estimated_input_tokens=0,
        output_allowance=1,
    )
    assert await controller.utilization("test") == pytest.approx(1.0)
    await controller.release(reservation)


@pytest.mark.asyncio
async def test_in_flight_reservations_stay_in_tpm_window_until_release():
    clock = _FakeClock()
    controller = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(100, 100)},
        target_fraction=0.8,
        clock=clock,
        sleep=clock.sleep,
    )
    reservation = await controller.acquire(
        "test",
        estimated_input_tokens=20,
        output_allowance=20,
    )
    clock.now = 90.0
    await controller.reconcile(
        reservation,
        actual_input_tokens=50,
        actual_output_tokens=20,
    )
    assert await controller.utilization("test") == pytest.approx(70 / 80)

    await controller.release(reservation)
    clock.now = 91.0
    assert await controller.utilization("test") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_global_pause_and_adaptive_reductions():
    clock = _FakeClock()
    rate = DualRateAdmissionController(
        model_limits={"test": config.ModelRateLimit(100, 1_000)},
        clock=clock,
        sleep=clock.sleep,
    )
    await rate.pause_model("test", 3.0)
    await rate.acquire(
        "test",
        estimated_input_tokens=1,
        output_allowance=1,
    )
    assert clock.sleeps == [3.0]

    adaptive = AdaptiveConcurrencyController(initial=128, ceiling=256)
    await adaptive.record_error("rate_limit")
    assert adaptive.limit == 64
    await adaptive.record_error("timeout")
    assert adaptive.limit == 51


def test_retry_categories_separate_transient_and_permanent_failures():
    class FakeHTTPError(Exception):
        def __init__(self, status: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status
            self.response = SimpleNamespace(status_code=status, headers={})

    rate_limit = categorize_error(FakeHTTPError(429, "too many requests"))
    assert rate_limit.category == "rate_limit"
    assert rate_limit.retriable is True

    quota = categorize_error(
        FakeHTTPError(429, "insufficient_quota: credit balance exhausted")
    )
    assert quota.category == "quota_or_billing"
    assert quota.retriable is False

    bad_request = categorize_error(
        FakeHTTPError(400, "unsupported parameter")
    )
    assert bad_request.category == "bad_request"
    assert bad_request.retriable is False

    timeout = categorize_error(asyncio.TimeoutError("ambiguous timeout"))
    assert timeout.retriable is True
    assert timeout.ambiguous_provider_billing is True


@pytest.mark.asyncio
async def test_each_cache_route_serializes_only_its_first_real_call():
    warmer = CacheRouteWarmer(("a",))
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> str:
        order.append("first-start")
        first_started.set()
        await release_first.wait()
        order.append("first-finish")
        return "first"

    async def second() -> str:
        order.append("second-start")
        return "second"

    first_task = asyncio.create_task(warmer.run("a", first))
    await first_started.wait()
    second_task = asyncio.create_task(warmer.run("a", second))
    await asyncio.sleep(0)
    assert order == ["first-start"]

    release_first.set()
    assert await asyncio.gather(first_task, second_task) == ["first", "second"]
    assert order == ["first-start", "first-finish", "second-start"]
    assert warmer.warmed_routes == frozenset({"a"})
