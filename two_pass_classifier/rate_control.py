"""Shared rate admission, adaptive concurrency, and cache-route warming."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, TypeVar

from . import config

T = TypeVar("T")


class AdmissionStopped(RuntimeError):
    """Dispatch stopped while a request was waiting for admission."""


@dataclass(slots=True)
class Reservation:
    model: str
    admitted_at: float
    tokens: int
    in_flight: bool = True


@dataclass(slots=True)
class _HeaderBudget:
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    requests_reset_at: float | None = None
    tokens_reset_at: float | None = None


@dataclass(slots=True)
class _ModelWindow:
    requests_per_minute: int
    tokens_per_minute: int
    reservations: deque[Reservation] = field(default_factory=deque)
    paused_until: float = 0.0
    header_budget: _HeaderBudget = field(default_factory=_HeaderBudget)


class DualRateAdmissionController:
    """Gate every physical request against shared RPM and TPM windows."""

    def __init__(
        self,
        *,
        model_limits: Mapping[str, config.ModelRateLimit] = config.MODEL_RATE_LIMITS,
        target_fraction: float = config.RATE_LIMIT_TARGET_FRACTION,
        window_seconds: float = config.RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        self._target_fraction = target_fraction
        self._window_seconds = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._models = {
            model: _ModelWindow(
                requests_per_minute=limit.requests_per_minute,
                tokens_per_minute=limit.tokens_per_minute,
            )
            for model, limit in model_limits.items()
        }

    def target_limits(self, model: str) -> tuple[int, int]:
        state = self._model(model)
        return (
            int(state.requests_per_minute * self._target_fraction),
            int(state.tokens_per_minute * self._target_fraction),
        )

    async def acquire(
        self,
        model: str,
        *,
        estimated_input_tokens: int,
        output_allowance: int,
        stop_event: asyncio.Event | None = None,
    ) -> Reservation:
        reserve = estimated_input_tokens + output_allowance
        if estimated_input_tokens < 0 or output_allowance < 0 or reserve < 1:
            raise ValueError("token reservation must be positive")

        while True:
            if stop_event is not None and stop_event.is_set():
                raise AdmissionStopped("dispatch stopped before rate admission")
            async with self._lock:
                now = self._clock()
                state = self._model(model)
                self._prune(state, now)
                request_target, token_target = self.target_limits(model)
                if reserve > token_target:
                    raise ValueError(
                        f"one request reserves {reserve} tokens, above the "
                        f"target TPM window {token_target}"
                    )
                wait_for = self._wait_seconds(
                    state,
                    now,
                    reserve,
                    request_target,
                    token_target,
                )
                if wait_for <= 0:
                    reservation = Reservation(
                        model=model,
                        admitted_at=now,
                        tokens=reserve,
                    )
                    state.reservations.append(reservation)
                    budget = state.header_budget
                    if budget.remaining_requests is not None:
                        budget.remaining_requests = max(
                            0, budget.remaining_requests - 1
                        )
                    if budget.remaining_tokens is not None:
                        budget.remaining_tokens = max(
                            0, budget.remaining_tokens - reserve
                        )
                    return reservation
            await self._sleep_interruptibly(wait_for, stop_event)

    async def reconcile(
        self,
        reservation: Reservation,
        *,
        actual_input_tokens: int | None,
        actual_output_tokens: int | None,
    ) -> None:
        if actual_input_tokens is None and actual_output_tokens is None:
            return
        actual = max(0, int(actual_input_tokens or 0)) + max(
            0, int(actual_output_tokens or 0)
        )
        if actual < 1:
            return
        async with self._lock:
            difference = actual - reservation.tokens
            reservation.tokens = actual
            state = self._model(reservation.model)
            remaining = state.header_budget.remaining_tokens
            if remaining is not None:
                state.header_budget.remaining_tokens = min(
                    state.tokens_per_minute,
                    max(
                        0,
                        remaining - difference,
                    ),
                )

    async def release(self, reservation: Reservation) -> None:
        """Mark one reservation finished so the sliding window may drop it."""
        async with self._lock:
            reservation.in_flight = False

    async def observe_headers(
        self,
        model: str,
        headers: Mapping[str, str],
    ) -> None:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        async with self._lock:
            now = self._clock()
            state = self._model(model)
            limit_requests = _positive_int(
                normalized.get("x-ratelimit-limit-requests")
            )
            limit_tokens = _positive_int(
                normalized.get("x-ratelimit-limit-tokens")
            )
            if limit_requests is not None:
                state.requests_per_minute = limit_requests
            if limit_tokens is not None:
                state.tokens_per_minute = limit_tokens

            budget = state.header_budget
            budget.remaining_requests = _nonnegative_int(
                normalized.get("x-ratelimit-remaining-requests")
            )
            budget.remaining_tokens = _nonnegative_int(
                normalized.get("x-ratelimit-remaining-tokens")
            )
            request_reset = _duration_seconds(
                normalized.get("x-ratelimit-reset-requests")
            )
            token_reset = _duration_seconds(
                normalized.get("x-ratelimit-reset-tokens")
            )
            budget.requests_reset_at = (
                now + request_reset if request_reset is not None else None
            )
            budget.tokens_reset_at = (
                now + token_reset if token_reset is not None else None
            )

    async def pause_model(self, model: str, seconds: float) -> None:
        async with self._lock:
            state = self._model(model)
            state.paused_until = max(
                state.paused_until,
                self._clock() + max(0.0, seconds),
            )

    async def utilization(self, model: str) -> float:
        breakdown = await self.utilization_breakdown(model)
        return breakdown["combined"]

    async def utilization_breakdown(self, model: str) -> dict[str, float]:
        """Return current request and token shares of the target windows."""
        async with self._lock:
            now = self._clock()
            state = self._model(model)
            self._prune(state, now)
            request_target, token_target = self.target_limits(model)
            request_share = len(state.reservations) / request_target
            token_share = (
                sum(item.tokens for item in state.reservations) / token_target
            )
            return {
                "rpm": request_share,
                "tpm": token_share,
                "combined": max(request_share, token_share),
            }

    def _model(self, model: str) -> _ModelWindow:
        try:
            return self._models[model]
        except KeyError as exc:
            raise ValueError(f"no rate-limit data for model {model!r}") from exc

    def _prune(self, state: _ModelWindow, now: float) -> None:
        cutoff = now - self._window_seconds
        # Keep in-flight reservations even after admitted_at ages out. Their
        # actual usage is reconciled only when the HTTP call finishes, so
        # dropping them early under-counts TPM and can admit past the target.
        state.reservations = deque(
            item
            for item in state.reservations
            if item.in_flight or item.admitted_at > cutoff
        )

        budget = state.header_budget
        if (
            budget.requests_reset_at is not None
            and now >= budget.requests_reset_at
        ):
            budget.remaining_requests = None
            budget.requests_reset_at = None
        if budget.tokens_reset_at is not None and now >= budget.tokens_reset_at:
            budget.remaining_tokens = None
            budget.tokens_reset_at = None

    def _wait_seconds(
        self,
        state: _ModelWindow,
        now: float,
        reserve: int,
        request_target: int,
        token_target: int,
    ) -> float:
        waits = [max(0.0, state.paused_until - now)]
        budget = state.header_budget
        if budget.remaining_requests is not None and budget.remaining_requests < 1:
            if budget.requests_reset_at is not None:
                waits.append(max(0.0, budget.requests_reset_at - now))
        if budget.remaining_tokens is not None and budget.remaining_tokens < reserve:
            if budget.tokens_reset_at is not None:
                waits.append(max(0.0, budget.tokens_reset_at - now))

        reservations = state.reservations
        if len(reservations) >= request_target:
            waits.append(
                self._reservation_wait(
                    reservations[len(reservations) - request_target],
                    now,
                )
            )

        current_tokens = sum(item.tokens for item in reservations)
        excess = current_tokens + reserve - token_target
        if excess > 0:
            released = 0
            for item in reservations:
                released += item.tokens
                if released >= excess:
                    waits.append(self._reservation_wait(item, now))
                    break
        return max(waits)

    def _reservation_wait(self, reservation: Reservation, now: float) -> float:
        remaining = reservation.admitted_at + self._window_seconds - now
        if remaining > 0:
            return remaining
        if reservation.in_flight:
            # Past the nominal window but still outstanding: poll until release
            # instead of treating capacity as free.
            return config.RATE_LIMIT_IN_FLIGHT_POLL_SECONDS
        return 0.0

    async def _sleep_interruptibly(
        self,
        delay: float,
        stop_event: asyncio.Event | None,
    ) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        if stop_event is None:
            await self._sleep(delay)
            return
        sleep_task = asyncio.create_task(self._sleep(delay))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {sleep_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if stop_task in done and stop_event.is_set():
            raise AdmissionStopped("dispatch stopped during rate wait")


class AdaptiveConcurrencyController:
    """Apply a small deterministic additive-increase, multiplicative-decrease rule."""

    def __init__(
        self,
        *,
        initial: int = config.INITIAL_CONCURRENCY,
        ceiling: int = config.MAX_CONCURRENCY,
    ) -> None:
        if initial < 1 or ceiling < initial:
            raise ValueError("concurrency must satisfy 1 <= initial <= ceiling")
        self.initial = initial
        self.ceiling = ceiling
        self.limit = initial
        self.active = 0
        self._condition = asyncio.Condition()
        self._outcomes: deque[bool] = deque(
            maxlen=config.ADAPTIVE_ERROR_WINDOW
        )
        self._successes_since_increase = 0
        self._latency_ewma: float | None = None
        self._writer_ack_ewma: float | None = None

    @asynccontextmanager
    async def slot(
        self,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[None]:
        await self._acquire(stop_event)
        try:
            yield
        finally:
            async with self._condition:
                self.active -= 1
                self._condition.notify_all()

    async def record_success(
        self,
        *,
        latency_seconds: float,
        writer_ack_seconds: float,
        rate_utilization: float,
    ) -> None:
        async with self._condition:
            self._outcomes.append(False)
            self._latency_ewma = _ewma(self._latency_ewma, latency_seconds)
            self._writer_ack_ewma = _ewma(
                self._writer_ack_ewma, writer_ack_seconds
            )
            self._successes_since_increase += 1
            healthy = (
                self._successes_since_increase
                >= config.ADAPTIVE_INCREASE_EVERY_SUCCESSES
                and rate_utilization
                < config.ADAPTIVE_MAX_RATE_UTILIZATION
                and (self._latency_ewma or 0.0)
                <= config.ADAPTIVE_MAX_LATENCY_SECONDS
                and (self._writer_ack_ewma or 0.0)
                <= config.ADAPTIVE_MAX_WRITER_ACK_SECONDS
                and self.error_rate <= config.ADAPTIVE_MAX_ERROR_RATE
            )
            if healthy and self.limit < self.ceiling:
                increment = max(
                    1,
                    int(self.limit * config.ADAPTIVE_INCREASE_FRACTION),
                )
                self.limit = min(self.ceiling, self.limit + increment)
                self._successes_since_increase = 0
                self._condition.notify_all()

    async def record_error(self, category: str) -> None:
        async with self._condition:
            self._outcomes.append(True)
            self._successes_since_increase = 0
            if category == "rate_limit":
                self.limit = max(
                    1,
                    int(self.limit * config.ADAPTIVE_429_REDUCTION),
                )
            elif category in {"service_unavailable", "timeout", "connection"}:
                self.limit = max(
                    1,
                    int(self.limit * config.ADAPTIVE_TRANSIENT_REDUCTION),
                )
            self._condition.notify_all()

    @property
    def error_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    async def _acquire(self, stop_event: asyncio.Event | None) -> None:
        async with self._condition:
            while self.active >= self.limit:
                if stop_event is not None and stop_event.is_set():
                    raise AdmissionStopped(
                        "dispatch stopped during concurrency wait"
                    )
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=0.100,
                    )
                except asyncio.TimeoutError:
                    pass
            if stop_event is not None and stop_event.is_set():
                raise AdmissionStopped("dispatch stopped before request")
            self.active += 1


class CacheRouteWarmer:
    """Let one first real call finish before releasing each cache route."""

    def __init__(self, routes: tuple[str, ...]) -> None:
        self._locks = {route: asyncio.Lock() for route in routes}
        self._warmed: set[str] = set()

    @property
    def warmed_routes(self) -> frozenset[str]:
        return frozenset(self._warmed)

    async def run(
        self,
        route: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        try:
            lock = self._locks[route]
        except KeyError as exc:
            raise ValueError(f"unknown cache route {route!r}") from exc
        if route in self._warmed:
            return await operation()
        async with lock:
            if route in self._warmed:
                return await operation()
            result = await operation()
            self._warmed.add(route)
            return result


def _ewma(previous: float | None, value: float, alpha: float = 0.20) -> float:
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def _positive_int(value: str | None) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _nonnegative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(float(value.replace(",", "").strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def _duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().lower()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3_600.0}
    matches = _DURATION_PART.findall(text)
    if not matches:
        return None
    return sum(float(amount) * units[unit] for amount, unit in matches)
