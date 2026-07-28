"""Offline run status derived from the immutable journal."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from .costing import actual_usage_cost
from .workflow import RunContext

# Inter-event gaps longer than this are treated as downtime (killed process,
# confirm prompt, laptop sleep) and excluded from throughput / ETA.
_ACTIVE_GAP_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class FailureBreakdown:
    total: int
    by_stage: dict[str, int]
    by_reason: dict[str, int]
    by_stage_and_reason: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class RunStatus:
    run_id: str
    kind: str
    model: str
    pass_a_effort: str
    pass_b_effort: str
    manifest_sha256: str
    manifest_total: int
    untouched: int
    pass_a_only: int
    complete: int
    remaining_runnable: int
    retryable_failures: FailureBreakdown
    terminal_failures: FailureBreakdown
    usage: dict[str, Any]
    elapsed_seconds: float | None
    throughput_companies_per_hour: float | None
    eta_seconds: float | None
    concurrency: dict[str, Any]
    rate_utilization: dict[str, Any]
    output_paths: dict[str, str]
    canonical_output_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_status(context: RunContext) -> RunStatus:
    """Summarize one run without constructing a provider client."""
    state = context.state
    total = context.manifest.row_count
    complete = len(state.completed)
    pass_a_only = len(set(state.pass_a) - set(state.completed))
    untouched = total - len(state.pass_a)

    retryable_events: list[Mapping[str, Any]] = []
    terminal_events: list[Mapping[str, Any]] = []
    for company_id, event in state.latest_errors.items():
        if company_id in state.completed:
            continue
        if event.get("retriable"):
            retryable_events.append(event)
        else:
            terminal_events.append(event)

    remaining_runnable = sum(
        1
        for row in context.manifest.rows
        if row.company_id not in state.completed
        and (
            row.company_id not in state.latest_errors
            or bool(state.latest_errors[row.company_id].get("retriable"))
        )
    )
    usage = actual_usage_cost(
        context.events,
        model=context.settings.model,
    )
    # Wall clock stays useful for "how long since this run began".
    # Throughput / ETA must use active processing time only, or long pauses
    # between resume sessions make a healthy runner look hours too slow.
    elapsed = _elapsed_seconds(context)
    active_elapsed = _active_elapsed_seconds(context)
    throughput = (
        complete / active_elapsed * 3_600
        if active_elapsed is not None and active_elapsed > 0 and complete > 0
        else None
    )
    if complete == total or remaining_runnable == 0:
        eta = 0.0
    elif throughput is not None and throughput > 0:
        eta = remaining_runnable / throughput * 3_600
    else:
        eta = None
    last_operational = _last_operational_event(context)
    rate_utilization = dict(last_operational.get("rate_utilization") or {})
    rate_targets = dict(last_operational.get("rate_limit_targets") or {})
    if rate_targets:
        rate_utilization["targets"] = rate_targets

    return RunStatus(
        run_id=context.run_id,
        kind=str(context.run_config.get("kind") or "unknown"),
        model=context.settings.model,
        pass_a_effort=str(context.header["request_identity"]["pass_a_effort"]),
        pass_b_effort=context.settings.pass_b_effort,
        manifest_sha256=context.manifest.manifest_sha256,
        manifest_total=total,
        untouched=untouched,
        pass_a_only=pass_a_only,
        complete=complete,
        remaining_runnable=remaining_runnable,
        retryable_failures=_failure_breakdown(retryable_events),
        terminal_failures=_failure_breakdown(terminal_events),
        usage=asdict(usage),
        elapsed_seconds=elapsed,
        throughput_companies_per_hour=throughput,
        eta_seconds=eta,
        concurrency={
            "last_recorded_limit": last_operational.get("concurrency_limit"),
            "active_at_last_event": last_operational.get("active_concurrency"),
        },
        rate_utilization=rate_utilization,
        output_paths={
            "journal": str(context.paths.journal),
            "in_progress_csv": str(context.paths.in_progress_csv),
            "final_csv": str(context.paths.final_csv),
            "failure_summary": str(context.paths.failure_summary),
            "run_summary": str(context.paths.run_summary),
        },
        canonical_output_ready=(
            complete == total and context.paths.final_csv.is_file()
        ),
    )


def _failure_breakdown(
    events: list[Mapping[str, Any]],
) -> FailureBreakdown:
    by_stage = Counter(str(event.get("stage") or "unknown") for event in events)
    by_reason = Counter(
        str(event.get("category") or "unknown") for event in events
    )
    nested: dict[str, Counter[str]] = {}
    for event in events:
        stage = str(event.get("stage") or "unknown")
        reason = str(event.get("category") or "unknown")
        nested.setdefault(stage, Counter())[reason] += 1
    return FailureBreakdown(
        total=len(events),
        by_stage=dict(sorted(by_stage.items())),
        by_reason=dict(sorted(by_reason.items())),
        by_stage_and_reason={
            stage: dict(sorted(counts.items()))
            for stage, counts in sorted(nested.items())
        },
    )


def _elapsed_seconds(context: RunContext) -> float | None:
    created = _parse_time(context.header.get("created_at"))
    if created is None:
        return None
    finished = [
        timestamp
        for event in context.events
        if (timestamp := _parse_time(event.get("finished_at"))) is not None
    ]
    end = max(finished) if finished else datetime.now(UTC)
    return max(0.0, (end - created).total_seconds())


def _active_elapsed_seconds(context: RunContext) -> float | None:
    """Sum productive time between journal timestamps, skipping long idle gaps."""
    timestamps = _operational_timestamps(context)
    if not timestamps:
        return None
    if len(timestamps) == 1:
        # In flight with no finishes yet: count wall time since start.
        return max(0.0, (datetime.now(UTC) - timestamps[0]).total_seconds())

    active = 0.0
    previous = timestamps[0]
    for current in timestamps[1:]:
        delta = (current - previous).total_seconds()
        if 0.0 < delta <= _ACTIVE_GAP_SECONDS:
            active += delta
        previous = current
    return active


def _operational_timestamps(context: RunContext) -> list[datetime]:
    stamps: list[datetime] = []
    created = _parse_time(context.header.get("created_at"))
    if created is not None:
        stamps.append(created)
    for event in context.events:
        finished = _parse_time(event.get("finished_at"))
        if finished is not None:
            stamps.append(finished)
    stamps.sort()
    return stamps


def _last_operational_event(context: RunContext) -> Mapping[str, Any]:
    for event in reversed(context.events):
        if (
            "concurrency_limit" in event
            or "rate_utilization" in event
            or "rate_limit_targets" in event
        ):
            return event
    return {}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
