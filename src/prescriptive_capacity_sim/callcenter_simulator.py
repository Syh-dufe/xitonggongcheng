"""Cloneable, call-level discrete-event simulation for interval staffing."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class SimulatedCall:
    call_key: str
    arrival_second: float
    service_seconds: float
    patience_seconds: float
    priority: int
    call_type: str


@dataclass(frozen=True)
class WaitingCall:
    call: SimulatedCall
    sequence: int

    @property
    def abandonment_second(self) -> float:
        return self.call.arrival_second + self.call.patience_seconds


@dataclass(frozen=True)
class BusyCall:
    call: SimulatedCall
    service_start_second: float
    completion_second: float
    wait_seconds: float


@dataclass
class CallCenterState:
    now_second: float
    waiting: list[WaitingCall] = field(default_factory=list)
    busy: list[BusyCall] = field(default_factory=list)
    next_sequence: int = 0

    @classmethod
    def empty(cls, now_second: float = 0.0) -> "CallCenterState":
        return cls(now_second=float(now_second))

    def clone(self) -> "CallCenterState":
        return CallCenterState(
            now_second=self.now_second,
            waiting=list(self.waiting),
            busy=list(self.busy),
            next_sequence=self.next_sequence,
        )


@dataclass(frozen=True)
class IntervalResult:
    arrivals: int
    started_service: int
    completed_service: int
    abandoned: int
    mean_wait_seconds: float
    service_level: float
    sla_violations: int
    ending_queue: int
    busy_seconds: float
    staffed_seconds: float
    started_call_keys: tuple[str, ...]
    completed_call_keys: tuple[str, ...]
    abandoned_call_keys: tuple[str, ...]
    next_state: CallCenterState


def _validate_call(call: SimulatedCall) -> None:
    if not isfinite(call.service_seconds) or call.service_seconds <= 0:
        raise ValueError(f"service_seconds must be positive for {call.call_key}")
    if not isfinite(call.patience_seconds) or call.patience_seconds < 0:
        raise ValueError(f"patience_seconds must be nonnegative for {call.call_key}")
    if not isfinite(call.arrival_second):
        raise ValueError(f"arrival_second must be finite for {call.call_key}")


def simulate_interval(
    state: CallCenterState,
    arrivals: Sequence[SimulatedCall],
    staffing: int,
    interval_seconds: int = 1800,
    service_level_seconds: int = 20,
) -> IntervalResult:
    """Advance a cloned state through one staffing decision interval.

    Staffing reductions never interrupt calls already in service. At a tie
    between an agent becoming free and a patience deadline, service wins.
    """

    if isinstance(staffing, bool) or not isinstance(staffing, int) or staffing < 0:
        raise ValueError("staffing must be a nonnegative integer")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if service_level_seconds < 0:
        raise ValueError("service_level_seconds must be nonnegative")

    working = state.clone()
    start = float(working.now_second)
    end = start + float(interval_seconds)
    indexed_arrivals = list(enumerate(arrivals))
    for _, call in indexed_arrivals:
        _validate_call(call)
        if call.arrival_second < start or call.arrival_second >= end:
            raise ValueError(
                f"arrival {call.call_key} at {call.arrival_second} is outside "
                f"[{start}, {end})"
            )
    indexed_arrivals.sort(key=lambda item: (item[1].arrival_second, item[0]))

    arrival_index = 0
    now = start
    busy_seconds = 0.0
    started_keys: list[str] = []
    completed_keys: list[str] = []
    abandoned_keys: list[str] = []
    realized_waits: list[float] = []
    timely_starts = 0

    while True:
        completed_now = [
            item for item in working.busy if item.completion_second <= now
        ]
        if completed_now:
            completed_now.sort(key=lambda item: (item.completion_second, item.call.call_key))
            completed_keys.extend(item.call.call_key for item in completed_now)
            completed_set = set(completed_now)
            working.busy = [item for item in working.busy if item not in completed_set]

        while (
            arrival_index < len(indexed_arrivals)
            and indexed_arrivals[arrival_index][1].arrival_second <= now
        ):
            call = indexed_arrivals[arrival_index][1]
            working.waiting.append(
                WaitingCall(call=call, sequence=working.next_sequence)
            )
            working.next_sequence += 1
            arrival_index += 1

        # A deadline strictly before the current clock can only arise from an
        # externally supplied state. Remove it before assigning service.
        overdue = [item for item in working.waiting if item.abandonment_second < now]
        if overdue:
            overdue_set = set(overdue)
            for item in sorted(overdue, key=lambda value: value.sequence):
                abandoned_keys.append(item.call.call_key)
                realized_waits.append(max(0.0, now - item.call.arrival_second))
            working.waiting = [item for item in working.waiting if item not in overdue_set]

        if now < end:
            working.waiting.sort(
                key=lambda item: (
                    -item.call.priority,
                    item.call.arrival_second,
                    item.sequence,
                )
            )
            while working.waiting and len(working.busy) < staffing:
                item = working.waiting.pop(0)
                wait = max(0.0, now - item.call.arrival_second)
                working.busy.append(
                    BusyCall(
                        call=item.call,
                        service_start_second=now,
                        completion_second=now + item.call.service_seconds,
                        wait_seconds=wait,
                    )
                )
                started_keys.append(item.call.call_key)
                realized_waits.append(wait)
                if wait <= service_level_seconds:
                    timely_starts += 1

        expired = [
            item for item in working.waiting if item.abandonment_second <= now
        ]
        if expired:
            expired_set = set(expired)
            for item in sorted(expired, key=lambda value: value.sequence):
                abandoned_keys.append(item.call.call_key)
                realized_waits.append(max(0.0, now - item.call.arrival_second))
            working.waiting = [item for item in working.waiting if item not in expired_set]

        if now >= end:
            break

        candidates = [end]
        if arrival_index < len(indexed_arrivals):
            candidates.append(indexed_arrivals[arrival_index][1].arrival_second)
        if working.busy:
            candidates.append(min(item.completion_second for item in working.busy))
        if working.waiting:
            candidates.append(
                min(item.abandonment_second for item in working.waiting)
            )
        next_time = min(value for value in candidates if value > now)
        busy_seconds += len(working.busy) * (next_time - now)
        now = next_time

    working.now_second = end
    resolved = len(started_keys) + len(abandoned_keys)
    service_level = timely_starts / resolved if resolved else 1.0
    sla_violations = resolved - timely_starts
    mean_wait = sum(realized_waits) / len(realized_waits) if realized_waits else 0.0
    return IntervalResult(
        arrivals=len(arrivals),
        started_service=len(started_keys),
        completed_service=len(completed_keys),
        abandoned=len(abandoned_keys),
        mean_wait_seconds=mean_wait,
        service_level=service_level,
        sla_violations=sla_violations,
        ending_queue=len(working.waiting),
        busy_seconds=busy_seconds,
        staffed_seconds=float(staffing * interval_seconds),
        started_call_keys=tuple(started_keys),
        completed_call_keys=tuple(completed_keys),
        abandoned_call_keys=tuple(abandoned_keys),
        next_state=working,
    )
