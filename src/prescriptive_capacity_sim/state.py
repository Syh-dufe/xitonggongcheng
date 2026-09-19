"""Immutable state and result records used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SimulationConfig


TASK_CLASSES = ("normal", "urgent", "critical")


@dataclass(frozen=True)
class SystemState:
    episode_id: int
    period: int
    disruption_state: int
    age_buckets: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @classmethod
    def empty(
        cls,
        config: "SimulationConfig",
        episode_id: int,
        disruption_state: int = 0,
    ) -> "SystemState":
        buckets = tuple(tuple(0 for _ in range(config.max_age_bucket + 1)) for _ in TASK_CLASSES)
        return cls(episode_id, 0, disruption_state, buckets)  # type: ignore[arg-type]

    @classmethod
    def from_backlog(
        cls,
        config: "SimulationConfig",
        episode_id: int,
        period: int,
        backlog: tuple[int, int, int],
        disruption_state: int = 0,
    ) -> "SystemState":
        buckets = tuple(
            (int(count),) + tuple(0 for _ in range(config.max_age_bucket))
            for count in backlog
        )
        return cls(episode_id, period, disruption_state, buckets)  # type: ignore[arg-type]

    @property
    def backlog(self) -> tuple[int, int, int]:
        return tuple(sum(row) for row in self.age_buckets)  # type: ignore[return-value]

    @property
    def total_backlog(self) -> int:
        return sum(self.backlog)

    @property
    def max_ages(self) -> tuple[int, int, int]:
        maxima = []
        for row in self.age_buckets:
            maxima.append(max((idx for idx, count in enumerate(row) if count > 0), default=0))
        return tuple(maxima)  # type: ignore[return-value]


@dataclass(frozen=True)
class ExogenousShock:
    arrivals: tuple[int, int, int]
    next_disruption_state: int
    service_multiplier: float
    manager_alarm: float


@dataclass(frozen=True)
class PeriodResult:
    action: int
    action_level: float
    arrivals: tuple[int, int, int]
    completed: tuple[int, int, int]
    next_state: SystemState
    staffing_cost: float
    backlog_cost: float
    late_cost: float
    overload_cost: float
    safety_cost: float
    total_cost: float
    safety_violation: bool
    temporary_staff: int
    service_rate: float

