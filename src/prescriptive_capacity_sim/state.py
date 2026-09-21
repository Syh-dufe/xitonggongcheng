"""Immutable state and outcome objects for the call-center simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace


SERVICE_CLASSES = ("regular", "specialist", "callback_special")


@dataclass(frozen=True)
class QueuedCall:
    service_class: int
    priority: bool
    service_minutes: float
    patience_periods: int
    waited_periods: int = 0

    def waited_one_period(self) -> "QueuedCall":
        return replace(self, waited_periods=self.waited_periods + 1)


@dataclass(frozen=True)
class SystemState:
    episode_id: int
    period: int
    weekday: str
    demand_state: int
    waiting: tuple[QueuedCall, ...] = ()

    @classmethod
    def empty(
        cls,
        episode_id: int,
        weekday: str,
        demand_state: int = 0,
    ) -> "SystemState":
        return cls(episode_id, 0, weekday.lower(), demand_state, ())

    @property
    def total_queue(self) -> int:
        return len(self.waiting)

    @property
    def queue_counts(self) -> tuple[int, int, int]:
        return tuple(
            sum(call.service_class == klass for call in self.waiting)
            for klass in range(3)
        )  # type: ignore[return-value]

    @property
    def priority_count(self) -> int:
        return sum(call.priority for call in self.waiting)

    @property
    def max_waited_periods(self) -> int:
        return max((call.waited_periods for call in self.waiting), default=0)


@dataclass(frozen=True)
class ExogenousDraw:
    new_calls: tuple[QueuedCall, ...]
    next_demand_state: int
    regular_availability: float
    specialist_availability: float
    service_efficiency: float
    manager_alarm: float

    @property
    def arrivals(self) -> tuple[int, int, int]:
        return tuple(
            sum(call.service_class == klass for call in self.new_calls)
            for klass in range(3)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class PeriodResult:
    action: int
    action_level: float
    arrivals: tuple[int, int, int]
    served: tuple[int, int, int]
    served_priority: tuple[int, int, int]
    served_nonpriority: tuple[int, int, int]
    abandoned: tuple[int, int, int]
    next_state: SystemState
    regular_agents: int
    specialist_agents: int
    temporary_agents: int
    specialist_minutes_cross_served: float
    base_staff_cost: float
    augmentation_cost: float
    waiting_cost: float
    abandonment_cost: float
    service_level_cost: float
    total_cost: float
    service_level: float
    mean_wait_minutes: float
    p95_wait_minutes: float
    safety_violation: bool
