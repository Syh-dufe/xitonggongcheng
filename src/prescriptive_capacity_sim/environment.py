"""Dynamic service-capacity environment with backlog aging and costs."""

from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .demand import DemandProcess
from .state import ExogenousShock, PeriodResult, SystemState


class CapacityEnvironment:
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.demand = DemandProcess(config)

    def reset(self, episode_id: int, disruption_state: int = 0) -> SystemState:
        return SystemState.empty(self.config, episode_id, disruption_state)

    def sample_exogenous(
        self,
        state: SystemState,
        rng: np.random.Generator,
        daily_effect: float,
    ) -> ExogenousShock:
        return self.demand.sample_shock(state, rng, daily_effect)

    def step(
        self,
        state: SystemState,
        action: int,
        rng: np.random.Generator,
        daily_effect: float,
    ) -> PeriodResult:
        shock = self.sample_exogenous(state, rng, daily_effect)
        return self.transition(state, action, shock)

    def transition(
        self,
        state: SystemState,
        action: int,
        shock: ExogenousShock,
    ) -> PeriodResult:
        if not 0 <= action < len(self.config.behavior.action_levels):
            raise ValueError(f"invalid action index: {action}")
        if any(value < 0 for value in shock.arrivals):
            raise ValueError("arrivals must be nonnegative")

        action_level = self.config.behavior.action_levels[action]
        capacity_multiplier = self.config.demand.shock_capacity_multipliers[state.disruption_state]
        base_available = int(math.floor(self.config.base_staff * capacity_multiplier))
        senior_available = int(math.floor(self.config.senior_staff * capacity_multiplier))
        temporary_staff = int(round((self.config.base_staff + self.config.senior_staff) * action_level))
        service_multiplier = (
            self.config.demand.shock_service_multipliers[state.disruption_state]
            * shock.service_multiplier
        )
        capacity_effort = max(
            0.0,
            (
                base_available * self.config.service_rates[0]
                + senior_available * self.config.service_rates[1]
                + temporary_staff * self.config.service_rates[2]
            )
            * service_multiplier,
        )

        queues = [list(row) for row in state.age_buckets]
        for task_class, arrived in enumerate(shock.arrivals):
            queues[task_class][0] += int(arrived)
        work_before = tuple(sum(row) for row in queues)

        completed = [0, 0, 0]
        remaining_effort = capacity_effort
        for task_class in (2, 1, 0):
            effort = self.config.service_effort[task_class]
            possible = min(work_before[task_class], int(remaining_effort // effort))
            if possible <= 0:
                continue
            completed[task_class] = _remove_oldest(queues[task_class], possible)
            remaining_effort -= completed[task_class] * effort

        aged_queues = tuple(_age_queue(row) for row in queues)
        next_state = SystemState(
            episode_id=state.episode_id,
            period=state.period + 1,
            disruption_state=shock.next_disruption_state,
            age_buckets=aged_queues,  # type: ignore[arg-type]
        )

        staffing_cost = (
            self.config.base_staff * self.config.costs.base_staff_unit_cost
            + self.config.senior_staff * self.config.costs.senior_staff_unit_cost
            + temporary_staff * self.config.costs.temporary_unit_cost
        )
        backlog_cost = float(
            sum(count * cost for count, cost in zip(next_state.backlog, self.config.costs.backlog_costs))
        )
        late_counts = tuple(
            sum(row[deadline:])
            for row, deadline in zip(next_state.age_buckets, self.config.safety.deadlines)
        )
        late_cost = float(
            sum(count * cost for count, cost in zip(late_counts, self.config.costs.late_costs))
        )
        workload_effort = sum(
            count * effort for count, effort in zip(work_before, self.config.service_effort)
        )
        overload = max(0.0, workload_effort - capacity_effort)
        overload_cost = overload * self.config.costs.overload_penalty

        critical_work = work_before[2]
        critical_rate = completed[2] / critical_work if critical_work else 1.0
        safety_violation = bool(
            critical_work > 0
            and (
                critical_rate < self.config.safety.critical_service_threshold
                or next_state.max_ages[2] > self.config.safety.critical_max_age
            )
        )
        safety_cost = (
            self.config.costs.safety_violation_penalty if safety_violation else 0.0
        )
        total_work = sum(work_before)
        service_rate = sum(completed) / total_work if total_work else 1.0
        total_cost = staffing_cost + backlog_cost + late_cost + overload_cost + safety_cost

        return PeriodResult(
            action=action,
            action_level=action_level,
            arrivals=shock.arrivals,
            completed=tuple(completed),  # type: ignore[arg-type]
            next_state=next_state,
            staffing_cost=float(staffing_cost),
            backlog_cost=backlog_cost,
            late_cost=late_cost,
            overload_cost=float(overload_cost),
            safety_cost=float(safety_cost),
            total_cost=float(total_cost),
            safety_violation=safety_violation,
            temporary_staff=temporary_staff,
            service_rate=float(service_rate),
        )


def _remove_oldest(queue: list[int], amount: int) -> int:
    removed = 0
    for age in range(len(queue) - 1, -1, -1):
        take = min(queue[age], amount - removed)
        queue[age] -= take
        removed += take
        if removed == amount:
            break
    return removed


def _age_queue(queue: list[int]) -> tuple[int, ...]:
    aged = [0 for _ in queue]
    for age, count in enumerate(queue):
        aged[min(age + 1, len(queue) - 1)] += count
    return tuple(aged)
