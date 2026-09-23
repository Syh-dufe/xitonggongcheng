"""Two-pool call-center queue with priority, cross-skilling, and abandonment."""

from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .state import ExogenousDraw, PeriodResult, QueuedCall, SystemState


class CallCenterEnvironment:
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config

    def reset(
        self,
        episode_id: int,
        weekday: str,
        demand_state: int = 0,
    ) -> SystemState:
        return SystemState.empty(episode_id, weekday, demand_state)

    def transition(
        self,
        state: SystemState,
        action: int,
        draw: ExogenousDraw,
    ) -> PeriodResult:
        actions = self.config.behavior.action_levels
        if not 0 <= action < len(actions):
            raise ValueError(f"invalid action: {action}")
        action_level = actions[action]
        resource = self.config.resources
        state_capacity = self.config.capacity_state_multipliers[state.demand_state]
        regular_agents = max(0, int(math.floor(
            resource.regular_agents * state_capacity * draw.regular_availability
        )))
        specialist_agents = max(0, int(math.floor(
            resource.specialist_agents * state_capacity * draw.specialist_availability
        )))
        base_total = resource.regular_agents + resource.specialist_agents
        temporary_agents = int(math.ceil(base_total * action_level)) if action_level > 0 else 0
        temp_regular = int(round(temporary_agents * resource.callback_regular_share))
        temp_specialist = temporary_agents - temp_regular
        regular_agents += temp_regular
        specialist_agents += temp_specialist

        regular_capacity = regular_agents * resource.minutes_per_period * draw.service_efficiency
        specialist_capacity = specialist_agents * resource.minutes_per_period * draw.service_efficiency
        calls = list(state.waiting) + list(draw.new_calls)
        active = set(range(len(calls)))
        served_indices: list[int] = []

        specialist_capacity, served = _serve(
            calls, active, specialist_capacity, eligible=(1, 2), class_order=(1, 2),
            efficiency=1.0,
        )
        served_indices.extend(served)
        regular_capacity, served = _serve(
            calls, active, regular_capacity, eligible=(0, 2), class_order=(0, 2),
            efficiency=1.0,
        )
        served_indices.extend(served)
        before_cross = specialist_capacity
        specialist_capacity, served = _serve(
            calls, active, specialist_capacity, eligible=(0,), class_order=(0,),
            efficiency=resource.cross_skill_efficiency,
        )
        served_indices.extend(served)
        specialist_cross_used = before_cross - specialist_capacity

        abandoned_indices: list[int] = []
        next_waiting: list[QueuedCall] = []
        for index in sorted(active):
            call = calls[index]
            if call.waited_periods + 1 >= call.patience_periods:
                abandoned_indices.append(index)
            else:
                next_waiting.append(call.waited_one_period())

        served_calls = [calls[index] for index in served_indices]
        abandoned_calls = [calls[index] for index in abandoned_indices]
        arrivals = draw.arrivals
        served_counts = _counts(served_calls)
        abandoned_counts = _counts(abandoned_calls)
        served_priority = _counts([call for call in served_calls if call.priority])
        served_nonpriority = tuple(
            served_counts[i] - served_priority[i] for i in range(3)
        )
        exit_calls = (*served_calls, *abandoned_calls)
        exit_wait_minutes_by_class = tuple(
            tuple(
                call.intraperiod_wait_minutes
                + call.waited_periods * resource.minutes_per_period
                for call in exit_calls
                if call.service_class == service_class
            )
            for service_class in range(3)
        )
        wait_observations = [
            wait
            for waits in exit_wait_minutes_by_class
            for wait in waits
        ]
        mean_wait = float(np.mean(wait_observations)) if wait_observations else 0.0
        mean_wait_by_class = tuple(
            float(np.mean(waits)) if waits else 0.0
            for waits in exit_wait_minutes_by_class
        )
        p95_wait = float(np.percentile(wait_observations, 95)) if wait_observations else 0.0
        on_time = sum(
            call.intraperiod_wait_minutes
            + call.waited_periods * resource.minutes_per_period
            <= self.config.safety.target_wait_minutes
            for call in served_calls
        )
        outcome_count = len(served_calls) + len(abandoned_calls)
        service_level = on_time / outcome_count if outcome_count else 1.0
        priority_work = [call for call in calls if call.priority]
        priority_served = sum(call.priority for call in served_calls)
        priority_level = priority_served / len(priority_work) if priority_work else 1.0
        safety_violation = bool(
            service_level < self.config.safety.target_service_level
            or priority_level < self.config.safety.priority_service_level
            or p95_wait > self.config.safety.target_wait_minutes
        )

        costs = self.config.costs
        base_staff_cost = (
            resource.regular_agents * costs.regular_agent_period
            + resource.specialist_agents * costs.specialist_agent_period
        )
        augmentation_cost = temporary_agents * costs.augmentation_agent_period
        waiting_cost = len(next_waiting) * costs.waiting_per_call_period
        abandon_weights = (
            costs.abandonment_regular,
            costs.abandonment_specialist,
            costs.abandonment_callback,
        )
        abandonment_cost = sum(
            abandoned_counts[i] * abandon_weights[i] for i in range(3)
        )
        service_level_cost = costs.service_level_penalty if safety_violation else 0.0
        total_cost = (
            base_staff_cost + augmentation_cost + waiting_cost
            + abandonment_cost + service_level_cost
        )
        next_state = SystemState(
            episode_id=state.episode_id,
            period=state.period + 1,
            weekday=state.weekday,
            demand_state=draw.next_demand_state,
            waiting=tuple(next_waiting),
        )
        return PeriodResult(
            action=action,
            action_level=action_level,
            arrivals=arrivals,
            served=served_counts,
            served_priority=served_priority,
            served_nonpriority=served_nonpriority,  # type: ignore[arg-type]
            abandoned=abandoned_counts,
            next_state=next_state,
            regular_agents=regular_agents,
            specialist_agents=specialist_agents,
            temporary_agents=temporary_agents,
            specialist_minutes_cross_served=float(specialist_cross_used),
            base_staff_cost=float(base_staff_cost),
            augmentation_cost=float(augmentation_cost),
            waiting_cost=float(waiting_cost),
            abandonment_cost=float(abandonment_cost),
            service_level_cost=float(service_level_cost),
            total_cost=float(total_cost),
            service_level=float(service_level),
            mean_wait_minutes=mean_wait,
            mean_wait_by_class=mean_wait_by_class,  # type: ignore[arg-type]
            exit_wait_minutes_by_class=exit_wait_minutes_by_class,  # type: ignore[arg-type]
            p95_wait_minutes=p95_wait,
            safety_violation=safety_violation,
        )


def _serve(
    calls: list[QueuedCall],
    active: set[int],
    capacity: float,
    *,
    eligible: tuple[int, ...],
    class_order: tuple[int, ...],
    efficiency: float,
) -> tuple[float, list[int]]:
    order_rank = {service_class: rank for rank, service_class in enumerate(class_order)}
    candidates = sorted(
        (index for index in active if calls[index].service_class in eligible),
        key=lambda index: (
            not calls[index].priority,
            -calls[index].waited_periods,
            order_rank[calls[index].service_class],
            index,
        ),
    )
    served: list[int] = []
    for index in candidates:
        required = calls[index].service_minutes / efficiency
        if required <= capacity + 1e-12:
            capacity -= required
            active.remove(index)
            served.append(index)
    return max(capacity, 0.0), served


def _counts(calls: list[QueuedCall]) -> tuple[int, int, int]:
    return tuple(
        sum(call.service_class == service_class for call in calls)
        for service_class in range(3)
    )  # type: ignore[return-value]
