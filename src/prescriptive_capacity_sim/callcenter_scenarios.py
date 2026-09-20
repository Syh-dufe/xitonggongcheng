"""Semi-synthetic call-center replay and counterfactual staffing branches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from .calibration import EmpiricalCalibration
from .callcenter_data import QueueCall
from .callcenter_simulator import (
    CallCenterState,
    IntervalResult,
    SimulatedCall,
    simulate_interval,
)


@dataclass(frozen=True)
class ReplayDay:
    day: date
    interval_seconds: int
    arrivals_by_interval: tuple[tuple[SimulatedCall, ...], ...]


@dataclass(frozen=True)
class CostConfig:
    staffing_minute: float = 0.25
    waiting_second: float = 0.02
    abandonment: float = 30.0
    sla_violation: float = 5.0
    terminal_queue: float = 10.0


@dataclass(frozen=True)
class SemiSyntheticDay:
    observed: list[dict[str, object]]
    oracle: list[dict[str, object]]
    daily_metrics: dict[str, object]


@dataclass(frozen=True)
class StaffingBehaviorPolicy:
    """A stochastic historical policy with explicit overlap.

    ``hidden_pressure`` is available to the behavior policy but deliberately
    omitted from the factual log, allowing experiments with historical
    decision confounding.
    """

    actions: tuple[int, ...]
    temperature: float = 1.0
    queue_weight: float = 0.35
    arrival_weight: float = 0.08
    hidden_weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.actions or any(action < 0 for action in self.actions):
            raise ValueError("actions must contain nonnegative staffing levels")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def probabilities(
        self,
        *,
        queue_length: int,
        expected_arrivals: int,
        hidden_pressure: float,
    ) -> dict[int, float]:
        target = (
            min(self.actions)
            + self.queue_weight * queue_length
            + self.arrival_weight * expected_arrivals
            + self.hidden_weight * hidden_pressure
        )
        actions = np.asarray(self.actions, dtype=float)
        logits = -np.square(actions - target) / self.temperature
        logits = np.clip(logits - logits.max(), -700.0, 0.0)
        weights = np.exp(logits)
        probabilities = weights / weights.sum()
        return {
            action: float(probability)
            for action, probability in zip(self.actions, probabilities, strict=True)
        }

    def choose(
        self,
        *,
        queue_length: int,
        expected_arrivals: int,
        hidden_pressure: float,
        rng: np.random.Generator,
    ) -> tuple[int, float, dict[int, float]]:
        probabilities = self.probabilities(
            queue_length=queue_length,
            expected_arrivals=expected_arrivals,
            hidden_pressure=hidden_pressure,
        )
        values = np.asarray(self.actions, dtype=int)
        weights = np.asarray([probabilities[action] for action in self.actions])
        action = int(rng.choice(values, p=weights))
        return action, probabilities[action], probabilities


def build_replay_day(
    calls: Sequence[QueueCall],
    calibration: EmpiricalCalibration,
    day: date,
    *,
    seed: int,
    interval_seconds: int = 1800,
    intervals_per_day: int = 48,
) -> ReplayDay:
    """Replay real arrivals while drawing de-identified service and patience."""

    if interval_seconds <= 0 or intervals_per_day <= 0:
        raise ValueError("interval dimensions must be positive")
    rng = np.random.default_rng(seed)
    groups: list[list[SimulatedCall]] = [[] for _ in range(intervals_per_day)]
    horizon = interval_seconds * intervals_per_day
    selected = sorted(
        (call for call in calls if call.queue_entry.date() == day),
        key=lambda call: (call.queue_entry, call.call_key),
    )
    for call in selected:
        timestamp = call.queue_entry
        arrival_second = (
            timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        )
        if arrival_second >= horizon:
            continue
        simulated = SimulatedCall(
            call_key=call.call_key,
            arrival_second=float(arrival_second),
            service_seconds=calibration.sample_service(call.call_type, rng),
            patience_seconds=calibration.sample_patience(rng),
            priority=call.priority,
            call_type=call.call_type,
        )
        groups[arrival_second // interval_seconds].append(simulated)
    return ReplayDay(
        day=day,
        interval_seconds=interval_seconds,
        arrivals_by_interval=tuple(tuple(group) for group in groups),
    )


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _state_fingerprint(state: CallCenterState) -> str:
    return _digest(
        {
            "now": state.now_second,
            "waiting": [
                (item.call.call_key, item.abandonment_second, item.sequence)
                for item in state.waiting
            ],
            "busy": [
                (item.call.call_key, item.completion_second) for item in state.busy
            ],
        }
    )


def _arrival_fingerprint(arrivals: Sequence[SimulatedCall]) -> str:
    return _digest(
        [
            (
                call.call_key,
                call.arrival_second,
                call.service_seconds,
                call.patience_seconds,
                call.priority,
            )
            for call in arrivals
        ]
    )


def _cost(result: IntervalResult, costs: CostConfig, *, is_terminal: bool) -> float:
    resolved = result.started_service + result.abandoned
    total_wait = result.mean_wait_seconds * resolved
    terminal = result.ending_queue if is_terminal else 0
    return float(
        costs.staffing_minute * (result.staffed_seconds / 60.0)
        + costs.waiting_second * total_wait
        + costs.abandonment * result.abandoned
        + costs.sla_violation * result.sla_violations
        + costs.terminal_queue * terminal
    )


def generate_semisynthetic_day(
    replay: ReplayDay,
    *,
    behavior_policy: StaffingBehaviorPolicy,
    costs: CostConfig,
    seed: int,
) -> SemiSyntheticDay:
    """Generate a factual log and a separate counterfactual oracle table."""

    rng = np.random.default_rng(seed)
    state = CallCenterState.empty()
    observed: list[dict[str, object]] = []
    oracle: list[dict[str, object]] = []
    total_cost = 0.0
    total_abandoned = 0
    total_arrivals = 0

    for interval, arrivals in enumerate(replay.arrivals_by_interval):
        pre_state = state.clone()
        state_fingerprint = _state_fingerprint(pre_state)
        arrival_fingerprint = _arrival_fingerprint(arrivals)
        hidden_pressure = (
            float(np.mean([call.service_seconds for call in arrivals])) / 60.0
            if arrivals
            else 0.0
        )
        action, propensity, probabilities = behavior_policy.choose(
            queue_length=len(pre_state.waiting),
            expected_arrivals=len(arrivals),
            hidden_pressure=hidden_pressure,
            rng=rng,
        )
        branch_results: dict[int, IntervalResult] = {}
        branch_costs: dict[int, float] = {}
        terminal = interval == len(replay.arrivals_by_interval) - 1
        for candidate in behavior_policy.actions:
            result = simulate_interval(
                pre_state,
                arrivals,
                staffing=candidate,
                interval_seconds=replay.interval_seconds,
            )
            potential_cost = _cost(result, costs, is_terminal=terminal)
            branch_results[candidate] = result
            branch_costs[candidate] = potential_cost
            oracle.append(
                {
                    "day": replay.day.isoformat(),
                    "interval": interval,
                    "action": candidate,
                    "propensity": probabilities[candidate],
                    "selected": candidate == action,
                    "pre_queue": len(pre_state.waiting),
                    "pre_busy": len(pre_state.busy),
                    "arrivals": len(arrivals),
                    "pre_state_fingerprint": state_fingerprint,
                    "arrival_fingerprint": arrival_fingerprint,
                    "potential_cost": potential_cost,
                    "potential_abandoned": result.abandoned,
                    "potential_service_level": result.service_level,
                    "potential_ending_queue": result.ending_queue,
                }
            )

        realized = branch_results[action]
        realized_cost = branch_costs[action]
        observed.append(
            {
                "day": replay.day.isoformat(),
                "interval": interval,
                "pre_queue": len(pre_state.waiting),
                "pre_busy": len(pre_state.busy),
                "arrivals": len(arrivals),
                "action": action,
                "propensity": propensity,
                "realized_cost": realized_cost,
                "abandoned": realized.abandoned,
                "service_level": realized.service_level,
                "ending_queue": realized.ending_queue,
                "mean_wait_seconds": realized.mean_wait_seconds,
                "sla_violations": realized.sla_violations,
            }
        )
        state = realized.next_state
        total_cost += realized_cost
        total_abandoned += realized.abandoned
        total_arrivals += len(arrivals)

    return SemiSyntheticDay(
        observed=observed,
        oracle=oracle,
        daily_metrics={
            "day": replay.day.isoformat(),
            "intervals": len(replay.arrivals_by_interval),
            "arrivals": total_arrivals,
            "abandoned": total_abandoned,
            "total_cost": total_cost,
            "terminal_queue": len(state.waiting),
            "terminal_busy": len(state.busy),
        },
    )
