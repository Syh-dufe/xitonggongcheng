"""Baseline policies and a small extension interface for new methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .behavior import HistoricalBehaviorPolicy
from .config import SimulationConfig
from .environment import CapacityEnvironment
from .oracle import evaluate_actions
from .state import ExogenousShock, SystemState


class Policy(Protocol):
    name: str

    def act(
        self,
        state: SystemState,
        shock: ExogenousShock,
        environment: CapacityEnvironment,
        rng: np.random.Generator,
    ) -> int: ...


@dataclass(frozen=True)
class FixedActionPolicy:
    action: int
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"fixed_{self.action}")

    def act(self, state, shock, environment, rng) -> int:
        if not 0 <= self.action < len(environment.config.behavior.action_levels):
            raise ValueError(f"invalid fixed action: {self.action}")
        return self.action


@dataclass(frozen=True)
class RandomPolicy:
    name: str = "random"

    def act(self, state, shock, environment, rng) -> int:
        return int(rng.integers(0, len(environment.config.behavior.action_levels)))


class HistoricalPolicy:
    name = "historical"

    def __init__(self, config: SimulationConfig):
        self._behavior = HistoricalBehaviorPolicy(config)

    def act(self, state, shock, environment, rng) -> int:
        action, _, _ = self._behavior.act(state, rng, shock.manager_alarm)
        return action


@dataclass(frozen=True)
class MyopicOraclePolicy:
    name: str = "myopic_oracle"

    def act(self, state, shock, environment, rng) -> int:
        return evaluate_actions(environment, state, shock).oracle_action


@dataclass(frozen=True)
class RollingOraclePolicy:
    horizon: int = 3
    name: str = "rolling_oracle"

    def act(self, state, shock, environment, rng) -> int:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        base_seed = int(rng.integers(0, 2**32 - 1))
        totals: list[float] = []
        for first_action in range(len(environment.config.behavior.action_levels)):
            local_rng = np.random.default_rng(base_seed)
            first = environment.transition(state, first_action, shock)
            total = first.total_cost
            rollout_state = first.next_state
            for _ in range(1, self.horizon):
                future_shock = environment.sample_exogenous(
                    rollout_state, local_rng, daily_effect=0.0
                )
                evaluation = evaluate_actions(environment, rollout_state, future_shock)
                chosen = evaluation.results[evaluation.oracle_action]
                total += chosen.total_cost
                rollout_state = chosen.next_state
            totals.append(total)
        return int(np.argmin(totals))


def default_policies(config: SimulationConfig) -> list[Policy]:
    return [
        HistoricalPolicy(config),
        RandomPolicy(),
        FixedActionPolicy(0),
        FixedActionPolicy(1),
        FixedActionPolicy(2),
        FixedActionPolicy(3),
        MyopicOraclePolicy(),
        RollingOraclePolicy(),
    ]

