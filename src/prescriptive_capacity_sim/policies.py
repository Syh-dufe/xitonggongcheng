"""Reference policies for paired out-of-sample evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .behavior import HistoricalBehaviorPolicy
from .config import SimulationConfig
from .environment import CallCenterEnvironment
from .oracle import evaluate_actions
from .state import ExogenousDraw, SystemState


class Policy(Protocol):
    name: str

    def act(
        self,
        state: SystemState,
        draw: ExogenousDraw,
        environment: CallCenterEnvironment,
        rng: np.random.Generator,
    ) -> int: ...


class HistoricalPolicy:
    name = "historical"

    def __init__(self, config: SimulationConfig):
        self.behavior = HistoricalBehaviorPolicy(config)

    def act(self, state, draw, environment, rng) -> int:
        action, _, _ = self.behavior.act(state, draw.manager_alarm, rng)
        return action


@dataclass(frozen=True)
class RandomPolicy:
    name: str = "random"

    def act(self, state, draw, environment, rng) -> int:
        return int(rng.integers(0, 4))


@dataclass(frozen=True)
class FixedActionPolicy:
    action: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", f"fixed_{self.action}")

    def act(self, state, draw, environment, rng) -> int:
        return self.action


@dataclass(frozen=True)
class MyopicOraclePolicy:
    name: str = "myopic_oracle"

    def act(self, state, draw, environment, rng) -> int:
        return evaluate_actions(environment, state, draw).oracle_action


@dataclass(frozen=True)
class RollingOraclePolicy:
    queue_penalty: float = 2.0
    name: str = "rolling_oracle"

    def act(self, state, draw, environment, rng) -> int:
        evaluation = evaluate_actions(environment, state, draw)
        return min(
            range(4),
            key=lambda action: (
                evaluation.results[action].total_cost
                + self.queue_penalty * evaluation.results[action].next_state.total_queue
            ),
        )


def default_policies(config: SimulationConfig) -> list[Policy]:
    return [
        HistoricalPolicy(config), RandomPolicy(),
        *(FixedActionPolicy(action) for action in range(4)),
        MyopicOraclePolicy(), RollingOraclePolicy(),
    ]
