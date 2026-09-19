"""Historical capacity decisions that induce observational selection bias."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .state import SystemState


class HistoricalBehaviorPolicy:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def probabilities(self, state: SystemState, manager_alarm: float) -> np.ndarray:
        normal, urgent, critical = state.backlog
        total = max(state.total_backlog, 1)
        risk = (
            -1.0
            + 0.025 * total
            + 0.9 * urgent / total
            + 1.8 * critical / total
            + 0.7 * state.disruption_state
            + self.config.behavior.hidden_confounding_strength * 0.8 * manager_alarm
        )
        target = 3.0 / (1.0 + np.exp(-risk))
        action_indices = np.arange(len(self.config.behavior.action_levels), dtype=float)
        scores = -self.config.behavior.bias_strength * (action_indices - target) ** 2
        scaled = scores / self.config.behavior.temperature
        scaled -= scaled.max()
        weights = np.exp(scaled)
        return weights / weights.sum()

    def act(
        self,
        state: SystemState,
        rng: np.random.Generator,
        manager_alarm: float,
    ) -> tuple[int, float, np.ndarray]:
        probabilities = self.probabilities(state, manager_alarm)
        action = int(rng.choice(len(probabilities), p=probabilities))
        return action, float(probabilities[action]), probabilities
