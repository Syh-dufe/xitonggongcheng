"""Biased historical staffing behavior used to create observational logs."""

from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .state import SystemState


class HistoricalBehaviorPolicy:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def probabilities(self, state: SystemState, manager_alarm: float) -> np.ndarray:
        counts = state.queue_counts
        total = max(state.total_queue, 1)
        specialist_share = counts[1] / total
        callback_share = counts[2] / total
        priority_share = state.priority_count / total
        phase = 2 * math.pi * state.period / max(self.config.periods_per_day, 1)
        risk = (
            -1.1
            + 0.12 * state.total_queue
            + 0.8 * specialist_share
            + 1.1 * callback_share
            + 1.0 * priority_share
            + 0.6 * state.demand_state
            + 0.25 * math.sin(phase)
            + 0.8 * self.config.behavior.hidden_confounding_strength * manager_alarm
        )
        target = 3.0 / (1.0 + math.exp(-risk))
        indices = np.arange(4, dtype=float)
        scores = -self.config.behavior.bias_strength * np.square(indices - target)
        scores /= self.config.behavior.temperature
        scores -= scores.max()
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum()

    def act(
        self,
        state: SystemState,
        manager_alarm: float,
        rng: np.random.Generator,
    ) -> tuple[int, float, np.ndarray]:
        probabilities = self.probabilities(state, manager_alarm)
        action = int(rng.choice(4, p=probabilities))
        return action, float(probabilities[action]), probabilities
