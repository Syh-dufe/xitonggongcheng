"""Stochastic arrivals and disruption transitions."""

from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .state import ExogenousShock, SystemState


class DemandProcess:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def expected_arrivals(
        self,
        period: int,
        disruption_state: int,
        daily_effect: float,
        manager_alarm: float,
    ) -> tuple[float, float, float]:
        phase = 2.0 * math.pi * (period % self.config.periods_per_day) / self.config.periods_per_day
        time_multiplier = 1.0 + self.config.demand.peak_amplitude * (math.sin(phase - math.pi / 2.0) + 1.0) / 2.0
        shock_multiplier = self.config.demand.shock_arrival_multipliers[disruption_state]
        hidden_multiplier = math.exp(
            0.20 * self.config.behavior.hidden_confounding_strength * manager_alarm
        )
        return tuple(
            mean * time_multiplier * shock_multiplier * math.exp(daily_effect) * hidden_multiplier
            for mean in self.config.demand.base_arrival_means
        )  # type: ignore[return-value]

    def sample_shock(
        self,
        state: SystemState,
        rng: np.random.Generator,
        daily_effect: float,
    ) -> ExogenousShock:
        alarm = float(rng.normal())
        means = self.expected_arrivals(
            state.period,
            state.disruption_state,
            daily_effect,
            alarm,
        )
        dispersion = self.config.demand.dispersion
        arrivals = tuple(
            int(rng.negative_binomial(dispersion, dispersion / (dispersion + mean)))
            for mean in means
        )
        probabilities = self.config.demand.transition_matrix[state.disruption_state]
        next_state = int(rng.choice(3, p=probabilities))
        sigma = 0.08
        service_multiplier = float(rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma))
        service_multiplier *= math.exp(
            -0.08 * self.config.behavior.hidden_confounding_strength * alarm
        )
        return ExogenousShock(
            arrivals=arrivals,  # type: ignore[arg-type]
            next_disruption_state=next_state,
            service_multiplier=service_multiplier,
            manager_alarm=alarm,
        )

