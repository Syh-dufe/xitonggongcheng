"""Stochastic call arrivals sampled from calibrated operational distributions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .state import ExogenousDraw, QueuedCall, SERVICE_CLASSES, SystemState


class CalibratedDemandProcess:
    def __init__(
        self,
        parameters: dict | str | Path,
        *,
        hidden_confounding_strength: float = 0.0,
    ):
        if isinstance(parameters, (str, Path)):
            parameters = json.loads(Path(parameters).read_text(encoding="utf-8"))
        self.parameters = parameters
        self.hidden_confounding_strength = float(hidden_confounding_strength)
        self.period_minutes = int(parameters["period_minutes"])
        self.periods_per_day = int(parameters.get("periods_per_day", 48))

    def expected_arrivals(
        self,
        period: int,
        weekday: str,
        demand_state: int,
        manager_alarm: float = 0.0,
    ) -> tuple[float, float, float]:
        multipliers = self.parameters["disruption"].get(
            "arrival_multipliers", [1.0, 1.25, 1.65]
        )
        state_multiplier = float(multipliers[demand_state])
        hidden = math.exp(
            0.15 * self.hidden_confounding_strength * float(manager_alarm)
        )
        values = []
        key = f"{weekday.lower()}:{period % self.periods_per_day}"
        for service_class in SERVICE_CLASSES:
            record = self.parameters["arrival"][service_class].get(key)
            if record is None:
                records = self.parameters["arrival"][service_class].values()
                mean = float(np.mean([float(item["mean"]) for item in records]))
            else:
                mean = float(record["mean"])
            values.append(max(mean * state_multiplier * hidden, 1e-9))
        return tuple(values)  # type: ignore[return-value]

    def sample(
        self,
        state: SystemState,
        rng: np.random.Generator,
    ) -> ExogenousDraw:
        alarm = float(rng.normal())
        means = self.expected_arrivals(
            state.period, state.weekday, state.demand_state, alarm
        )
        new_calls: list[QueuedCall] = []
        key = f"{state.weekday}:{state.period % self.periods_per_day}"
        for class_index, (service_class, mean) in enumerate(zip(SERVICE_CLASSES, means)):
            record = self.parameters["arrival"][service_class].get(key, {})
            dispersion = float(record.get("dispersion", 1_000_000.0))
            probability = dispersion / (dispersion + mean)
            count = int(rng.negative_binomial(dispersion, probability))
            priority_rate = float(self.parameters["priority_rate"].get(service_class, 0.0))
            service_grid = np.asarray(
                self.parameters["empirical"]["service_seconds"][service_class], dtype=float
            )
            patience_grid = np.asarray(
                self.parameters["empirical"]["patience_seconds"][service_class], dtype=float
            )
            queue_grid = np.asarray(
                self.parameters["empirical"].get("queue_seconds", {}).get(
                    service_class, [0.0]
                ),
                dtype=float,
            )
            if service_grid.size == 0 or patience_grid.size == 0:
                raise ValueError(f"empty empirical distribution for {service_class}")
            for _ in range(count):
                service_minutes = max(float(rng.choice(service_grid)) / 60.0, 1.0 / 60.0)
                patience_periods = max(
                    1,
                    int(math.ceil(float(rng.choice(patience_grid)) / (60 * self.period_minutes))),
                )
                new_calls.append(QueuedCall(
                    service_class=class_index,
                    priority=bool(rng.random() < priority_rate),
                    service_minutes=service_minutes,
                    patience_periods=patience_periods,
                    intraperiod_wait_minutes=max(float(rng.choice(queue_grid)) / 60.0, 0.0),
                ))
        transition = np.asarray(
            self.parameters["disruption"]["transition_matrix"][state.demand_state],
            dtype=float,
        )
        transition /= transition.sum()
        next_state = int(rng.choice(3, p=transition))
        hidden = self.hidden_confounding_strength * alarm
        return ExogenousDraw(
            new_calls=tuple(new_calls),
            next_demand_state=next_state,
            regular_availability=max(0.5, float(rng.lognormal(-0.01 - 0.03 * hidden, 0.04))),
            specialist_availability=max(0.5, float(rng.lognormal(-0.01 - 0.04 * hidden, 0.05))),
            service_efficiency=max(0.5, float(rng.lognormal(-0.01 - 0.04 * hidden, 0.06))),
            manager_alarm=alarm,
        )
