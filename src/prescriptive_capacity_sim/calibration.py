"""Empirical calibration of service and caller-patience primitives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .callcenter_data import QueueCall


@dataclass(frozen=True)
class EmpiricalCalibration:
    """Empirical distributions fitted without retaining raw identities."""

    service_seconds_by_type: dict[str, np.ndarray]
    pooled_service_seconds: np.ndarray
    patience_support_seconds: np.ndarray
    patience_survival: np.ndarray
    _patience_cdf: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        cdf = 1.0 - np.asarray(self.patience_survival, dtype=float)
        if len(cdf) == 0 or cdf[-1] <= 0:
            raise ValueError("patience curve must contain positive event mass")
        object.__setattr__(self, "_patience_cdf", cdf)

    def sample_service(self, call_type: str, rng: np.random.Generator) -> float:
        values = self.service_seconds_by_type.get(
            call_type, self.pooled_service_seconds
        )
        return float(rng.choice(values))

    def sample_patience(self, rng: np.random.Generator) -> float:
        # Kaplan-Meier probability masses are the drops in survival. If the
        # right tail is censored, condition on an observed finite event so the
        # discrete-event simulation remains bounded.
        draw = rng.random() * self._patience_cdf[-1]
        index = int(np.searchsorted(self._patience_cdf, draw, side="left"))
        return float(self.patience_support_seconds[index])


def fit_empirical_calibration(
    calls: Sequence[QueueCall],
) -> EmpiricalCalibration:
    """Fit service-time samples and a Kaplan-Meier caller-patience curve."""

    by_type: dict[str, list[float]] = defaultdict(list)
    pooled: list[float] = []
    durations: list[float] = []
    events: list[bool] = []

    for call in calls:
        if call.outcome == "AGENT" and call.observed_service_seconds is not None:
            if call.observed_service_seconds > 0:
                value = float(call.observed_service_seconds)
                by_type[call.call_type].append(value)
                pooled.append(value)
        durations.append(float(call.observed_wait_seconds))
        events.append(call.outcome == "HANG")

    if not pooled:
        raise ValueError("calibration requires a positive service observation")
    if not any(events):
        raise ValueError("calibration requires an abandonment observation")

    duration_array = np.asarray(durations, dtype=float)
    event_array = np.asarray(events, dtype=bool)
    if np.any(duration_array < 0):
        raise ValueError("calibration wait durations cannot be negative")

    survival = 1.0
    at_risk = len(duration_array)
    support: list[float] = []
    survival_values: list[float] = []
    for time in np.unique(duration_array):
        at_time = duration_array == time
        event_count = int(np.count_nonzero(event_array & at_time))
        total_count = int(np.count_nonzero(at_time))
        if event_count:
            survival *= 1.0 - event_count / at_risk
            support.append(float(time))
            survival_values.append(survival)
        at_risk -= total_count

    return EmpiricalCalibration(
        service_seconds_by_type={
            name: np.asarray(values, dtype=float) for name, values in by_type.items()
        },
        pooled_service_seconds=np.asarray(pooled, dtype=float),
        patience_support_seconds=np.asarray(support, dtype=float),
        patience_survival=np.asarray(survival_values, dtype=float),
    )
