"""Configuration objects and validation for the simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DemandConfig:
    base_arrival_means: tuple[float, float, float] = (18.0, 6.0, 2.0)
    dispersion: float = 8.0
    daily_effect_sigma: float = 0.15
    peak_amplitude: float = 0.60
    shock_arrival_multipliers: tuple[float, float, float] = (1.0, 1.4, 2.0)
    shock_capacity_multipliers: tuple[float, float, float] = (1.0, 0.9, 0.65)
    shock_service_multipliers: tuple[float, float, float] = (1.0, 0.95, 0.8)
    transition_matrix: tuple[tuple[float, float, float], ...] = (
        (0.94, 0.05, 0.01),
        (0.25, 0.65, 0.10),
        (0.10, 0.35, 0.55),
    )


@dataclass(frozen=True)
class BehaviorConfig:
    bias_strength: float = 1.0
    temperature: float = 1.0
    hidden_confounding_strength: float = 0.0
    action_levels: tuple[float, float, float, float] = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class CostConfig:
    base_staff_unit_cost: float = 5.0
    senior_staff_unit_cost: float = 8.0
    temporary_unit_cost: float = 12.0
    backlog_costs: tuple[float, float, float] = (1.0, 3.0, 8.0)
    late_costs: tuple[float, float, float] = (3.0, 10.0, 30.0)
    safety_violation_penalty: float = 200.0
    overload_penalty: float = 15.0


@dataclass(frozen=True)
class SafetyConfig:
    deadlines: tuple[int, int, int] = (8, 4, 2)
    critical_service_threshold: float = 0.80
    critical_max_age: int = 3


@dataclass(frozen=True)
class SimulationConfig:
    periods_per_day: int = 48
    train_fraction: float = 0.70
    base_staff: int = 18
    senior_staff: int = 5
    service_rates: tuple[float, float, float] = (1.0, 1.4, 0.8)
    service_effort: tuple[float, float, float] = (1.0, 1.4, 1.8)
    max_age_bucket: int = 12
    demand: DemandConfig = DemandConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    costs: CostConfig = CostConfig()
    safety: SafetyConfig = SafetyConfig()

    @classmethod
    def default(cls) -> "SimulationConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("configuration root must be a mapping")
        merged = _deep_merge(asdict(cls.default()), raw)
        cfg = cls(
            periods_per_day=int(merged["periods_per_day"]),
            train_fraction=float(merged["train_fraction"]),
            base_staff=int(merged["base_staff"]),
            senior_staff=int(merged["senior_staff"]),
            service_rates=_tuple_of(merged["service_rates"], float, 3),
            service_effort=_tuple_of(merged["service_effort"], float, 3),
            max_age_bucket=int(merged["max_age_bucket"]),
            demand=DemandConfig(
                base_arrival_means=_tuple_of(merged["demand"]["base_arrival_means"], float, 3),
                dispersion=float(merged["demand"]["dispersion"]),
                daily_effect_sigma=float(merged["demand"]["daily_effect_sigma"]),
                peak_amplitude=float(merged["demand"]["peak_amplitude"]),
                shock_arrival_multipliers=_tuple_of(merged["demand"]["shock_arrival_multipliers"], float, 3),
                shock_capacity_multipliers=_tuple_of(merged["demand"]["shock_capacity_multipliers"], float, 3),
                shock_service_multipliers=_tuple_of(merged["demand"]["shock_service_multipliers"], float, 3),
                transition_matrix=tuple(
                    _tuple_of(row, float, 3) for row in merged["demand"]["transition_matrix"]
                ),
            ),
            behavior=BehaviorConfig(
                bias_strength=float(merged["behavior"]["bias_strength"]),
                temperature=float(merged["behavior"]["temperature"]),
                hidden_confounding_strength=float(merged["behavior"]["hidden_confounding_strength"]),
                action_levels=_tuple_of(merged["behavior"]["action_levels"], float, 4),
            ),
            costs=CostConfig(
                base_staff_unit_cost=float(merged["costs"]["base_staff_unit_cost"]),
                senior_staff_unit_cost=float(merged["costs"]["senior_staff_unit_cost"]),
                temporary_unit_cost=float(merged["costs"]["temporary_unit_cost"]),
                backlog_costs=_tuple_of(merged["costs"]["backlog_costs"], float, 3),
                late_costs=_tuple_of(merged["costs"]["late_costs"], float, 3),
                safety_violation_penalty=float(merged["costs"]["safety_violation_penalty"]),
                overload_penalty=float(merged["costs"]["overload_penalty"]),
            ),
            safety=SafetyConfig(
                deadlines=_tuple_of(merged["safety"]["deadlines"], int, 3),
                critical_service_threshold=float(merged["safety"]["critical_service_threshold"]),
                critical_max_age=int(merged["safety"]["critical_max_age"]),
            ),
        )
        cfg.validate()
        return cfg

    def with_overrides(self, **overrides: Any) -> "SimulationConfig":
        """Return a validated copy with top-level or nested dataclass overrides."""
        allowed = {"periods_per_day", "train_fraction", "base_staff", "senior_staff", "service_rates", "service_effort", "max_age_bucket", "demand", "behavior", "costs", "safety"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        prepared: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, Mapping) and key in {"demand", "behavior", "costs", "safety"}:
                prepared[key] = replace(current, **value)
            else:
                prepared[key] = value
        result = replace(self, **prepared)
        result.validate()
        return result

    def validate(self) -> None:
        if self.periods_per_day <= 0 or self.base_staff < 0 or self.senior_staff < 0:
            raise ValueError("period and staffing values must be nonnegative, with periods positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be strictly between zero and one")
        if self.max_age_bucket < max(self.safety.deadlines):
            raise ValueError("max_age_bucket must cover every deadline")
        if any(value <= 0 for value in (*self.service_rates, *self.service_effort)):
            raise ValueError("service rates and effort values must be positive")
        matrix = self.demand.transition_matrix
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("transition_matrix must be 3 by 3")
        if any(value < 0 for row in matrix for value in row):
            raise ValueError("transition probabilities must be nonnegative")
        if any(abs(sum(row) - 1.0) > 1e-9 for row in matrix):
            raise ValueError("each transition row must sum to one")
        if self.demand.dispersion <= 0 or any(v <= 0 for v in self.demand.base_arrival_means):
            raise ValueError("arrival means and dispersion must be positive")
        if self.behavior.temperature <= 0 or self.behavior.bias_strength < 0:
            raise ValueError("behavior temperature must be positive and bias nonnegative")
        actions = self.behavior.action_levels
        if actions[0] < 0 or any(right <= left for left, right in zip(actions, actions[1:])):
            raise ValueError("action levels must be nonnegative and strictly increasing")
        cost_values = (
            self.costs.base_staff_unit_cost,
            self.costs.senior_staff_unit_cost,
            self.costs.temporary_unit_cost,
            self.costs.safety_violation_penalty,
            self.costs.overload_penalty,
            *self.costs.backlog_costs,
            *self.costs.late_costs,
        )
        if any(value < 0 for value in cost_values):
            raise ValueError("costs must be nonnegative")
        if not 0 <= self.safety.critical_service_threshold <= 1:
            raise ValueError("critical_service_threshold must be in [0, 1]")


def _tuple_of(values: Any, cast, expected: int) -> tuple:
    result = tuple(cast(value) for value in values)
    if len(result) != expected:
        raise ValueError(f"expected {expected} values, received {len(result)}")
    return result


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"unknown configuration key: {key}")
        if isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
