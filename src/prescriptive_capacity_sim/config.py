"""Validated configuration for the semisynthetic call-center simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ResourceConfig:
    regular_agents: int = 8
    specialist_agents: int = 5
    supervisor_emergency_agents: int = 1
    cross_skill_efficiency: float = 0.75
    callback_regular_share: float = 0.5
    minutes_per_period: int = 30


@dataclass(frozen=True)
class BehaviorConfig:
    bias_strength: float = 1.0
    temperature: float = 1.0
    hidden_confounding_strength: float = 0.0
    action_levels: tuple[float, float, float, float] = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class CostConfig:
    regular_agent_period: float = 5.0
    specialist_agent_period: float = 8.0
    augmentation_agent_period: float = 12.0
    waiting_per_call_period: float = 1.0
    abandonment_regular: float = 8.0
    abandonment_specialist: float = 15.0
    abandonment_callback: float = 20.0
    service_level_penalty: float = 200.0


@dataclass(frozen=True)
class SafetyConfig:
    target_wait_minutes: float = 2.0
    target_service_level: float = 0.80
    priority_service_level: float = 0.90


@dataclass(frozen=True)
class SimulationConfig:
    calibration_path: Path = Path("data/processed/calibration_parameters.json")
    periods_per_day: int = 34
    train_fraction: float = 0.70
    resources: ResourceConfig = ResourceConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    costs: CostConfig = CostConfig()
    safety: SafetyConfig = SafetyConfig()
    demand_state_multipliers: tuple[float, float, float] = (1.0, 1.25, 1.65)
    capacity_state_multipliers: tuple[float, float, float] = (1.0, 0.9, 0.75)

    @classmethod
    def default(cls) -> "SimulationConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("configuration root must be a mapping")
        merged = _deep_merge(asdict(cls.default()), raw)
        cfg = cls(
            calibration_path=Path(merged["calibration_path"]),
            periods_per_day=int(merged["periods_per_day"]),
            train_fraction=float(merged["train_fraction"]),
            resources=ResourceConfig(**merged["resources"]),
            behavior=BehaviorConfig(
                bias_strength=float(merged["behavior"]["bias_strength"]),
                temperature=float(merged["behavior"]["temperature"]),
                hidden_confounding_strength=float(
                    merged["behavior"]["hidden_confounding_strength"]
                ),
                action_levels=tuple(float(v) for v in merged["behavior"]["action_levels"]),
            ),
            costs=CostConfig(**merged["costs"]),
            safety=SafetyConfig(**merged["safety"]),
            demand_state_multipliers=tuple(
                float(v) for v in merged["demand_state_multipliers"]
            ),
            capacity_state_multipliers=tuple(
                float(v) for v in merged["capacity_state_multipliers"]
            ),
        )
        cfg.validate()
        return cfg

    def with_overrides(self, **overrides: Any) -> "SimulationConfig":
        allowed = {
            "calibration_path", "periods_per_day", "train_fraction", "resources",
            "behavior", "costs", "safety", "demand_state_multipliers",
            "capacity_state_multipliers",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, Mapping) and key in {"resources", "behavior", "costs", "safety"}:
                if key == "behavior" and "action_levels" in value:
                    value = dict(value)
                    value["action_levels"] = tuple(value["action_levels"])
                values[key] = replace(current, **value)
            elif key == "calibration_path":
                values[key] = Path(value)
            else:
                values[key] = value
        cfg = replace(self, **values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.periods_per_day <= 0:
            raise ValueError("periods_per_day must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        r = self.resources
        if r.regular_agents < 0 or r.specialist_agents < 0:
            raise ValueError("agent counts must be nonnegative")
        if not 0 < r.cross_skill_efficiency <= 1:
            raise ValueError("cross_skill_efficiency must be in (0, 1]")
        if not 0 <= r.callback_regular_share <= 1:
            raise ValueError("callback_regular_share must be in [0, 1]")
        if r.minutes_per_period <= 0:
            raise ValueError("minutes_per_period must be positive")
        actions = self.behavior.action_levels
        if len(actions) != 4 or actions[0] < 0 or any(
            b <= a for a, b in zip(actions, actions[1:])
        ):
            raise ValueError("four strictly increasing nonnegative actions are required")
        if self.behavior.temperature <= 0 or self.behavior.bias_strength < 0:
            raise ValueError("invalid behavior configuration")
        if len(self.demand_state_multipliers) != 3 or len(self.capacity_state_multipliers) != 3:
            raise ValueError("three demand and capacity states are required")
        if any(v <= 0 for v in (*self.demand_state_multipliers, *self.capacity_state_multipliers)):
            raise ValueError("state multipliers must be positive")
        if any(v < 0 for v in asdict(self.costs).values()):
            raise ValueError("costs must be nonnegative")
        if not 0 <= self.safety.target_service_level <= 1:
            raise ValueError("target_service_level must be in [0, 1]")

    def validate_for_simulation(self) -> None:
        self.validate()
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"calibration file not found: {self.calibration_path}")


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
