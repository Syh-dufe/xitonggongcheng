"""Factual-only simulation helpers for capacity-scenario validation.

This module deliberately runs exactly one historical action per period.  It does
not generate potential outcomes or use an oracle evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from .behavior import HistoricalBehaviorPolicy
from .config import SimulationConfig
from .demand import CalibratedDemandProcess
from .environment import CallCenterEnvironment
from .state import PeriodResult, SystemState


WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


SELECTION_WEIGHTS = {
    "abandonment_rate_error": 0.4,
    "mean_wait_relative_error": 0.4,
    "overall_p90_wait_relative_error": 0.2,
}
RESOURCE_COLUMNS = (
    "regular_agents",
    "specialist_agents",
    "cross_skill_efficiency",
)


@dataclass(frozen=True)
class ResourceCandidate:
    """A transparent, semisynthetic resource scenario."""

    name: str
    regular_agents: int
    specialist_agents: int
    cross_skill_efficiency: float = 0.75

    def resource_overrides(self) -> dict[str, int | float]:
        return {
            "regular_agents": self.regular_agents,
            "specialist_agents": self.specialist_agents,
            "cross_skill_efficiency": self.cross_skill_efficiency,
        }


def load_candidates(path: str | Path) -> tuple[ResourceCandidate, ...]:
    """Load and validate a nonempty, uniquely named candidate YAML grid."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("candidate grid root must be a mapping")
    rows = raw.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a nonempty list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("each candidate must be a mapping")
    supported_fields = {
        "name", "regular_agents", "specialist_agents", "cross_skill_efficiency",
    }
    unsupported_fields = {
        field for row in rows for field in row if field not in supported_fields
    }
    if unsupported_fields:
        raise ValueError(
            "unsupported candidate fields: " + ", ".join(sorted(unsupported_fields))
        )

    candidates = tuple(ResourceCandidate(**row) for row in rows)
    if len({item.name for item in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        SimulationConfig.default().with_overrides(
            resources=candidate.resource_overrides()
        )
    return candidates


def summarize_candidates(runs: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by their predeclared held-out operational error score."""

    summary_columns = [
        "candidate",
        "selection_score",
        "selection_score_std",
        "seed_count",
        "rank",
    ]
    if set(RESOURCE_COLUMNS).issubset(runs.columns):
        summary_columns.extend(RESOURCE_COLUMNS)
    if runs.empty:
        return pd.DataFrame(columns=summary_columns)

    required_columns = {"candidate", "seed", "metric", "service_class", "error"}
    missing_columns = required_columns.difference(runs.columns)
    if missing_columns:
        raise ValueError(
            "runs missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if set(RESOURCE_COLUMNS).issubset(runs.columns):
        resource_variation = runs.groupby("candidate")[
            list(RESOURCE_COLUMNS)
        ].nunique(dropna=False)
        if resource_variation.gt(1).any().any():
            raise ValueError("resource values must be fixed per candidate")

    operational = runs.loc[runs["metric"].isin(SELECTION_WEIGHTS)]
    metric_averages = operational.groupby(
        ["candidate", "seed", "metric"], as_index=False, sort=False
    )["error"].mean()
    score_inputs = metric_averages.pivot(
        index=["candidate", "seed"], columns="metric", values="error"
    ).reindex(columns=SELECTION_WEIGHTS)
    if score_inputs.isna().any().any():
        raise ValueError("each candidate and seed must contain every scored metric")
    scores = score_inputs.mul(pd.Series(SELECTION_WEIGHTS)).sum(axis=1).rename(
        "selection_score"
    )
    summary = scores.groupby(level="candidate").agg(
        selection_score="mean",
        selection_score_std=lambda values: values.std(ddof=0),
        seed_count="count",
    ).reset_index()
    summary["seed_count"] = summary["seed_count"].astype(int)

    if set(RESOURCE_COLUMNS).issubset(runs.columns):
        resource_values = runs.groupby("candidate", as_index=False)[
            list(RESOURCE_COLUMNS)
        ].first()
        summary = summary.merge(resource_values, on="candidate", how="left")

    summary = summary.sort_values(
        ["selection_score", "candidate"], kind="stable"
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
    return summary[summary_columns]


def simulate_historical_periods(
    config: SimulationConfig,
    parameters: dict | str | Path,
    *,
    days: int,
    seed: int,
    weekdays: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Simulate factual historical actions and their realized period outcomes."""

    if days <= 0:
        raise ValueError("days must be positive")
    parameter_values = _load_parameter_values(parameters)
    _validate_time_grid(config, parameter_values)
    episode_weekdays = _episode_weekdays(days, weekdays)

    demand = CalibratedDemandProcess(
        parameter_values,
        hidden_confounding_strength=config.behavior.hidden_confounding_strength,
        demand_shock_scale=config.demand_shock_scale,
    )
    environment = CallCenterEnvironment(config)
    behavior = HistoricalBehaviorPolicy(config)
    rows: list[dict[str, int | float | bool | str]] = []

    for episode_id in range(days):
        weekday = episode_weekdays[episode_id]
        environment_rng = np.random.default_rng(
            np.random.SeedSequence([seed, episode_id, 0])
        )
        behavior_rng = np.random.default_rng(
            np.random.SeedSequence([seed, episode_id, 1])
        )
        state = environment.reset(episode_id, weekday)
        for _ in range(config.periods_per_day):
            draw = demand.sample(state, environment_rng)
            action, propensity, _ = behavior.act(
                state, draw.manager_alarm, behavior_rng
            )
            result = environment.transition(state, action, draw)
            rows.append(_factual_row(state, result, propensity))
            state = result.next_state

    return pd.DataFrame(rows)


def _load_parameter_values(parameters: dict | str | Path) -> dict:
    if isinstance(parameters, (str, Path)):
        parameters = json.loads(Path(parameters).read_text(encoding="utf-8"))
    if not isinstance(parameters, dict):
        raise ValueError("calibration parameters must be a mapping")
    return parameters


def _validate_time_grid(config: SimulationConfig, parameters: dict) -> None:
    if parameters.get("periods_per_day") != config.periods_per_day:
        raise ValueError(
            "config periods_per_day must match calibration parameters periods_per_day"
        )
    if parameters.get("period_minutes") != config.resources.minutes_per_period:
        raise ValueError(
            "config minutes_per_period must match calibration parameters period_minutes"
        )


def _episode_weekdays(
    days: int,
    weekdays: Sequence[str] | None,
) -> tuple[str, ...]:
    if weekdays is None:
        return tuple(WEEKDAYS[episode_id % len(WEEKDAYS)] for episode_id in range(days))
    if isinstance(weekdays, str) or len(weekdays) != days:
        raise ValueError("weekdays length must equal days")
    return tuple(str(weekday).lower() for weekday in weekdays)


def _factual_row(
    state: SystemState,
    result: PeriodResult,
    propensity: float,
) -> dict[str, int | float | bool | str]:
    queue_flow_error = result.next_state.total_queue - (
        state.total_queue
        + sum(result.arrivals)
        - sum(result.served)
        - sum(result.abandoned)
    )
    return {
        "episode_id": state.episode_id,
        "weekday": state.weekday,
        "period": state.period,
        "demand_state": state.demand_state,
        "queue_regular": state.queue_counts[0],
        "queue_specialist": state.queue_counts[1],
        "queue_callback_special": state.queue_counts[2],
        "queue_priority": state.priority_count,
        "max_waited_periods": state.max_waited_periods,
        "action": result.action,
        "action_level": result.action_level,
        "propensity": propensity,
        "arrival_regular": result.arrivals[0],
        "arrival_specialist": result.arrivals[1],
        "arrival_callback_special": result.arrivals[2],
        "served_regular": result.served[0],
        "served_specialist": result.served[1],
        "served_callback_special": result.served[2],
        "abandoned_regular": result.abandoned[0],
        "abandoned_specialist": result.abandoned[1],
        "abandoned_callback_special": result.abandoned[2],
        "next_queue": result.next_state.total_queue,
        "queue_flow_error": float(queue_flow_error),
        "regular_agents": result.regular_agents,
        "specialist_agents": result.specialist_agents,
        "temporary_agents": result.temporary_agents,
        "specialist_minutes_cross_served": result.specialist_minutes_cross_served,
        "total_cost": result.total_cost,
        "service_level": result.service_level,
        "mean_wait_minutes": result.mean_wait_minutes,
        "mean_wait_regular": result.mean_wait_by_class[0],
        "mean_wait_specialist": result.mean_wait_by_class[1],
        "mean_wait_callback_special": result.mean_wait_by_class[2],
        "exit_wait_regular": result.exit_wait_minutes_by_class[0],
        "exit_wait_specialist": result.exit_wait_minutes_by_class[1],
        "exit_wait_callback_special": result.exit_wait_minutes_by_class[2],
        "p95_wait_minutes": result.p95_wait_minutes,
        "safety_violation": result.safety_violation,
    }
