"""Historical observational log generation and isolated oracle tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .behavior import HistoricalBehaviorPolicy
from .config import SimulationConfig
from .environment import CapacityEnvironment
from .oracle import evaluate_actions
from .state import SystemState


@dataclass(frozen=True)
class GeneratedDataset:
    observed: pd.DataFrame
    oracle: pd.DataFrame
    episodes: pd.DataFrame


def generate_dataset(
    config: SimulationConfig,
    days: int,
    seed: int,
) -> GeneratedDataset:
    if days <= 0:
        raise ValueError("days must be positive")
    environment = CapacityEnvironment(config)
    behavior = HistoricalBehaviorPolicy(config)
    observed_rows: list[dict] = []
    oracle_rows: list[dict] = []
    episode_rows: list[dict] = []
    train_days = (
        max(1, min(days - 1, int(np.floor(days * config.train_fraction))))
        if days > 1
        else 1
    )

    for episode_id in range(days):
        split = "train" if episode_id < train_days else "test"
        env_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 0]))
        policy_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 1]))
        daily_effect = float(env_rng.normal(0.0, config.demand.daily_effect_sigma))
        state = environment.reset(episode_id)
        episode_cost = 0.0
        episode_completed = 0
        episode_arrivals = 0
        episode_violations = 0
        episode_propensities: list[float] = []

        for _ in range(config.periods_per_day):
            shock = environment.sample_exogenous(state, env_rng, daily_effect)
            action, propensity, probabilities = behavior.act(
                state, policy_rng, shock.manager_alarm
            )
            counterfactuals = evaluate_actions(environment, state, shock)
            realized = counterfactuals.results[action]
            observed_rows.append(
                _observed_row(state, realized, propensity, probabilities, split)
            )
            oracle_rows.append(
                _oracle_row(state, counterfactuals.results, counterfactuals.oracle_action, split)
            )
            episode_cost += realized.total_cost
            episode_completed += sum(realized.completed)
            episode_arrivals += sum(realized.arrivals)
            episode_violations += int(realized.safety_violation)
            episode_propensities.append(propensity)
            state = realized.next_state

        inverse_weights = np.reciprocal(np.asarray(episode_propensities))
        episode_rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "seed": seed,
                "daily_effect": daily_effect,
                "total_cost": episode_cost,
                "total_arrivals": episode_arrivals,
                "total_completed": episode_completed,
                "final_backlog": state.total_backlog,
                "safety_violations": episode_violations,
                "min_propensity": min(episode_propensities),
                "propensity_ess": float(
                    inverse_weights.sum() ** 2 / np.square(inverse_weights).sum()
                ),
            }
        )

    return GeneratedDataset(
        observed=pd.DataFrame(observed_rows),
        oracle=pd.DataFrame(oracle_rows),
        episodes=pd.DataFrame(episode_rows),
    )


def _observed_row(
    state: SystemState,
    result,
    propensity: float,
    probabilities: np.ndarray,
    split: str,
) -> dict:
    normal, urgent, critical = state.backlog
    max_normal, max_urgent, max_critical = state.max_ages
    next_normal, next_urgent, next_critical = result.next_state.backlog
    row = {
        "episode_id": state.episode_id,
        "split": split,
        "period": state.period,
        "disruption_state": state.disruption_state,
        "backlog_normal": normal,
        "backlog_urgent": urgent,
        "backlog_critical": critical,
        "max_age_normal": max_normal,
        "max_age_urgent": max_urgent,
        "max_age_critical": max_critical,
        "action": result.action,
        "action_level": result.action_level,
        "propensity": propensity,
        "arrival_normal": result.arrivals[0],
        "arrival_urgent": result.arrivals[1],
        "arrival_critical": result.arrivals[2],
        "completed_normal": result.completed[0],
        "completed_urgent": result.completed[1],
        "completed_critical": result.completed[2],
        "next_backlog_normal": next_normal,
        "next_backlog_urgent": next_urgent,
        "next_backlog_critical": next_critical,
        "staffing_cost": result.staffing_cost,
        "backlog_cost": result.backlog_cost,
        "late_cost": result.late_cost,
        "overload_cost": result.overload_cost,
        "safety_cost": result.safety_cost,
        "total_cost": result.total_cost,
        "temporary_staff": result.temporary_staff,
        "service_rate": result.service_rate,
        "safety_violation": result.safety_violation,
    }
    for action, probability in enumerate(probabilities):
        row[f"prob_action_{action}"] = float(probability)
    return row


def _oracle_row(
    state: SystemState,
    results: tuple,
    oracle_action: int,
    split: str,
) -> dict:
    row = {
        "episode_id": state.episode_id,
        "split": split,
        "period": state.period,
        "oracle_action": oracle_action,
    }
    for action, result in enumerate(results):
        row[f"potential_cost_a{action}"] = result.total_cost
        row[f"potential_service_rate_a{action}"] = result.service_rate
        row[f"potential_safety_violation_a{action}"] = result.safety_violation
        row[f"potential_next_backlog_a{action}"] = result.next_state.total_backlog
    return row
