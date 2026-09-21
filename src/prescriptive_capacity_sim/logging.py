"""Generate factual historical logs and isolated oracle counterfactual tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .behavior import HistoricalBehaviorPolicy
from .config import SimulationConfig
from .demand import CalibratedDemandProcess
from .environment import CallCenterEnvironment
from .oracle import evaluate_actions
from .state import PeriodResult, SystemState


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class GeneratedDataset:
    observed: pd.DataFrame
    oracle: pd.DataFrame
    episodes: pd.DataFrame
    metadata: dict


def generate_dataset(
    config: SimulationConfig,
    parameters: dict | str,
    *,
    days: int,
    seed: int,
) -> GeneratedDataset:
    if days <= 0:
        raise ValueError("days must be positive")
    demand = CalibratedDemandProcess(
        parameters,
        hidden_confounding_strength=config.behavior.hidden_confounding_strength,
    )
    environment = CallCenterEnvironment(config)
    behavior = HistoricalBehaviorPolicy(config)
    observed_rows: list[dict] = []
    oracle_rows: list[dict] = []
    episode_rows: list[dict] = []
    train_days = max(1, min(days - 1, int(days * config.train_fraction))) if days > 1 else 1

    for episode_id in range(days):
        weekday = WEEKDAYS[episode_id % len(WEEKDAYS)]
        split = "train" if episode_id < train_days else "test"
        env_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 0]))
        policy_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 1]))
        state = environment.reset(episode_id, weekday)
        episode_cost = 0.0
        episode_arrivals = 0
        episode_served = 0
        episode_abandoned = 0
        episode_violations = 0
        propensities: list[float] = []
        for _ in range(config.periods_per_day):
            draw = demand.sample(state, env_rng)
            action, propensity, probabilities = behavior.act(
                state, draw.manager_alarm, policy_rng
            )
            counterfactuals = evaluate_actions(environment, state, draw)
            realized = counterfactuals.results[action]
            observed_rows.append(
                _observed_row(state, realized, propensity, probabilities, split)
            )
            oracle_rows.append(
                _oracle_row(state, counterfactuals.results,
                            counterfactuals.oracle_action, split)
            )
            episode_cost += realized.total_cost
            episode_arrivals += sum(realized.arrivals)
            episode_served += sum(realized.served)
            episode_abandoned += sum(realized.abandoned)
            episode_violations += int(realized.safety_violation)
            propensities.append(propensity)
            state = realized.next_state
        weights = 1.0 / np.asarray(propensities)
        episode_rows.append({
            "episode_id": episode_id,
            "weekday": weekday,
            "split": split,
            "total_cost": episode_cost,
            "total_arrivals": episode_arrivals,
            "total_served": episode_served,
            "total_abandoned": episode_abandoned,
            "final_queue": state.total_queue,
            "safety_violations": episode_violations,
            "min_propensity": min(propensities),
            "propensity_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
        })
    return GeneratedDataset(
        observed=pd.DataFrame(observed_rows),
        oracle=pd.DataFrame(oracle_rows),
        episodes=pd.DataFrame(episode_rows),
        metadata={
            "seed": seed,
            "days": days,
            "hidden_confounding_strength": config.behavior.hidden_confounding_strength,
        },
    )


def _observed_row(
    state: SystemState,
    result: PeriodResult,
    propensity: float,
    probabilities: np.ndarray,
    split: str,
) -> dict:
    row = {
        "episode_id": state.episode_id,
        "split": split,
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
        "temporary_agents": result.temporary_agents,
        "base_staff_cost": result.base_staff_cost,
        "augmentation_cost": result.augmentation_cost,
        "waiting_cost": result.waiting_cost,
        "abandonment_cost": result.abandonment_cost,
        "service_level_cost": result.service_level_cost,
        "total_cost": result.total_cost,
        "service_level": result.service_level,
        "mean_wait_minutes": result.mean_wait_minutes,
        "mean_wait_regular": result.mean_wait_by_class[0],
        "mean_wait_specialist": result.mean_wait_by_class[1],
        "mean_wait_callback_special": result.mean_wait_by_class[2],
        "p95_wait_minutes": result.p95_wait_minutes,
        "safety_violation": result.safety_violation,
    }
    for action, probability in enumerate(probabilities):
        row[f"prob_action_{action}"] = float(probability)
    return row


def _oracle_row(
    state: SystemState,
    results: tuple[PeriodResult, ...],
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
        row[f"potential_service_level_a{action}"] = result.service_level
        row[f"potential_abandoned_a{action}"] = sum(result.abandoned)
        row[f"potential_p95_wait_a{action}"] = result.p95_wait_minutes
        row[f"potential_safety_violation_a{action}"] = result.safety_violation
        row[f"potential_next_queue_a{action}"] = result.next_state.total_queue
    return row
