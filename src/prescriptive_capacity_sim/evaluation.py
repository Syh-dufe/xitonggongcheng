"""Paired out-of-sample evaluation for capacity policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .environment import CapacityEnvironment
from .policies import Policy


def evaluate_policies(
    config: SimulationConfig,
    policies: list[Policy],
    days: int,
    seed: int,
) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")
    names = [str(policy.name) for policy in policies]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")

    rows: list[dict] = []
    for policy_index, policy in enumerate(policies):
        environment = CapacityEnvironment(config)
        episode_costs: list[float] = []
        backlogs: list[int] = []
        total_completed = 0
        total_arrivals = 0
        temporary_capacity = 0
        violations = 0
        periods = days * config.periods_per_day

        for episode_id in range(days):
            env_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 0]))
            policy_rng = np.random.default_rng(
                np.random.SeedSequence([seed, episode_id, policy_index + 100])
            )
            daily_effect = float(env_rng.normal(0.0, config.demand.daily_effect_sigma))
            state = environment.reset(episode_id)
            episode_cost = 0.0
            for _ in range(config.periods_per_day):
                shock = environment.sample_exogenous(state, env_rng, daily_effect)
                action = policy.act(state, shock, environment, policy_rng)
                result = environment.transition(state, action, shock)
                episode_cost += result.total_cost
                total_completed += sum(result.completed)
                total_arrivals += sum(result.arrivals)
                temporary_capacity += result.temporary_staff
                violations += int(result.safety_violation)
                backlogs.append(result.next_state.total_backlog)
                state = result.next_state
            episode_costs.append(episode_cost)

        total_cost = float(sum(episode_costs))
        rows.append(
            {
                "policy": str(policy.name),
                "total_cost": total_cost,
                "mean_episode_cost": float(np.mean(episode_costs)),
                "mean_period_cost": total_cost / periods,
                "service_rate": total_completed / total_arrivals if total_arrivals else 1.0,
                "mean_backlog": float(np.mean(backlogs)),
                "p95_backlog": float(np.percentile(backlogs, 95)),
                "max_backlog": int(max(backlogs, default=0)),
                "temporary_capacity": int(temporary_capacity),
                "safety_violation_rate": violations / periods,
            }
        )

    metrics = pd.DataFrame(rows)
    oracle_rows = metrics.loc[metrics["policy"] == "myopic_oracle", "total_cost"]
    myopic_oracle_cost = (
        float(oracle_rows.iloc[0]) if len(oracle_rows) else float(metrics.total_cost.min())
    )
    best_evaluated_cost = float(metrics.total_cost.min())
    history_rows = metrics.loc[metrics["policy"] == "historical", "total_cost"]
    history_cost = float(history_rows.iloc[0]) if len(history_rows) else float("nan")
    metrics["cost_gap_vs_myopic_oracle"] = metrics["total_cost"] - myopic_oracle_cost
    metrics["empirical_regret_vs_best"] = metrics["total_cost"] - best_evaluated_cost
    metrics["cost_improvement_vs_history"] = (
        (history_cost - metrics["total_cost"]) / history_cost
        if np.isfinite(history_cost) and history_cost != 0
        else np.nan
    )
    return metrics
