"""Paired evaluation of staffing policies under shared stochastic episodes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import CalibratedDemandProcess
from .environment import CallCenterEnvironment
from .logging import WEEKDAYS
from .policies import Policy


def evaluate_policies(
    config: SimulationConfig,
    parameters: dict | str,
    policies: list[Policy],
    *,
    days: int,
    seed: int,
) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")
    names = [policy.name for policy in policies]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")
    rows: list[dict] = []
    for policy_index, policy in enumerate(policies):
        demand = CalibratedDemandProcess(
            parameters,
            hidden_confounding_strength=config.behavior.hidden_confounding_strength,
            demand_shock_scale=config.demand_shock_scale,
        )
        environment = CallCenterEnvironment(config)
        total_cost = 0.0
        total_arrivals = 0
        total_served = 0
        total_abandoned = 0
        temporary_agents = 0
        violations = 0
        waits: list[float] = []
        service_levels: list[float] = []
        for episode_id in range(days):
            env_rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id, 0]))
            policy_rng = np.random.default_rng(
                np.random.SeedSequence([seed, episode_id, policy_index + 100])
            )
            state = environment.reset(episode_id, WEEKDAYS[episode_id % 7])
            for _ in range(config.periods_per_day):
                draw = demand.sample(state, env_rng)
                action = policy.act(state, draw, environment, policy_rng)
                result = environment.transition(state, action, draw)
                total_cost += result.total_cost
                total_arrivals += sum(result.arrivals)
                total_served += sum(result.served)
                total_abandoned += sum(result.abandoned)
                temporary_agents += result.temporary_agents
                violations += int(result.safety_violation)
                waits.append(result.mean_wait_minutes)
                service_levels.append(result.service_level)
                state = result.next_state
        periods = days * config.periods_per_day
        rows.append({
            "policy": policy.name,
            "total_cost": float(total_cost),
            "mean_period_cost": float(total_cost / periods),
            "service_level": float(np.mean(service_levels)),
            "service_completion_rate": total_served / total_arrivals if total_arrivals else 1.0,
            "abandonment_rate": total_abandoned / total_arrivals if total_arrivals else 0.0,
            "mean_wait_minutes": float(np.mean(waits)),
            "p95_wait_minutes": float(np.percentile(waits, 95)),
            "safety_violation_rate": violations / periods,
            "temporary_agents": temporary_agents,
        })
    metrics = pd.DataFrame(rows)
    oracle = metrics.loc[metrics["policy"].eq("myopic_oracle"), "total_cost"]
    oracle_cost = float(oracle.iloc[0]) if len(oracle) else float(metrics["total_cost"].min())
    history = metrics.loc[metrics["policy"].eq("historical"), "total_cost"]
    history_cost = float(history.iloc[0]) if len(history) else float("nan")
    metrics["cost_gap_vs_oracle"] = metrics["total_cost"] - oracle_cost
    metrics["cost_improvement_vs_history"] = (
        (history_cost - metrics["total_cost"]) / history_cost
        if np.isfinite(history_cost) and history_cost else np.nan
    )
    return metrics
