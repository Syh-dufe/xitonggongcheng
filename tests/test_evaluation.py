import numpy as np
import pandas as pd

from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.environment import CapacityEnvironment
from prescriptive_capacity_sim.evaluation import evaluate_policies
from prescriptive_capacity_sim.policies import (
    FixedActionPolicy,
    HistoricalPolicy,
    MyopicOraclePolicy,
    RandomPolicy,
    RollingOraclePolicy,
)


def test_builtin_policies_return_valid_actions() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=3)
    env = CapacityEnvironment(cfg)
    state = env.reset(0)
    shock = env.sample_exogenous(state, np.random.default_rng(1), daily_effect=0.0)
    policies = [
        FixedActionPolicy(0),
        RandomPolicy(),
        HistoricalPolicy(cfg),
        MyopicOraclePolicy(),
        RollingOraclePolicy(horizon=2),
    ]
    for policy in policies:
        action = policy.act(state, shock, env, np.random.default_rng(9))
        assert 0 <= action < 4


def test_paired_evaluation_reports_required_metrics() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=4)
    policies = [
        FixedActionPolicy(0, name="fixed_0"),
        FixedActionPolicy(3, name="fixed_3"),
        HistoricalPolicy(cfg),
        MyopicOraclePolicy(),
    ]
    metrics = evaluate_policies(cfg, policies, days=2, seed=55)
    required = {
        "policy",
        "total_cost",
        "mean_episode_cost",
        "service_rate",
        "mean_backlog",
        "p95_backlog",
        "max_backlog",
        "temporary_capacity",
        "safety_violation_rate",
        "cost_gap_vs_myopic_oracle",
        "empirical_regret_vs_best",
        "cost_improvement_vs_history",
    }
    assert required.issubset(metrics.columns)
    assert len(metrics) == len(policies)
    low_temp = metrics.loc[metrics.policy == "fixed_0", "temporary_capacity"].item()
    high_temp = metrics.loc[metrics.policy == "fixed_3", "temporary_capacity"].item()
    assert high_temp > low_temp
    assert (metrics["empirical_regret_vs_best"] >= 0).all()
    assert "regret_vs_oracle" not in metrics.columns


def test_paired_evaluation_is_reproducible() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=3)
    policies = [HistoricalPolicy(cfg), MyopicOraclePolicy()]
    first = evaluate_policies(cfg, policies, days=2, seed=8)
    second = evaluate_policies(cfg, policies, days=2, seed=8)
    pd.testing.assert_frame_equal(first, second)
