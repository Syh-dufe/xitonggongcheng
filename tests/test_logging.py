import pandas as pd

from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.logging import generate_dataset


def test_generated_logs_separate_observed_and_counterfactual_data() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=4)
    data = generate_dataset(cfg, days=2, seed=123)
    assert len(data.observed) == 8
    assert len(data.oracle) == 8
    assert not any("potential" in column or column == "oracle_action" for column in data.observed.columns)
    for action in range(4):
        assert f"potential_cost_a{action}" in data.oracle.columns
        assert f"potential_service_rate_a{action}" in data.oracle.columns


def test_oracle_action_minimizes_one_step_potential_cost() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=5)
    data = generate_dataset(cfg, days=1, seed=77)
    cost_columns = [f"potential_cost_a{action}" for action in range(4)]
    expected = data.oracle[cost_columns].to_numpy().argmin(axis=1)
    assert (data.oracle["oracle_action"].to_numpy() == expected).all()


def test_dataset_generation_is_seed_reproducible() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=3)
    first = generate_dataset(cfg, days=2, seed=812)
    second = generate_dataset(cfg, days=2, seed=812)
    pd.testing.assert_frame_equal(first.observed, second.observed)
    pd.testing.assert_frame_equal(first.oracle, second.oracle)
    pd.testing.assert_frame_equal(first.episodes, second.episodes)


def test_different_seed_changes_stochastic_log() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=3)
    first = generate_dataset(cfg, days=1, seed=1)
    second = generate_dataset(cfg, days=1, seed=2)
    assert not first.observed.equals(second.observed)


def test_episode_split_and_overlap_diagnostics_are_recorded() -> None:
    cfg = SimulationConfig.default().with_overrides(periods_per_day=3)
    data = generate_dataset(cfg, days=10, seed=31)
    split_by_episode = data.observed.groupby("episode_id")["split"].nunique()
    assert (split_by_episode == 1).all()
    assert set(data.observed["split"]) == {"train", "test"}
    assert {"min_propensity", "propensity_ess", "split"}.issubset(data.episodes.columns)
    assert (data.episodes["min_propensity"] > 0).all()
    assert (data.episodes["propensity_ess"] > 0).all()
