import numpy as np

from prescriptive_capacity_sim.behavior import HistoricalBehaviorPolicy
from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.demand import DemandProcess
from prescriptive_capacity_sim.state import SystemState


def test_seeded_demand_shocks_are_reproducible() -> None:
    cfg = SimulationConfig.default()
    process = DemandProcess(cfg)
    state = SystemState.empty(cfg, episode_id=4, disruption_state=1)
    first = process.sample_shock(state, np.random.default_rng(91), daily_effect=0.2)
    second = process.sample_shock(state, np.random.default_rng(91), daily_effect=0.2)
    assert first == second


def test_severe_disruption_has_larger_expected_arrivals() -> None:
    cfg = SimulationConfig.default()
    process = DemandProcess(cfg)
    normal = process.expected_arrivals(period=12, disruption_state=0, daily_effect=0.0, manager_alarm=0.0)
    severe = process.expected_arrivals(period=12, disruption_state=2, daily_effect=0.0, manager_alarm=0.0)
    assert all(high > low for low, high in zip(normal, severe))


def test_behavior_probabilities_are_valid_and_action_uses_them() -> None:
    cfg = SimulationConfig.default()
    policy = HistoricalBehaviorPolicy(cfg)
    state = SystemState.from_backlog(cfg, episode_id=0, period=8, backlog=(30, 12, 5), disruption_state=1)
    probabilities = policy.probabilities(state, manager_alarm=0.25)
    action, propensity, returned = policy.act(state, np.random.default_rng(5), manager_alarm=0.25)
    assert probabilities.shape == (4,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0)
    assert 0 <= action < 4
    assert propensity == returned[action]


def test_lower_temperature_concentrates_behavior_distribution() -> None:
    base = SimulationConfig.default()
    state = SystemState.from_backlog(base, episode_id=0, period=8, backlog=(45, 18, 8), disruption_state=2)
    cold = HistoricalBehaviorPolicy(base.with_overrides(behavior={"temperature": 0.25}))
    warm = HistoricalBehaviorPolicy(base.with_overrides(behavior={"temperature": 2.0}))
    assert cold.probabilities(state, 0.0).max() > warm.probabilities(state, 0.0).max()
