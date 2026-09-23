import numpy as np

from prescriptive_capacity_sim.behavior import HistoricalBehaviorPolicy
from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.logging import generate_dataset
from prescriptive_capacity_sim.state import SystemState


def _parameters():
    arrivals = {}
    for klass, mean in (("regular", 3.0), ("specialist", 1.5), ("callback_special", 0.5)):
        arrivals[klass] = {}
        for weekday in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
            for period in range(48):
                arrivals[klass][f"{weekday}:{period}"] = {
                    "mean": mean, "dispersion": 5.0,
                }
    grids = {klass: [60.0, 120.0, 180.0] for klass in arrivals}
    return {
        "classes": list(arrivals), "period_minutes": 30, "periods_per_day": 48,
        "arrival": arrivals,
        "priority_rate": {klass: 0.2 for klass in arrivals},
        "empirical": {"service_seconds": grids, "patience_seconds": grids},
        "disruption": {
            "arrival_multipliers": [1.0, 1.3, 1.7],
            "transition_matrix": [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
        },
    }


def test_observed_log_has_one_action_and_no_potential_outcomes():
    data = generate_dataset(SimulationConfig.default(), _parameters(), days=2, seed=9)
    assert "action" in data.observed
    assert not any(column.startswith("potential_") for column in data.observed)
    assert "oracle_action" not in data.observed


def test_observed_log_keeps_exact_exit_waits_by_service_class():
    data = generate_dataset(SimulationConfig.default(), _parameters(), days=2, seed=9)

    assert {
        "exit_wait_regular",
        "exit_wait_specialist",
        "exit_wait_callback_special",
    }.issubset(data.observed.columns)
    assert all(isinstance(value, tuple) for value in data.observed["exit_wait_regular"])


def test_oracle_contains_four_actions_and_correct_minimizer():
    data = generate_dataset(SimulationConfig.default(), _parameters(), days=2, seed=9)
    costs = [f"potential_cost_a{action}" for action in range(4)]
    assert set(costs).issubset(data.oracle)
    expected = data.oracle[costs].to_numpy().argmin(axis=1)
    assert (data.oracle["oracle_action"].to_numpy() == expected).all()


def test_hidden_alarm_affects_metadata_but_is_not_observed():
    cfg = SimulationConfig.default().with_overrides(
        behavior={"hidden_confounding_strength": 1.0}
    )
    data = generate_dataset(cfg, _parameters(), days=2, seed=4)
    assert "manager_alarm" not in data.observed.columns
    assert data.metadata["hidden_confounding_strength"] == 1.0


def test_lower_temperature_reduces_action_overlap():
    state = SystemState.empty(0, "monday")
    base = SimulationConfig.default()
    low = HistoricalBehaviorPolicy(
        base.with_overrides(behavior={"temperature": 0.25})
    ).probabilities(state, manager_alarm=0.0)
    high = HistoricalBehaviorPolicy(
        base.with_overrides(behavior={"temperature": 2.0})
    ).probabilities(state, manager_alarm=0.0)
    assert -(low * np.log(low)).sum() < -(high * np.log(high)).sum()


def test_generation_is_seed_reproducible():
    left = generate_dataset(SimulationConfig.default(), _parameters(), days=2, seed=12)
    right = generate_dataset(SimulationConfig.default(), _parameters(), days=2, seed=12)
    assert left.observed.equals(right.observed)
    assert left.oracle.equals(right.oracle)
