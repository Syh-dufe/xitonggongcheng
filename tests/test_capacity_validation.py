import pytest

from prescriptive_capacity_sim.capacity_validation import (
    load_candidates,
    simulate_historical_periods,
)
from prescriptive_capacity_sim.config import SimulationConfig


def test_candidate_grid_rejects_duplicate_names(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "candidates:\n"
        "  - name: baseline\n"
        "    regular_agents: 8\n"
        "    specialist_agents: 5\n"
        "  - name: baseline\n"
        "    regular_agents: 9\n"
        "    specialist_agents: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_candidates(path)


def _parameters(periods_per_day: int) -> dict:
    service_classes = ("regular", "specialist", "callback_special")
    arrival = {
        service_class: {
            f"monday:{period}": {"mean": 1.0, "dispersion": 5.0}
            for period in range(periods_per_day)
        }
        for service_class in service_classes
    }
    empirical_grid = {service_class: [60.0, 120.0] for service_class in service_classes}
    return {
        "period_minutes": 30,
        "periods_per_day": periods_per_day,
        "arrival": arrival,
        "priority_rate": {service_class: 0.2 for service_class in service_classes},
        "empirical": {
            "service_seconds": empirical_grid,
            "patience_seconds": empirical_grid,
        },
        "disruption": {
            "arrival_multipliers": [1.0, 1.3, 1.7],
            "transition_matrix": [
                [0.8, 0.15, 0.05],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ],
        },
    }


def test_factual_runner_is_reproducible_and_excludes_oracle_columns():
    config = SimulationConfig.default().with_overrides(periods_per_day=2)
    parameters = _parameters(periods_per_day=2)

    left = simulate_historical_periods(config, parameters, days=2, seed=23)
    right = simulate_historical_periods(config, parameters, days=2, seed=23)

    assert left.equals(right)
    assert len(left) == 4
    assert not any(
        "potential_" in column or "oracle" in column for column in left.columns
    )
