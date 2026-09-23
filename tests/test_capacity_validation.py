import pandas as pd
import pytest

from prescriptive_capacity_sim.capacity_validation import (
    SELECTION_WEIGHTS,
    load_candidates,
    simulate_historical_periods,
    summarize_candidates,
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
    assert {
        "exit_wait_regular",
        "exit_wait_specialist",
        "exit_wait_callback_special",
    }.issubset(left.columns)
    assert all(isinstance(value, tuple) for value in left["exit_wait_regular"])


def test_factual_runner_uses_explicit_validation_weekdays():
    config = SimulationConfig.default().with_overrides(periods_per_day=1)
    parameters = _parameters(periods_per_day=1)

    observed = simulate_historical_periods(
        config,
        parameters,
        days=2,
        seed=23,
        weekdays=("thursday", "friday"),
    )

    assert observed["weekday"].tolist() == ["thursday", "friday"]
    with pytest.raises(ValueError, match="length"):
        simulate_historical_periods(
            config,
            parameters,
            days=2,
            seed=23,
            weekdays=("thursday",),
        )


@pytest.mark.parametrize(
    ("config", "parameters", "message"),
    [
        (
            SimulationConfig.default().with_overrides(periods_per_day=2),
            _parameters(periods_per_day=3),
            "periods_per_day",
        ),
        (
            SimulationConfig.default().with_overrides(
                resources={"minutes_per_period": 15}
            ),
            _parameters(periods_per_day=48),
            "period_minutes",
        ),
    ],
)
def test_factual_runner_rejects_incompatible_time_grids(
    config, parameters, message
):
    with pytest.raises(ValueError, match=message):
        simulate_historical_periods(config, parameters, days=1, seed=23)


def test_summary_averages_classes_before_predeclared_metric_weights():
    rows = []
    resource_values = {
        "regular_agents": 8,
        "specialist_agents": 5,
        "supervisor_emergency_agents": 1,
        "cross_skill_efficiency": 0.75,
    }
    for candidate, seed_errors in {
        "higher": {11: (0.3, 0.5), 13: (0.5, 0.7)},
        "lower": {11: (0.1, 0.3), 13: (0.3, 0.5)},
    }.items():
        for seed, class_errors in seed_errors.items():
            for metric in SELECTION_WEIGHTS:
                for service_class, error in zip(("regular", "specialist"), class_errors):
                    rows.append({
                        "candidate": candidate,
                        "seed": seed,
                        "metric": metric,
                        "service_class": service_class,
                        "error": error,
                        **resource_values,
                    })
            rows.append({
                "candidate": candidate,
                "seed": seed,
                "metric": "arrival_mae",
                "service_class": "regular",
                "error": 99.0,
                **resource_values,
            })

    summary = summarize_candidates(pd.DataFrame(rows))

    assert summary["candidate"].tolist() == ["lower", "higher"]
    lower = summary.iloc[0]
    assert lower["selection_score"] == pytest.approx(0.3)
    assert lower["selection_score_std"] == pytest.approx(0.1)
    assert lower["seed_count"] == 2
    assert lower["rank"] == 1
    assert lower["regular_agents"] == 8
    assert lower["specialist_agents"] == 5
    assert lower["supervisor_emergency_agents"] == 1
    assert lower["cross_skill_efficiency"] == 0.75


def test_summary_rejects_resource_values_that_are_not_fixed_per_candidate():
    rows = [
        {
            "candidate": "unstable",
            "seed": 11,
            "metric": metric,
            "service_class": "overall",
            "error": 0.1,
            "regular_agents": 8,
            "specialist_agents": 5,
            "supervisor_emergency_agents": 1,
            "cross_skill_efficiency": 0.75,
        }
        for metric in SELECTION_WEIGHTS
    ]
    rows[-1]["regular_agents"] = 9

    with pytest.raises(ValueError, match="fixed"):
        summarize_candidates(pd.DataFrame(rows))
