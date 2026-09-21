from prescriptive_capacity_sim.config import SimulationConfig
from prescriptive_capacity_sim.evaluation import evaluate_policies
from prescriptive_capacity_sim.policies import default_policies


def _parameters():
    classes = ("regular", "specialist", "callback_special")
    arrival = {
        klass: {
            f"{weekday}:{period}": {"mean": mean, "dispersion": 5.0}
            for weekday in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
            for period in range(48)
        }
        for klass, mean in zip(classes, (3.0, 1.5, 0.5))
    }
    grid = {klass: [60.0, 120.0, 180.0] for klass in classes}
    return {
        "classes": list(classes), "period_minutes": 30, "periods_per_day": 48,
        "arrival": arrival, "priority_rate": {klass: 0.2 for klass in classes},
        "empirical": {"service_seconds": grid, "patience_seconds": grid},
        "disruption": {"arrival_multipliers": [1.0, 1.3, 1.7],
                       "transition_matrix": [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]},
    }


def test_policy_metrics_cover_decision_and_service_outcomes():
    cfg = SimulationConfig.default()
    metrics = evaluate_policies(cfg, _parameters(), default_policies(cfg), days=2, seed=11)
    assert {
        "total_cost", "service_level", "abandonment_rate",
        "mean_wait_minutes", "p95_wait_minutes", "safety_violation_rate",
        "cost_gap_vs_oracle", "cost_improvement_vs_history",
    }.issubset(metrics.columns)
    assert set(metrics["policy"]).issuperset({"historical", "random", "myopic_oracle"})


def test_paired_evaluation_is_reproducible():
    cfg = SimulationConfig.default()
    policies = default_policies(cfg)
    left = evaluate_policies(cfg, _parameters(), policies, days=2, seed=13)
    right = evaluate_policies(cfg, _parameters(), policies, days=2, seed=13)
    assert left.equals(right)
