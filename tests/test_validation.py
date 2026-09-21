import pandas as pd

from prescriptive_capacity_sim.validation import compare_real_and_simulated


def test_validation_reports_required_metrics_for_each_class():
    real_intervals = pd.DataFrame({
        "service_class": ["regular", "specialist", "callback_special"],
        "arrivals": [10, 5, 2], "abandoned": [2, 1, 0],
        "mean_queue_seconds": [60.0, 90.0, 30.0],
        "p90_queue_seconds": [120.0, 180.0, 60.0],
    })
    simulated = pd.DataFrame({
        "arrival_regular": [9], "arrival_specialist": [6],
        "arrival_callback_special": [2], "abandoned_regular": [1],
        "abandoned_specialist": [1], "abandoned_callback_special": [0],
        "mean_wait_minutes": [1.0], "p95_wait_minutes": [2.0],
    })
    real_calls = pd.DataFrame({
        "service_class": ["regular", "specialist", "callback_special"],
        "service_seconds": [120.0, 180.0, 90.0],
    })
    params = {"empirical": {"service_seconds": {
        "regular": [100.0, 120.0], "specialist": [160.0, 180.0],
        "callback_special": [80.0, 100.0],
    }}}
    report = compare_real_and_simulated(real_intervals, simulated, real_calls, params)
    assert {"metric", "service_class", "real", "simulated", "error"}.issubset(report.columns)
    assert {"arrival_mae", "service_wasserstein", "abandonment_rate_error"}.issubset(
        set(report["metric"])
    )
    assert set(report["service_class"]) == {"regular", "specialist", "callback_special"}


def test_arrival_validation_includes_real_zero_intervals():
    real_intervals = pd.DataFrame({
        "call_date": ["1999-01-01"], "period": [0],
        "service_class": ["regular"], "arrivals": [10], "abandoned": [0],
        "mean_queue_seconds": [0.0], "p90_queue_seconds": [0.0],
        "mean_service_seconds": [120.0],
    })
    simulated = pd.DataFrame({
        "arrival_regular": [10, 0], "arrival_specialist": [0, 0],
        "arrival_callback_special": [0, 0], "abandoned_regular": [0, 0],
        "abandoned_specialist": [0, 0], "abandoned_callback_special": [0, 0],
        "mean_wait_minutes": [0.0, 0.0], "p95_wait_minutes": [0.0, 0.0],
    })
    params = {
        "periods_per_day": 2,
        "empirical": {"service_seconds": {
            "regular": [120.0], "specialist": [120.0],
            "callback_special": [120.0],
        }},
    }
    real_calls = pd.DataFrame({
        "service_class": ["regular"], "service_seconds": [120.0]
    })
    report = compare_real_and_simulated(real_intervals, simulated, real_calls, params)
    row = report.loc[
        report["metric"].eq("arrival_mae")
        & report["service_class"].eq("regular")
    ].iloc[0]
    assert row["real"] == 5.0
    assert row["simulated"] == 5.0
    assert row["error"] == 0.0
