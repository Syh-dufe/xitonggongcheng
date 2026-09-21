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
