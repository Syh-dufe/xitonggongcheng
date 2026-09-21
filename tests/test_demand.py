import numpy as np

from prescriptive_capacity_sim.demand import CalibratedDemandProcess
from prescriptive_capacity_sim.state import SystemState


def _parameters():
    arrivals = {}
    for klass, mean in (("regular", 4.0), ("specialist", 2.0), ("callback_special", 1.0)):
        arrivals[klass] = {
            f"monday:{period}": {
                "mean": mean,
                "dispersion": 5.0,
                "fit_scope": "test",
                "n_days": 10,
            }
            for period in range(48)
        }
    samples = {
        klass: [60.0, 120.0, 180.0]
        for klass in ("regular", "specialist", "callback_special")
    }
    return {
        "classes": ["regular", "specialist", "callback_special"],
        "period_minutes": 30,
        "periods_per_day": 48,
        "arrival": arrivals,
        "priority_rate": {"regular": 0.1, "specialist": 0.2, "callback_special": 0.3},
        "empirical": {
            "service_seconds": samples,
            "patience_seconds": samples,
            "queue_seconds": samples,
        },
        "disruption": {
            "arrival_multipliers": [1.0, 1.5, 2.0],
            "transition_matrix": [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
        },
    }


def test_same_seed_reproduces_customer_draws():
    process = CalibratedDemandProcess(_parameters())
    state = SystemState.empty(episode_id=0, weekday="monday")
    a = process.sample(state, np.random.default_rng(7))
    b = process.sample(state, np.random.default_rng(7))
    assert a == b


def test_severe_state_increases_expected_arrivals():
    process = CalibratedDemandProcess(_parameters())
    assert sum(process.expected_arrivals(4, "monday", 2)) > sum(
        process.expected_arrivals(4, "monday", 0)
    )


def test_sampled_calls_have_valid_operational_attributes():
    process = CalibratedDemandProcess(_parameters())
    state = SystemState.empty(episode_id=0, weekday="monday")
    draw = process.sample(state, np.random.default_rng(11))
    assert all(call.service_minutes > 0 for call in draw.new_calls)
    assert all(call.patience_periods >= 1 for call in draw.new_calls)
    assert all(call.service_class in range(3) for call in draw.new_calls)
    assert all(call.intraperiod_wait_minutes >= 1.0 for call in draw.new_calls)
