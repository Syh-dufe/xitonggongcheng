from __future__ import annotations

from datetime import date, datetime

import numpy as np

from prescriptive_capacity_sim.calibration import EmpiricalCalibration
from prescriptive_capacity_sim.callcenter_data import QueueCall
from prescriptive_capacity_sim.callcenter_scenarios import (
    CostConfig,
    StaffingBehaviorPolicy,
    build_replay_day,
    generate_semisynthetic_day,
)


def _calibration() -> EmpiricalCalibration:
    return EmpiricalCalibration(
        service_seconds_by_type={"PS": np.asarray([30.0, 60.0])},
        pooled_service_seconds=np.asarray([30.0, 60.0]),
        patience_support_seconds=np.asarray([20.0, 90.0]),
        patience_survival=np.asarray([0.5, 0.0]),
    )


def _observed_call(key: str, timestamp: datetime) -> QueueCall:
    return QueueCall(
        call_key=key,
        queue_entry=timestamp,
        call_type="PS",
        priority=1,
        observed_wait_seconds=5,
        outcome="AGENT",
        observed_service_seconds=30,
    )


def test_real_day_replay_preserves_arrivals_and_is_seeded() -> None:
    calls = [
        _observed_call("a", datetime(1999, 1, 1, 0, 5)),
        _observed_call("b", datetime(1999, 1, 1, 0, 35)),
        _observed_call("other-day", datetime(1999, 1, 2, 0, 5)),
    ]

    first = build_replay_day(
        calls,
        _calibration(),
        date(1999, 1, 1),
        seed=11,
        interval_seconds=1800,
        intervals_per_day=2,
    )
    second = build_replay_day(
        calls,
        _calibration(),
        date(1999, 1, 1),
        seed=11,
        interval_seconds=1800,
        intervals_per_day=2,
    )

    assert first == second
    assert [[call.call_key for call in group] for group in first.arrivals_by_interval] == [
        ["a"],
        ["b"],
    ]
    assert first.arrivals_by_interval[0][0].arrival_second == 300
    assert first.arrivals_by_interval[1][0].arrival_second == 2100


def test_behavior_policy_has_overlap_for_every_action() -> None:
    policy = StaffingBehaviorPolicy(actions=(0, 1, 2, 3), temperature=0.4)

    probabilities = policy.probabilities(
        queue_length=10, expected_arrivals=20, hidden_pressure=1.2
    )

    assert set(probabilities) == {0, 1, 2, 3}
    assert all(value > 0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_oracle_branches_share_state_and_exogenous_calls() -> None:
    calls = [
        _observed_call("a", datetime(1999, 1, 1, 0, 0, 1)),
        _observed_call("b", datetime(1999, 1, 1, 0, 0, 2)),
    ]
    replay = build_replay_day(
        calls,
        _calibration(),
        date(1999, 1, 1),
        seed=9,
        interval_seconds=60,
        intervals_per_day=1,
    )
    policy = StaffingBehaviorPolicy(actions=(0, 1, 2), temperature=1.0)

    data = generate_semisynthetic_day(
        replay,
        behavior_policy=policy,
        costs=CostConfig(),
        seed=99,
    )

    assert len(data.observed) == 1
    assert len(data.oracle) == 3
    assert {row["action"] for row in data.oracle} == {0, 1, 2}
    assert len({row["pre_state_fingerprint"] for row in data.oracle}) == 1
    assert len({row["arrival_fingerprint"] for row in data.oracle}) == 1
    chosen = data.observed[0]["action"]
    chosen_oracle = next(row for row in data.oracle if row["action"] == chosen)
    assert data.observed[0]["realized_cost"] == chosen_oracle["potential_cost"]
    assert not any(key.startswith("potential_") for key in data.observed[0])
    assert all(row["propensity"] > 0 for row in data.oracle)
