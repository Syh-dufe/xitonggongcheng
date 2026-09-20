from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from prescriptive_capacity_sim.calibration import fit_empirical_calibration
from prescriptive_capacity_sim.callcenter_data import QueueCall


def _call(
    key: str,
    *,
    call_type: str,
    wait: float,
    outcome: str,
    service: float | None,
) -> QueueCall:
    return QueueCall(
        call_key=key,
        queue_entry=datetime(1999, 1, 1) + timedelta(seconds=int(key)),
        call_type=call_type,
        priority=1,
        observed_wait_seconds=wait,
        outcome=outcome,
        observed_service_seconds=service,
    )


def test_service_sampling_is_stratified_with_pooled_fallback() -> None:
    calibration = fit_empirical_calibration(
        [
            _call("1", call_type="PS", wait=2, outcome="AGENT", service=10),
            _call("2", call_type="PS", wait=4, outcome="AGENT", service=20),
            _call("3", call_type="NE", wait=5, outcome="AGENT", service=90),
            _call("4", call_type="PS", wait=8, outcome="HANG", service=None),
        ]
    )

    rng = np.random.default_rng(7)
    ps_samples = {calibration.sample_service("PS", rng) for _ in range(30)}
    unknown_samples = {
        calibration.sample_service("UNKNOWN", rng) for _ in range(60)
    }

    assert ps_samples <= {10.0, 20.0}
    assert 90.0 in unknown_samples


def test_sampling_is_reproducible_for_identical_seeds() -> None:
    calibration = fit_empirical_calibration(
        [
            _call("1", call_type="PS", wait=5, outcome="HANG", service=None),
            _call("2", call_type="PS", wait=10, outcome="AGENT", service=20),
            _call("3", call_type="PS", wait=15, outcome="HANG", service=None),
            _call("4", call_type="PS", wait=20, outcome="AGENT", service=40),
        ]
    )

    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)
    first = [
        (calibration.sample_service("PS", first_rng), calibration.sample_patience(first_rng))
        for _ in range(20)
    ]
    second = [
        (calibration.sample_service("PS", second_rng), calibration.sample_patience(second_rng))
        for _ in range(20)
    ]

    assert first == second


def test_patience_curve_uses_answered_calls_as_right_censored() -> None:
    calibration = fit_empirical_calibration(
        [
            _call("1", call_type="PS", wait=5, outcome="HANG", service=None),
            _call("2", call_type="PS", wait=10, outcome="AGENT", service=20),
            _call("3", call_type="PS", wait=15, outcome="HANG", service=None),
        ]
    )

    np.testing.assert_array_equal(calibration.patience_support_seconds, [5.0, 15.0])
    np.testing.assert_allclose(calibration.patience_survival, [2 / 3, 0.0])


def test_calibration_requires_positive_service_and_abandonment_observations() -> None:
    answered_only = [
        _call("1", call_type="PS", wait=1, outcome="AGENT", service=10)
    ]
    abandoned_only = [
        _call("2", call_type="PS", wait=2, outcome="HANG", service=None)
    ]

    with pytest.raises(ValueError, match="abandonment"):
        fit_empirical_calibration(answered_only)
    with pytest.raises(ValueError, match="service"):
        fit_empirical_calibration(abandoned_only)
