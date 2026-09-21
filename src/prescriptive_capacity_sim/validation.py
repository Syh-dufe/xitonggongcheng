"""Held-out diagnostics comparing real and semisynthetic operations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


SERVICE_CLASSES = ("regular", "specialist", "callback_special")


def compare_real_and_simulated(
    real_intervals: pd.DataFrame,
    simulated_log: pd.DataFrame,
    real_calls: pd.DataFrame | None,
    parameters: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for service_class in SERVICE_CLASSES:
        real = real_intervals.loc[real_intervals["service_class"].eq(service_class)]
        real_arrivals = real["arrivals"].to_numpy(dtype=float)
        simulated_arrivals = simulated_log[f"arrival_{service_class}"].to_numpy(dtype=float)
        _append(rows, "arrival_mae", service_class,
                float(np.mean(real_arrivals)) if len(real_arrivals) else 0.0,
                float(np.mean(simulated_arrivals)) if len(simulated_arrivals) else 0.0)
        _append(rows, "arrival_variance_error", service_class,
                float(np.var(real_arrivals)) if len(real_arrivals) else 0.0,
                float(np.var(simulated_arrivals)) if len(simulated_arrivals) else 0.0)
        real_abandoned = float(real["abandoned"].sum())
        real_total = float(real["arrivals"].sum())
        sim_abandoned = float(simulated_log[f"abandoned_{service_class}"].sum())
        sim_total = float(simulated_log[f"arrival_{service_class}"].sum())
        _append(rows, "abandonment_rate_error", service_class,
                real_abandoned / real_total if real_total else 0.0,
                sim_abandoned / sim_total if sim_total else 0.0)
        if real_calls is not None and not real_calls.empty:
            real_service = real_calls.loc[
                real_calls["service_class"].eq(service_class)
                & real_calls["service_seconds"].gt(0), "service_seconds"
            ].to_numpy(dtype=float)
        else:
            real_service = real.get(
                "mean_service_seconds", pd.Series(dtype=float)
            ).dropna().to_numpy(dtype=float)
        simulated_service = np.asarray(
            parameters["empirical"]["service_seconds"][service_class], dtype=float
        )
        distance = float(wasserstein_distance(real_service, simulated_service)) if len(real_service) else 0.0
        rows.append({
            "metric": "service_wasserstein",
            "service_class": service_class,
            "real": float(np.mean(real_service)) if len(real_service) else 0.0,
            "simulated": float(np.mean(simulated_service)),
            "error": distance,
        })
        real_wait = float(real["mean_queue_seconds"].mean() / 60.0) if len(real) else 0.0
        sim_wait = float(simulated_log["mean_wait_minutes"].mean())
        _append(rows, "mean_wait_relative_error", service_class, real_wait, sim_wait,
                relative=True)
    return pd.DataFrame(rows)


def _append(
    rows: list[dict],
    metric: str,
    service_class: str,
    real: float,
    simulated: float,
    *,
    relative: bool = False,
) -> None:
    error = abs(simulated - real) / max(abs(real), 1e-9) if relative else abs(simulated - real)
    rows.append({
        "metric": metric,
        "service_class": service_class,
        "real": real,
        "simulated": simulated,
        "error": float(error),
    })
