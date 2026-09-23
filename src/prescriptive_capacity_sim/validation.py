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
        real_arrivals = _complete_real_arrivals(
            real_intervals, service_class, int(parameters.get("periods_per_day", 48))
        )
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
        if real_calls is not None and "queue_seconds" in real_calls:
            real_wait_values = real_calls.loc[
                real_calls["service_class"].eq(service_class), "queue_seconds"
            ].to_numpy(dtype=float)
            real_wait = float(np.mean(real_wait_values) / 60.0) if len(real_wait_values) else 0.0
        else:
            real_wait = float(real["mean_queue_seconds"].mean() / 60.0) if len(real) else 0.0
        pooled_waits = _simulated_exit_waits(simulated_log, (service_class,))
        if pooled_waits is not None:
            sim_wait = _mean_or_zero(pooled_waits)
        else:
            class_wait_column = f"mean_wait_{service_class}"
            sim_wait = float(simulated_log[
                class_wait_column if class_wait_column in simulated_log else "mean_wait_minutes"
            ].mean())
        _append(rows, "mean_wait_relative_error", service_class, real_wait, sim_wait,
                relative=True)
    real_p90_wait = _positive_wait_p90_minutes(real_calls)
    pooled_waits = _simulated_exit_waits(simulated_log, SERVICE_CLASSES)
    simulated_p90_wait = (
        _p90_or_zero(pooled_waits)
        if pooled_waits is not None
        else _p90_or_zero(simulated_log.get("p95_wait_minutes", pd.Series(dtype=float)))
    )
    _append(
        rows,
        "overall_p90_wait_relative_error",
        "overall",
        real_p90_wait,
        simulated_p90_wait,
        relative=True,
    )
    return pd.DataFrame(rows)


def _complete_real_arrivals(
    intervals: pd.DataFrame,
    service_class: str,
    periods_per_day: int,
) -> np.ndarray:
    if {"call_date", "period"}.issubset(intervals.columns):
        dates = pd.Index(intervals["call_date"].drop_duplicates())
        index = pd.MultiIndex.from_product(
            [dates, range(periods_per_day)], names=["call_date", "period"]
        )
        values = intervals.loc[
            intervals["service_class"].eq(service_class)
        ].groupby(["call_date", "period"], observed=True)["arrivals"].sum()
        return values.reindex(index, fill_value=0).to_numpy(dtype=float)
    return intervals.loc[
        intervals["service_class"].eq(service_class), "arrivals"
    ].to_numpy(dtype=float)


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


def _positive_wait_p90_minutes(real_calls: pd.DataFrame | None) -> float:
    if real_calls is None or real_calls.empty or "queue_seconds" not in real_calls:
        return 0.0
    positive_waits = real_calls.loc[
        real_calls["queue_seconds"].gt(0), "queue_seconds"
    ]
    return _p90_or_zero(positive_waits) / 60.0


def _p90_or_zero(values: pd.Series | np.ndarray) -> float:
    array = _finite_values(values)
    return float(np.percentile(array, 90)) if len(array) else 0.0


def _mean_or_zero(values: pd.Series | np.ndarray) -> float:
    array = _finite_values(values)
    return float(np.mean(array)) if len(array) else 0.0


def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _simulated_exit_waits(
    simulated_log: pd.DataFrame,
    service_classes: tuple[str, ...],
) -> np.ndarray | None:
    columns = [f"exit_wait_{service_class}" for service_class in service_classes]
    if not all(column in simulated_log for column in columns):
        return None
    values: list[float] = []
    for column in columns:
        for exit_waits in simulated_log[column]:
            if exit_waits is None:
                continue
            values.extend(np.asarray(exit_waits, dtype=float).reshape(-1))
    return np.asarray(values, dtype=float)
