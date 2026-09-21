"""Calibrate semisynthetic demand and service processes from call logs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ingest import DataQualityCounts


SERVICE_CLASSES = ("regular", "specialist", "callback_special")
WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
)


@dataclass(frozen=True)
class CalibrationPaths:
    intervals: Path
    parameters: Path
    quality: Path
    manifest: Path


@dataclass(frozen=True)
class CalibrationResult:
    train_calls: pd.DataFrame
    validation_calls: pd.DataFrame
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    intervals: pd.DataFrame
    validation_intervals: pd.DataFrame
    parameters: dict
    quality: DataQualityCounts


def calibrate_calls(
    calls: pd.DataFrame,
    *,
    quality: DataQualityCounts | None = None,
    train_fraction: float = 0.70,
    period_minutes: int = 30,
) -> CalibrationResult:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if period_minutes <= 0 or 1440 % period_minutes:
        raise ValueError("period_minutes must divide one day")
    if calls.empty:
        raise ValueError("cannot calibrate an empty call frame")
    dates = tuple(pd.Timestamp(v) for v in sorted(calls["call_date"].dropna().unique()))
    if len(dates) < 2:
        raise ValueError("at least two distinct dates are required")
    cutoff = max(1, min(len(dates) - 1, int(np.floor(len(dates) * train_fraction))))
    train_dates, validation_dates = dates[:cutoff], dates[cutoff:]
    train = calls.loc[calls["call_date"].isin(train_dates)].copy()
    validation = calls.loc[calls["call_date"].isin(validation_dates)].copy()
    periods_per_day = 1440 // period_minutes
    train_intervals = _aggregate_intervals(train, period_minutes)
    validation_intervals = _aggregate_intervals(validation, period_minutes)
    parameters = {
        "schema_version": 1,
        "fit_scope": "train_only",
        "period_minutes": period_minutes,
        "periods_per_day": periods_per_day,
        "classes": list(SERVICE_CLASSES),
        "train_start": min(train_dates).date().isoformat(),
        "train_end": max(train_dates).date().isoformat(),
        "validation_start": min(validation_dates).date().isoformat(),
        "validation_end": max(validation_dates).date().isoformat(),
        "arrival": _fit_arrivals(train, train_dates, period_minutes),
        "priority_rate": _fit_priority_rates(train),
        "empirical": {
            "service_seconds": _empirical_quantiles(
                train.loc[train["service_seconds"].gt(0)], "service_seconds"
            ),
            "patience_seconds": _empirical_quantiles(
                train.loc[train["patience_seconds"].gt(0)], "patience_seconds"
            ),
            "queue_seconds": _empirical_quantiles(train, "queue_seconds"),
        },
        "disruption": _fit_disruption(train, train_dates),
    }
    return CalibrationResult(
        train_calls=train.reset_index(drop=True),
        validation_calls=validation.reset_index(drop=True),
        train_dates=train_dates,
        validation_dates=validation_dates,
        intervals=train_intervals,
        validation_intervals=validation_intervals,
        parameters=parameters,
        quality=quality or DataQualityCounts(total_rows=len(calls)),
    )


def write_calibration(
    result: CalibrationResult,
    output_dir: str | Path,
    *,
    source_files: Iterable[str | Path] = (),
) -> CalibrationPaths:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = CalibrationPaths(
        intervals=output / "calibration_intervals.csv",
        parameters=output / "calibration_parameters.json",
        quality=output / "data_quality_report.json",
        manifest=output / "calibration_manifest.json",
    )
    sources = [Path(p) for p in source_files]
    source_hashes = {
        p.name: _sha256(p) for p in sources if p.is_file() and not p.name.startswith("._")
    }
    manifest = {
        "schema_version": 1,
        "source_files": sorted(source_hashes),
        "source_sha256": source_hashes,
        "train_dates": len(result.train_dates),
        "validation_dates": len(result.validation_dates),
        "training_calls": len(result.train_calls),
        "validation_calls": len(result.validation_calls),
        "parameters_sha256": _json_sha256(result.parameters),
    }
    train_intervals = result.intervals.copy()
    train_intervals.insert(0, "split", "train")
    validation_intervals = result.validation_intervals.copy()
    validation_intervals.insert(0, "split", "validation")
    _atomic_csv(pd.concat([train_intervals, validation_intervals], ignore_index=True), paths.intervals)
    _atomic_json(result.parameters, paths.parameters)
    _atomic_json(asdict(result.quality), paths.quality)
    _atomic_json(manifest, paths.manifest)
    return paths


def _aggregate_intervals(calls: pd.DataFrame, period_minutes: int) -> pd.DataFrame:
    data = calls.copy()
    data["period"] = (
        data["arrival_ts"].dt.hour * 60 + data["arrival_ts"].dt.minute
    ) // period_minutes
    data["completed"] = data["outcome"].eq("AGENT").astype(int)
    data["abandoned"] = data["outcome"].eq("HANG").astype(int)
    grouped = data.groupby(
        ["call_date", "weekday", "period", "service_class", "priority"],
        observed=True,
        sort=True,
    )
    intervals = grouped.agg(
        arrivals=("outcome", "size"),
        completed=("completed", "sum"),
        abandoned=("abandoned", "sum"),
        mean_queue_seconds=("queue_seconds", "mean"),
        p90_queue_seconds=("queue_seconds", lambda x: float(x.quantile(0.9))),
        mean_service_seconds=("service_seconds", "mean"),
        p90_service_seconds=("service_seconds", lambda x: float(x.quantile(0.9))),
    ).reset_index()
    return intervals


def _fit_arrivals(
    train: pd.DataFrame,
    train_dates: tuple[pd.Timestamp, ...],
    period_minutes: int,
) -> dict[str, dict[str, dict[str, float | str]]]:
    data = train.copy()
    data["period"] = (
        data["arrival_ts"].dt.hour * 60 + data["arrival_ts"].dt.minute
    ) // period_minutes
    counts = data.groupby(
        ["call_date", "service_class", "period"], observed=True
    ).size()
    result: dict[str, dict[str, dict[str, float | str]]] = {}
    for service_class in SERVICE_CLASSES:
        class_result: dict[str, dict[str, float | str]] = {}
        class_values = data.loc[data["service_class"].eq(service_class)]
        global_mean = max(len(class_values) / max(len(train_dates) * (1440 // period_minutes), 1), 1e-6)
        for weekday in WEEKDAYS:
            weekday_dates = tuple(d for d in train_dates if d.day_name().lower() == weekday)
            for period in range(1440 // period_minutes):
                samples = np.array([
                    float(counts.get((date, service_class, period), 0))
                    for date in weekday_dates
                ])
                scope = "weekday_period"
                if len(samples) < 2:
                    samples = np.array([
                        float(counts.get((date, service_class, period), 0))
                        for date in train_dates
                    ])
                    scope = "period"
                mean, dispersion = _negative_binomial_moments(samples, global_mean)
                class_result[f"{weekday}:{period}"] = {
                    "mean": mean,
                    "dispersion": dispersion,
                    "fit_scope": scope,
                    "n_days": int(len(samples)),
                }
        result[service_class] = class_result
    return result


def _negative_binomial_moments(
    samples: np.ndarray,
    fallback_mean: float,
) -> tuple[float, float]:
    if samples.size == 0:
        return float(fallback_mean), 1_000_000.0
    mean = float(samples.mean())
    variance = float(samples.var(ddof=1)) if samples.size > 1 else mean
    if mean <= 0:
        mean = max(float(fallback_mean), 1e-6)
    dispersion = mean * mean / (variance - mean) if variance > mean + 1e-12 else 1_000_000.0
    return mean, float(max(dispersion, 1e-6))


def _fit_priority_rates(train: pd.DataFrame) -> dict[str, float]:
    global_rate = float(train["priority"].mean()) if len(train) else 0.0
    return {
        service_class: float(
            train.loc[train["service_class"].eq(service_class), "priority"].mean()
        ) if train["service_class"].eq(service_class).any() else global_rate
        for service_class in SERVICE_CLASSES
    }


def _empirical_quantiles(frame: pd.DataFrame, column: str) -> dict[str, list[float]]:
    positive_all = frame.loc[frame[column].gt(0), column].astype(float)
    if positive_all.empty:
        positive_all = pd.Series([30.0], dtype=float)
    probabilities = np.linspace(0.0, 1.0, 101)
    result: dict[str, list[float]] = {}
    for service_class in SERVICE_CLASSES:
        values = frame.loc[
            frame["service_class"].eq(service_class) & frame[column].gt(0), column
        ].astype(float)
        if len(values) < 2:
            values = positive_all
        result[service_class] = [float(v) for v in np.quantile(values, probabilities)]
    return result


def _fit_disruption(
    train: pd.DataFrame,
    train_dates: tuple[pd.Timestamp, ...],
) -> dict:
    daily = train.groupby("call_date", observed=True).size().reindex(train_dates, fill_value=0)
    q75, q95 = (float(v) for v in np.quantile(daily.to_numpy(dtype=float), [0.75, 0.95]))
    states = np.where(daily > q95, 2, np.where(daily > q75, 1, 0)).astype(int)
    transitions = np.ones((3, 3), dtype=float)
    for left, right in zip(states[:-1], states[1:]):
        transitions[left, right] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)
    base = float(daily[states == 0].mean()) if np.any(states == 0) else float(daily.mean())
    multipliers = []
    for state in range(3):
        value = float(daily[states == state].mean()) if np.any(states == state) else base
        multipliers.append(max(value / max(base, 1e-6), 1.0))
    return {
        "thresholds": {"high": q75, "severe": q95},
        "transition_matrix": transitions.tolist(),
        "arrival_multipliers": multipliers,
        "training_state_counts": {
            "normal": int((states == 0).sum()),
            "high": int((states == 1).sum()),
            "severe": int((states == 2).sum()),
        },
    }


def _json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
