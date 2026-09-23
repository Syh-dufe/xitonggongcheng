"""Command-line workflow for calibration, generation, and benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .calibration import calibrate_calls, write_calibration
from .capacity_validation import (
    RESOURCE_COLUMNS,
    SELECTION_WEIGHTS,
    load_candidates,
    simulate_historical_periods,
    summarize_candidates,
)
from .config import SimulationConfig
from .evaluation import evaluate_policies
from .ingest import load_raw_directory
from .logging import generate_dataset
from .policies import default_policies
from .validation import compare_real_and_simulated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and run the semisynthetic call-center simulator."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--raw-dir", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--train-fraction", type=float, default=0.70)
    validate_capacity = commands.add_parser("validate-capacity")
    validate_capacity.add_argument("--config", required=True, type=Path)
    validate_capacity.add_argument("--candidates", required=True, type=Path)
    validate_capacity.add_argument("--days", required=True, type=int)
    validate_capacity.add_argument("--seeds", required=True, type=int, nargs="+")
    validate_capacity.add_argument("--output", required=True, type=Path)
    for name in ("generate", "benchmark"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--days", required=True, type=int)
        child.add_argument("--seed", required=True, type=int)
        child.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        return _calibrate(args.raw_dir, args.output, args.train_fraction)
    if args.command == "validate-capacity":
        return _validate_capacity(
            args.config, args.candidates, args.days, args.seeds, args.output
        )
    return _simulate(args.config, args.days, args.seed, args.output, args.command)


def _calibrate(raw_dir: Path, output: Path, train_fraction: float) -> int:
    ingested = load_raw_directory(raw_dir)
    result = calibrate_calls(
        ingested.calls, quality=ingested.quality, train_fraction=train_fraction
    )
    sources = [raw_dir / name for name in ingested.source_files]
    write_calibration(result, output, source_files=sources)
    return 0


def _simulate(
    config_path: Path,
    days: int,
    seed: int,
    output: Path,
    command: str,
) -> int:
    config = SimulationConfig.from_yaml(config_path)
    config.validate_for_simulation()
    parameters = json.loads(config.calibration_path.read_text(encoding="utf-8"))
    manifest_path = config.calibration_path.with_name("calibration_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibration manifest not found: {manifest_path}")
    calibration_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameter_hash = _json_hash(parameters)
    if calibration_manifest.get("parameters_sha256") != parameter_hash:
        raise ValueError("calibration parameter hash does not match its manifest")

    data = generate_dataset(config, parameters, days=days, seed=seed)
    metrics = evaluate_policies(
        config, parameters, default_policies(config), days=days, seed=seed
    )
    interval_path = config.calibration_path.with_name("calibration_intervals.csv")
    intervals = pd.read_csv(interval_path)
    validation_intervals = intervals.loc[intervals["split"].eq("validation")]
    validation_calls = _load_validation_reference_calls(config.calibration_path)
    validation = compare_real_and_simulated(
        validation_intervals, data.observed, validation_calls, parameters
    )
    output.mkdir(parents=True, exist_ok=True)
    data.observed.to_csv(output / "observed_log.csv", index=False)
    data.oracle.to_csv(output / "oracle_counterfactuals.csv", index=False)
    data.episodes.to_csv(output / "episode_summary.csv", index=False)
    metrics.to_csv(output / "policy_metrics.csv", index=False)
    validation.to_csv(output / "simulation_validation.csv", index=False)
    run_manifest = {
        "command": command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory",
        "config_path": str(config_path.resolve()),
        "config": _serialize_config(config),
        "calibration_parameters_sha256": parameter_hash,
        "validation_reference_calls_sha256": calibration_manifest.get(
            "validation_reference_calls_sha256"
        ),
        "raw_source_sha256": calibration_manifest.get("source_sha256", {}),
        "days": days,
        "seed": seed,
        "observed_rows": len(data.observed),
        "oracle_rows": len(data.oracle),
        "episode_rows": len(data.episodes),
        "policy_rows": len(metrics),
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": version("prescriptive-capacity-sim"),
        "git_revision": _git_revision(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def _validate_capacity(
    config_path: Path,
    candidates_path: Path,
    days: int,
    seeds: list[int],
    output: Path,
) -> int:
    """Run factual-only held-out diagnostics for resource candidates."""

    if days <= 0:
        raise ValueError("days must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    config, parameters, calibration_manifest = _load_calibration_artifacts(config_path)
    candidates = load_candidates(candidates_path)
    interval_path = config.calibration_path.with_name("calibration_intervals.csv")
    intervals = pd.read_csv(interval_path)
    validation_intervals = intervals.loc[intervals["split"].eq("validation")].copy()
    if validation_intervals.empty:
        raise ValueError("calibration intervals contain no validation split")
    validation_calls = _load_validation_reference_calls(config.calibration_path)
    weekdays, weekday_schedule_source, validation_dates = _validation_weekday_schedule(
        validation_intervals, days
    )

    run_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        candidate_config = config.with_overrides(
            resources=candidate.resource_overrides()
        )
        for seed in seeds:
            simulated = simulate_historical_periods(
                candidate_config, parameters, days=days, seed=seed, weekdays=weekdays
            )
            comparison = compare_real_and_simulated(
                validation_intervals, simulated, validation_calls, parameters
            )
            comparison.insert(0, "seed", seed)
            comparison.insert(0, "candidate", candidate.name)
            for column, value in candidate.resource_overrides().items():
                comparison[column] = value
            run_frames.append(comparison)

    runs = pd.concat(run_frames, ignore_index=True)
    runs = runs[
        ["candidate", "seed", "metric", "service_class", "real", "simulated", "error",
         *RESOURCE_COLUMNS]
    ]
    summary = summarize_candidates(runs)
    parameter_hash = _json_hash(parameters)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "candidate_validation_runs.csv", index=False)
    summary.to_csv(output / "candidate_validation_summary.csv", index=False)
    report_manifest = {
        "command": "validate-capacity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "factual_only_semisynthetic_validation",
        "config_path": str(config_path.resolve()),
        "config": _serialize_config(config),
        "calibration_parameters_sha256": parameter_hash,
        "validation_reference_calls_sha256": calibration_manifest.get(
            "validation_reference_calls_sha256"
        ),
        "raw_source_sha256": calibration_manifest.get("source_sha256", {}),
        "candidate_yaml_path": str(candidates_path.resolve()),
        "candidate_yaml_sha256": _file_hash(candidates_path),
        "candidates": [
            {"name": candidate.name, **candidate.resource_overrides()}
            for candidate in candidates
        ],
        "days": days,
        "seeds": list(seeds),
        "validation_dates_available": validation_dates,
        "weekday_schedule": list(weekdays),
        "weekday_schedule_source": weekday_schedule_source,
        "selection_weights": SELECTION_WEIGHTS,
        "counterfactual_data_used": False,
        "runs_rows": len(runs),
        "summary_rows": len(summary),
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": version("prescriptive-capacity-sim"),
        "git_revision": _git_revision(),
    }
    (output / "candidate_validation_manifest.json").write_text(
        json.dumps(report_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def _load_calibration_artifacts(
    config_path: Path,
) -> tuple[SimulationConfig, dict, dict]:
    config = SimulationConfig.from_yaml(config_path)
    config.validate_for_simulation()
    parameters = json.loads(config.calibration_path.read_text(encoding="utf-8"))
    manifest_path = config.calibration_path.with_name("calibration_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibration manifest not found: {manifest_path}")
    calibration_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if calibration_manifest.get("parameters_sha256") != _json_hash(parameters):
        raise ValueError("calibration parameter hash does not match its manifest")
    return config, parameters, calibration_manifest


def _load_validation_reference_calls(calibration_path: Path) -> pd.DataFrame:
    """Load the exact, deidentified held-out calls used for diagnostics."""

    path = calibration_path.with_name("validation_reference_calls.csv")
    if not path.is_file():
        raise FileNotFoundError(f"validation reference calls not found: {path}")
    frame = pd.read_csv(path)
    expected = ["service_class", "queue_seconds", "service_seconds"]
    if frame.columns.tolist() != expected:
        raise ValueError(
            "validation reference calls must contain only: " + ", ".join(expected)
        )
    return frame


def _validation_weekday_schedule(
    validation_intervals: pd.DataFrame,
    days: int,
) -> tuple[tuple[str, ...], str, list[str]]:
    """Return validation-date weekdays, cycling only when episode counts differ."""

    required = {"call_date", "weekday"}
    if not required.issubset(validation_intervals.columns):
        raise ValueError("validation intervals must include call_date and weekday")
    dates = validation_intervals[["call_date", "weekday"]].drop_duplicates()
    dates = dates.sort_values("call_date", kind="stable")
    date_values = dates["call_date"].astype(str).tolist()
    weekday_values = tuple(dates["weekday"].astype(str).str.lower())
    if not weekday_values:
        raise ValueError("validation intervals contain no validation dates")
    if days == len(weekday_values):
        return weekday_values, "validation_dates_exact", date_values
    if days < len(weekday_values):
        return weekday_values[:days], "validation_dates_truncated", date_values
    schedule = tuple(weekday_values[index % len(weekday_values)] for index in range(days))
    return schedule, "validation_dates_cycled", date_values


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_config(config: SimulationConfig) -> dict:
    value = asdict(config)
    value["calibration_path"] = str(config.calibration_path)
    return value


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
