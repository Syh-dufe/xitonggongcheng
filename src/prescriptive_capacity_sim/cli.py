"""Command-line entry points for dataset generation and policy benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .calibration import fit_empirical_calibration
from .callcenter_data import load_callcenter_month
from .callcenter_scenarios import (
    CostConfig,
    StaffingBehaviorPolicy,
    build_replay_day,
    generate_semisynthetic_day,
)
from .config import SimulationConfig
from .evaluation import evaluate_policies
from .logging import generate_dataset
from .policies import default_policies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate biased capacity logs and leakage-safe counterfactuals."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "benchmark"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--days", required=True, type=int)
        child.add_argument("--seed", required=True, type=int)
        child.add_argument("--output", required=True, type=Path)
    callcenter = subparsers.add_parser("callcenter-generate")
    callcenter.add_argument("--data-dir", required=True, type=Path)
    callcenter.add_argument("--config", required=True, type=Path)
    callcenter.add_argument("--seed", required=True, type=int)
    callcenter.add_argument("--output", required=True, type=Path)
    return parser


_MONTH_NAMES = {
    "January.txt",
    "February.txt",
    "March.txt",
    "April.txt",
    "May.txt",
    "June.txt",
    "July.txt",
    "August.txt",
    "September.txt",
    "October.txt",
    "November.txt",
    "December.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_days(
    days: list, calibration_fraction: float, validation_fraction: float
) -> dict:
    if len(days) < 2:
        raise ValueError("callcenter workflow requires at least two distinct days")
    if not 0 < calibration_fraction < 1:
        raise ValueError("split.calibration must be between zero and one")
    if validation_fraction < 0 or calibration_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a nonempty test partition")
    calibration_count = max(1, int(len(days) * calibration_fraction))
    validation_count = int(len(days) * validation_fraction)
    if validation_fraction > 0 and validation_count == 0 and len(days) >= 3:
        validation_count = 1
    if calibration_count + validation_count >= len(days):
        validation_count = max(0, len(days) - calibration_count - 1)
    return {
        day: (
            "calibration"
            if index < calibration_count
            else "validation"
            if index < calibration_count + validation_count
            else "test"
        )
        for index, day in enumerate(days)
    }


def _run_callcenter(args: argparse.Namespace) -> int:
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    files = sorted(args.data_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"no .txt call-center files found in {args.data_dir}")
    present_names = {path.name for path in files}
    if present_names & _MONTH_NAMES and not _MONTH_NAMES <= present_names:
        missing = sorted(_MONTH_NAMES - present_names)
        raise ValueError(f"incomplete twelve-month dataset; missing: {missing}")

    included_types = raw_config.get("included_types")
    type_filter = set(included_types) if included_types is not None else None
    calls = [
        call
        for path in files
        for call in load_callcenter_month(path, included_types=type_filter)
    ]
    days = sorted({call.queue_entry.date() for call in calls})
    split_config = raw_config.get("split", {})
    partitions = _partition_days(
        days,
        float(split_config.get("calibration", 0.6)),
        float(split_config.get("validation", 0.2)),
    )
    calibration_calls = [
        call for call in calls if partitions[call.queue_entry.date()] == "calibration"
    ]
    calibration = fit_empirical_calibration(calibration_calls)

    interval_seconds = int(raw_config.get("interval_seconds", 1800))
    intervals_per_day = int(raw_config.get("intervals_per_day", 48))
    actions = tuple(int(value) for value in raw_config.get("actions", [2, 3, 4, 5, 6, 7, 8]))
    behavior = StaffingBehaviorPolicy(
        actions=actions, **raw_config.get("behavior", {})
    )
    costs = CostConfig(**raw_config.get("costs", {}))

    observed_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(days):
        replay = build_replay_day(
            calls,
            calibration,
            day,
            seed=args.seed + 2 * day_index,
            interval_seconds=interval_seconds,
            intervals_per_day=intervals_per_day,
        )
        generated = generate_semisynthetic_day(
            replay,
            behavior_policy=behavior,
            costs=costs,
            seed=args.seed + 2 * day_index + 1,
        )
        partition = partitions[day]
        observed_rows.extend({**row, "partition": partition} for row in generated.observed)
        oracle_rows.extend({**row, "partition": partition} for row in generated.oracle)
        daily_rows.append({**generated.daily_metrics, "partition": partition})

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observed_rows).to_csv(output / "observed_log.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(
        output / "oracle_counterfactuals.csv", index=False
    )
    pd.DataFrame(daily_rows).to_csv(output / "daily_metrics.csv", index=False)

    service_values = calibration.pooled_service_seconds
    summary = {
        "calibration_calls": len(calibration_calls),
        "service_observations": int(len(service_values)),
        "service_seconds": {
            "mean": float(np.mean(service_values)),
            "p50": float(np.quantile(service_values, 0.5)),
            "p90": float(np.quantile(service_values, 0.9)),
        },
        "service_observations_by_type": {
            name: int(len(values))
            for name, values in calibration.service_seconds_by_type.items()
        },
        "patience_event_times": int(len(calibration.patience_support_seconds)),
        "date_min": days[0].isoformat(),
        "date_max": days[-1].isoformat(),
    }
    (output / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "command": "callcenter-generate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config": raw_config,
        "seed": args.seed,
        "data_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "included_calls": len(calls),
        "days": len(days),
        "observed_rows": len(observed_rows),
        "oracle_rows": len(oracle_rows),
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": version("prescriptive-capacity-sim"),
        "status": "semi-synthetic",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "callcenter-generate":
        return _run_callcenter(args)
    config = SimulationConfig.from_yaml(args.config)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    data = generate_dataset(config, days=args.days, seed=args.seed)
    metrics = evaluate_policies(
        config,
        default_policies(config),
        days=args.days,
        seed=args.seed,
    )
    data.observed.to_csv(output / "observed_log.csv", index=False)
    data.oracle.to_csv(output / "oracle_counterfactuals.csv", index=False)
    data.episodes.to_csv(output / "episode_summary.csv", index=False)
    metrics.to_csv(output / "policy_metrics.csv", index=False)

    manifest = {
        "command": args.command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config": asdict(config),
        "days": args.days,
        "seed": args.seed,
        "observed_rows": len(data.observed),
        "oracle_rows": len(data.oracle),
        "episode_rows": len(data.episodes),
        "policy_rows": len(metrics),
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": version("prescriptive-capacity-sim"),
        "status": "exploratory",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

