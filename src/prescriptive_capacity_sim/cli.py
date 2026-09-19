"""Command-line entry points for dataset generation and policy benchmarking."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

