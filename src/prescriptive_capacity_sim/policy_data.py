"""Export factual simulator logs for external policy-learning tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = (
    "period",
    "queue_regular",
    "queue_specialist",
    "queue_callback_special",
    "queue_priority",
    "max_waited_periods",
    "demand_state",
)
POLICY_COLUMNS = ("episode_id", "split", *FEATURE_COLUMNS, "action", "cost")


@dataclass(frozen=True)
class PolicyDataExport:
    """Paths produced by :func:`export_policy_data`."""

    training_path: Path
    test_path: Path
    manifest_path: Path


def export_policy_data(
    source: Path,
    output: Path,
    *,
    queue_penalty: float,
) -> PolicyDataExport:
    """Create train/test factual policy-learning tables from an observed log."""

    frame = pd.read_csv(source)
    selected = frame.loc[
        :,
        [
            "episode_id",
            "split",
            *FEATURE_COLUMNS,
            "action",
            "total_cost",
            "next_queue",
        ],
    ].copy()
    selected["cost"] = (
        selected.pop("total_cost") + queue_penalty * selected.pop("next_queue")
    )
    output.mkdir(parents=True, exist_ok=True)
    training_path = output / "policy_training.csv"
    test_path = output / "policy_test.csv"
    selected.loc[selected["split"].eq("train"), POLICY_COLUMNS].to_csv(
        training_path, index=False
    )
    selected.loc[selected["split"].eq("test"), POLICY_COLUMNS].to_csv(
        test_path, index=False
    )
    return PolicyDataExport(
        training_path=training_path,
        test_path=test_path,
        manifest_path=output / "policy_manifest.json",
    )
