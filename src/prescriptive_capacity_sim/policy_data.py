"""Export factual simulator logs for external policy-learning tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    _validate_factual_log(frame)
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
    manifest_path = output / "policy_manifest.json"
    training = selected.loc[selected["split"].eq("train")]
    test = selected.loc[selected["split"].eq("test")]
    manifest = {
        "source_observed_log_sha256": _file_hash(source),
        "feature_columns": list(FEATURE_COLUMNS),
        "action_column": "action",
        "cost_column": "cost",
        "cost_definition": "total_cost + queue_penalty * next_queue",
        "queue_penalty": queue_penalty,
        "train_episode_count": int(training["episode_id"].nunique()),
        "test_episode_count": int(test["episode_id"].nunique()),
        "action_counts_train": {
            str(action): int((training["action"] == action).sum())
            for action in range(4)
        },
        "counterfactual_data_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PolicyDataExport(
        training_path=training_path,
        test_path=test_path,
        manifest_path=manifest_path,
    )


def _validate_factual_log(frame: pd.DataFrame) -> None:
    required = {
        "episode_id",
        "split",
        *FEATURE_COLUMNS,
        "action",
        "total_cost",
        "next_queue",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("observed log is missing required factual columns: " + ", ".join(missing))
    if frame.loc[:, sorted(required)].isna().any().any():
        raise ValueError("observed log contains missing policy-learning values")
    if set(frame["split"]) != {"train", "test"}:
        raise ValueError("observed log must contain train and test splits")
    train_episodes = set(frame.loc[frame["split"].eq("train"), "episode_id"])
    test_episodes = set(frame.loc[frame["split"].eq("test"), "episode_id"])
    if train_episodes.intersection(test_episodes):
        raise ValueError("episode_id occurs in both train and test splits")
    train_actions = set(frame.loc[frame["split"].eq("train"), "action"])
    if train_actions != {0, 1, 2, 3}:
        raise ValueError("training data must contain all four actions")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
