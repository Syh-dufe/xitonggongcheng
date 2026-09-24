import json
from pathlib import Path

import pandas as pd
import pytest

from prescriptive_capacity_sim.policy_data import export_policy_data


def _write_log(
    tmp_path: Path,
    *,
    splits: list[str],
    episodes: list[int],
    actions: list[int],
) -> Path:
    rows = len(actions)
    frame = pd.DataFrame({
        "episode_id": episodes,
        "split": splits,
        "period": list(range(rows)),
        "queue_regular": [1] * rows,
        "queue_specialist": [0] * rows,
        "queue_callback_special": [0] * rows,
        "queue_priority": [0] * rows,
        "max_waited_periods": [0] * rows,
        "demand_state": ["normal"] * rows,
        "action": actions,
        "total_cost": [10.0] * rows,
        "next_queue": [1] * rows,
    })
    source = tmp_path / "observed_log.csv"
    frame.to_csv(source, index=False)
    return source


def _write_valid_log(tmp_path: Path) -> Path:
    return _write_log(
        tmp_path,
        splits=["train"] * 4 + ["test"] * 4,
        episodes=list(range(8)),
        actions=[0, 1, 2, 3, 0, 1, 2, 3],
    )


def test_export_policy_data_keeps_only_factual_columns(tmp_path: Path) -> None:
    log = pd.DataFrame({
        "episode_id": [0, 1, 2, 3],
        "split": ["train", "train", "test", "test"],
        "period": [0, 0, 1, 1],
        "queue_regular": [1, 2, 3, 4],
        "queue_specialist": [0, 1, 1, 0],
        "queue_callback_special": [0, 0, 0, 1],
        "queue_priority": [0, 1, 0, 1],
        "max_waited_periods": [0, 1, 1, 2],
        "demand_state": ["normal", "high", "normal", "severe"],
        "action": [0, 1, 2, 3],
        "total_cost": [10.0, 11.0, 12.0, 13.0],
        "next_queue": [2, 3, 4, 5],
        "potential_cost_a0": [1.0] * 4,
        "oracle_action": [0] * 4,
    })
    source = tmp_path / "observed_log.csv"
    log.to_csv(source, index=False)

    result = export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)

    train = pd.read_csv(result.training_path)
    assert train.columns.tolist() == [
        "episode_id",
        "split",
        "period",
        "queue_regular",
        "queue_specialist",
        "queue_callback_special",
        "queue_priority",
        "max_waited_periods",
        "demand_state",
        "action",
        "cost",
    ]
    assert train["cost"].tolist() == [14.0, 17.0]


def test_export_policy_data_rejects_overlapping_episode_ids(tmp_path: Path) -> None:
    source = _write_log(
        tmp_path,
        splits=["train", "test"],
        episodes=[7, 7],
        actions=[0, 1],
    )

    with pytest.raises(ValueError, match="episode_id"):
        export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)


def test_export_policy_data_rejects_missing_action_support(tmp_path: Path) -> None:
    source = _write_log(
        tmp_path,
        splits=["train"] * 4 + ["test"] * 4,
        episodes=list(range(8)),
        actions=[0, 1, 2, 0, 0, 1, 2, 3],
    )

    with pytest.raises(ValueError, match="four actions"):
        export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)


def test_export_policy_data_manifest_declares_no_counterfactuals(tmp_path: Path) -> None:
    source = _write_valid_log(tmp_path)

    result = export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counterfactual_data_used"] is False
    assert manifest["action_counts_train"] == {"0": 1, "1": 1, "2": 1, "3": 1}
