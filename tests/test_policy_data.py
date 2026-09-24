from pathlib import Path

import pandas as pd

from prescriptive_capacity_sim.policy_data import export_policy_data


def test_export_policy_data_keeps_only_factual_columns(tmp_path: Path) -> None:
    log = pd.DataFrame({
        "episode_id": [0, 1, 0, 1],
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
