from __future__ import annotations

import json

import pandas as pd
import yaml

from prescriptive_capacity_sim.cli import main


HEADER = (
    "vru.line\tcall_id\tcustomer_id\tpriority\ttype\tdate\tvru_entry\t"
    "vru_exit\tvru_time\tq_start\tq_exit\tq_time\toutcome\tser_start\t"
    "ser_exit\tser_time\tserver\tday.of.week"
)


def _row(
    number: int,
    call_id: int,
    date: str,
    time: str,
    *,
    outcome: str,
    wait: int,
    service: int,
) -> str:
    fields = [
        number,
        "AA0101",
        call_id,
        f"customer-{call_id}",
        2,
        "PS",
        date,
        time,
        time,
        5,
        time,
        time,
        wait,
        outcome,
        time if outcome == "AGENT" else "0:00:00",
        time if outcome == "AGENT" else "0:00:00",
        service,
        f"agent-{call_id}" if outcome == "AGENT" else "NO_SERVER",
        "friday",
    ]
    return "\t".join(map(str, fields))


def test_callcenter_generate_writes_privacy_safe_outputs(tmp_path) -> None:
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    (data_dir / "fixture.txt").write_text(
        "\n".join(
            [
                HEADER,
                _row(1, 101, "990101", "0:05:00", outcome="AGENT", wait=5, service=30),
                _row(2, 102, "990101", "0:06:00", outcome="HANG", wait=20, service=0),
                _row(3, 103, "990102", "0:35:00", outcome="AGENT", wait=2, service=45),
                _row(4, 104, "990102", "0:36:00", outcome="HANG", wait=30, service=0),
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "included_types": ["PS"],
                "interval_seconds": 1800,
                "intervals_per_day": 2,
                "actions": [0, 1, 2],
                "behavior": {"temperature": 1.0},
                "split": {"calibration": 0.5, "validation": 0.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    exit_code = main(
        [
            "callcenter-generate",
            "--data-dir",
            str(data_dir),
            "--config",
            str(config),
            "--output",
            str(output),
            "--seed",
            "20260920",
        ]
    )

    assert exit_code == 0
    expected = {
        "calibration_summary.json",
        "observed_log.csv",
        "oracle_counterfactuals.csv",
        "daily_metrics.csv",
        "run_manifest.json",
    }
    assert expected <= {path.name for path in output.iterdir()}

    observed = pd.read_csv(output / "observed_log.csv")
    oracle = pd.read_csv(output / "oracle_counterfactuals.csv")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (output / "calibration_summary.json").read_text(encoding="utf-8")
    )

    assert set(observed["partition"]) == {"calibration", "test"}
    assert not any(column.startswith("potential_") for column in observed.columns)
    assert not {"customer_id", "server"} & set(observed.columns)
    assert {"potential_cost", "potential_abandoned"} <= set(oracle.columns)
    assert manifest["seed"] == 20260920
    assert len(manifest["data_files"]) == 1
    assert len(manifest["data_files"][0]["sha256"]) == 64
    assert summary["calibration_calls"] == 2
