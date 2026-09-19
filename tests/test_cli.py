import json

import pandas as pd

from prescriptive_capacity_sim.cli import main


def test_generate_cli_writes_complete_artifact_set(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text("periods_per_day: 2\n", encoding="utf-8")
    output = tmp_path / "run"
    exit_code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--days",
            "2",
            "--seed",
            "101",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    required = {
        "observed_log.csv",
        "oracle_counterfactuals.csv",
        "episode_summary.csv",
        "policy_metrics.csv",
        "run_manifest.json",
    }
    assert required == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 101
    assert manifest["days"] == 2
    assert manifest["observed_rows"] == 4
    observed = pd.read_csv(output / "observed_log.csv")
    oracle = pd.read_csv(output / "oracle_counterfactuals.csv")
    assert len(observed) == len(oracle) == 4
    assert "oracle_action" not in observed.columns
