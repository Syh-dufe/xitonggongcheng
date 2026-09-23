from pathlib import Path

import pandas as pd

from prescriptive_capacity_sim.cli import main


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_calibrate_cli_writes_four_artifacts(tmp_path):
    output = tmp_path / "calibration"
    assert main([
        "calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(output)
    ]) == 0
    assert {path.name for path in output.iterdir()} == {
        "calibration_intervals.csv", "calibration_parameters.json",
        "data_quality_report.json", "calibration_manifest.json",
    }


def test_generate_cli_writes_six_artifacts_and_manifest(tmp_path):
    calibration = tmp_path / "calibration"
    assert main([
        "calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(calibration)
    ]) == 0
    config = tmp_path / "config.yaml"
    config.write_text(
        f"calibration_path: {calibration.joinpath('calibration_parameters.json').as_posix()}\n"
        "periods_per_day: 4\n",
        encoding="utf-8",
    )
    output = tmp_path / "simulation"
    assert main([
        "generate", "--config", str(config), "--days", "2", "--seed", "17",
        "--output", str(output),
    ]) == 0
    assert {path.name for path in output.iterdir()} == {
        "observed_log.csv", "oracle_counterfactuals.csv",
        "episode_summary.csv", "policy_metrics.csv",
        "simulation_validation.csv", "run_manifest.json",
    }
    observed = pd.read_csv(output / "observed_log.csv")
    assert {
        "exit_wait_regular",
        "exit_wait_specialist",
        "exit_wait_callback_special",
    }.issubset(observed.columns)


def test_cli_csv_outputs_never_contain_identifiers(tmp_path):
    calibration = tmp_path / "calibration"
    main(["calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(calibration)])
    for csv_path in calibration.glob("*.csv"):
        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        assert {"customer_id", "server"}.isdisjoint(columns)
