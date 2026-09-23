import json
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


def test_validate_capacity_writes_factual_reports_and_manifest(tmp_path):
    calibration = tmp_path / "calibration"
    assert main([
        "calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(calibration)
    ]) == 0
    config = tmp_path / "config.yaml"
    config.write_text(
        f"calibration_path: {calibration.joinpath('calibration_parameters.json').as_posix()}\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        "candidates:\n"
        "  - name: baseline\n"
        "    regular_agents: 8\n"
        "    specialist_agents: 5\n"
        "    supervisor_emergency_agents: 1\n"
        "    cross_skill_efficiency: 0.75\n",
        encoding="utf-8",
    )
    output = tmp_path / "validation"

    assert main([
        "validate-capacity", "--config", str(config), "--candidates", str(candidates),
        "--days", "2", "--seeds", "17", "19", "--output", str(output),
    ]) == 0

    assert {item.name for item in output.iterdir()} == {
        "candidate_validation_runs.csv", "candidate_validation_summary.csv",
        "candidate_validation_manifest.json",
    }
    runs = pd.read_csv(output / "candidate_validation_runs.csv")
    assert {
        "candidate", "seed", "metric", "service_class", "real", "simulated", "error",
        "regular_agents", "specialist_agents", "supervisor_emergency_agents",
        "cross_skill_efficiency",
    }.issubset(runs.columns)
    assert not any("oracle" in column or "potential_" in column for column in runs.columns)
    manifest = json.loads(
        (output / "candidate_validation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counterfactual_data_used"] is False
    assert manifest["weekday_schedule_source"] == "validation_dates_cycled"
