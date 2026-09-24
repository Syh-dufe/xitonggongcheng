import json
from pathlib import Path

import pandas as pd

from prescriptive_capacity_sim.cli import _validation_weekday_schedule, build_parser, main


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_cli_parses_export_policy_data():
    args = build_parser().parse_args([
        "export-policy-data",
        "--observed-log", "outputs/historical_factual_730d/observed_log.csv",
        "--queue-penalty", "2.0",
        "--output", "outputs/policy_data",
    ])

    assert args.command == "export-policy-data"
    assert args.queue_penalty == 2.0


def test_calibrate_cli_writes_exact_deidentified_validation_reference(tmp_path):
    output = tmp_path / "calibration"
    assert main([
        "calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(output)
    ]) == 0
    assert {path.name for path in output.iterdir()} == {
        "calibration_intervals.csv", "calibration_parameters.json",
        "data_quality_report.json", "calibration_manifest.json",
        "validation_reference_calls.csv",
    }
    reference = pd.read_csv(output / "validation_reference_calls.csv")
    assert reference.columns.tolist() == [
        "service_class", "queue_seconds", "service_seconds",
    ]


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
    pd.DataFrame({
        "service_class": ["regular"] * 10 + ["specialist"],
        "queue_seconds": [60.0] * 10 + [600.0],
        "service_seconds": [60.0] * 11,
    }).to_csv(calibration / "validation_reference_calls.csv", index=False)
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
    validation = pd.read_csv(output / "simulation_validation.csv")
    overall_p90 = validation.loc[
        validation["metric"].eq("overall_p90_wait_relative_error")
    ].iloc[0]
    assert overall_p90["real"] == 1.0


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
        "regular_agents", "specialist_agents", "cross_skill_efficiency",
    }.issubset(runs.columns)
    assert "supervisor_emergency_agents" not in runs.columns
    assert not any("oracle" in column or "potential_" in column for column in runs.columns)
    manifest = json.loads(
        (output / "candidate_validation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counterfactual_data_used"] is False
    assert manifest["weekday_schedule_source"] == "validation_dates_cycled"


def test_validation_weekday_schedule_reports_exact_truncated_and_cycled_sources():
    intervals = pd.DataFrame({
        "call_date": ["1999-01-01", "1999-01-02", "1999-01-03"],
        "weekday": ["friday", "saturday", "sunday"],
    })

    exact, exact_source, _ = _validation_weekday_schedule(intervals, days=3)
    truncated, truncated_source, _ = _validation_weekday_schedule(intervals, days=2)
    cycled, cycled_source, _ = _validation_weekday_schedule(intervals, days=5)

    assert exact == ("friday", "saturday", "sunday")
    assert exact_source == "validation_dates_exact"
    assert truncated == ("friday", "saturday")
    assert truncated_source == "validation_dates_truncated"
    assert cycled == ("friday", "saturday", "sunday", "friday", "saturday")
    assert cycled_source == "validation_dates_cycled"
