import json
from pathlib import Path

import pandas as pd
import pytest

from prescriptive_capacity_sim.calibration import calibrate_calls, write_calibration
from prescriptive_capacity_sim.ingest import load_month


FIXTURE = Path(__file__).parent / "fixtures" / "calls_sample.txt"


def _sample_calls():
    frame, quality = load_month(FIXTURE)
    return frame, quality


def test_split_is_chronological_and_uses_train_dates_only():
    calls, quality = _sample_calls()
    result = calibrate_calls(calls, quality=quality, train_fraction=0.70)
    assert max(result.train_dates) < min(result.validation_dates)
    assert result.parameters["fit_scope"] == "train_only"


def test_half_hour_aggregation_conserves_training_arrivals():
    calls, quality = _sample_calls()
    result = calibrate_calls(calls, quality=quality, train_fraction=0.70)
    assert result.intervals["arrivals"].sum() == len(result.train_calls)


def test_calibration_records_empirical_distributions(tmp_path):
    calls, quality = _sample_calls()
    result = calibrate_calls(calls, quality=quality, train_fraction=0.70)
    paths = write_calibration(result, tmp_path, source_files=(FIXTURE,))
    params = json.loads(paths.parameters.read_text(encoding="utf-8"))
    assert set(params["classes"]) == {
        "regular", "specialist", "callback_special"
    }
    assert "service_seconds" in params["empirical"]
    assert "patience_seconds" in params["empirical"]
    assert paths.intervals.is_file()
    assert paths.quality.is_file()
    assert paths.manifest.is_file()
    assert paths.validation_reference.is_file()
    validation_reference = pd.read_csv(paths.validation_reference)
    assert validation_reference.columns.tolist() == [
        "service_class", "queue_seconds", "service_seconds",
    ]
    assert len(validation_reference) == len(result.validation_calls)
    assert {"customer_id", "server", "raw_row"}.isdisjoint(
        validation_reference.columns
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["validation_reference_calls_sha256"]
    quality_report = json.loads(paths.quality.read_text(encoding="utf-8"))
    assert "validation_reference" not in quality_report


def test_analysis_artifacts_contain_no_identifiers(tmp_path):
    calls, quality = _sample_calls()
    result = calibrate_calls(calls, quality=quality, train_fraction=0.70)
    paths = write_calibration(result, tmp_path, source_files=(FIXTURE,))
    header = paths.intervals.read_text(encoding="utf-8").splitlines()[0]
    assert "customer_id" not in header
    assert "server" not in header
    reference_header = paths.validation_reference.read_text(encoding="utf-8").splitlines()[0]
    assert "customer_id" not in reference_header
    assert "server" not in reference_header


def test_disruption_multipliers_preserve_overall_arrival_scale():
    calls, quality = _sample_calls()
    result = calibrate_calls(calls, quality=quality, train_fraction=0.70)
    disruption = result.parameters["disruption"]
    counts = disruption["training_state_counts"]
    weights = [counts["normal"], counts["high"], counts["severe"]]
    multipliers = disruption["arrival_multipliers"]
    weighted_mean = sum(w * m for w, m in zip(weights, multipliers)) / sum(weights)
    assert weighted_mean == pytest.approx(1.0)
