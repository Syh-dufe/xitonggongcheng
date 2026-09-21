from pathlib import Path

from prescriptive_capacity_sim.ingest import load_month, load_raw_directory


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parser_maps_three_classes():
    frame, quality = load_month(FIXTURE_DIR / "calls_sample.txt")
    assert set(frame["service_class"]) == {
        "regular", "specialist", "callback_special"
    }
    assert quality.total_rows == 6
    assert quality.unknown_type_rows == 1
    assert quality.phantom_rows == 1
    assert quality.prequeue_exit_rows == 1


def test_analysis_frame_is_deidentified():
    frame, _ = load_month(FIXTURE_DIR / "calls_sample.txt")
    assert {"customer_id", "server", "source_row"}.isdisjoint(frame.columns)


def test_directory_loader_is_deterministic():
    left = load_raw_directory(FIXTURE_DIR)
    right = load_raw_directory(FIXTURE_DIR)
    assert left.calls.equals(right.calls)
    assert left.quality == right.quality


def test_service_and_patience_samples_are_identified():
    frame, _ = load_month(FIXTURE_DIR / "calls_sample.txt")
    assert frame.loc[frame["outcome"] == "AGENT", "service_seconds"].tolist() == [120, 240]
    assert frame.loc[frame["outcome"] == "HANG", "patience_seconds"].tolist() == [60]
