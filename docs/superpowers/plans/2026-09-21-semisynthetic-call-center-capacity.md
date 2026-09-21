# Semisynthetic Call-Center Capacity Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the supply-chain simulator with a reproducible call-center capacity simulator calibrated from the downloaded Technion anonymous-bank logs.

**Architecture:** A calibration pipeline converts immutable monthly logs into de-identified half-hour aggregates, empirical distributions, and a versioned parameter file. A discrete-time two-pool queueing simulator consumes that parameter file, synthesizes biased historical capacity actions, and separates observed factual logs from common-random-number oracle counterfactuals.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, PyYAML, pytest, uv

---

## File map

- Recreate `pyproject.toml`, `.gitignore`, `README.md`, and `runs.md` with call-center semantics.
- Recreate `config.py`; add `ingest.py` and `calibration.py`.
- Recreate `state.py`, `demand.py`, `behavior.py`, and `environment.py` for a calibrated two-pool queue.
- Recreate `oracle.py`, `logging.py`, `policies.py`, and `evaluation.py`; add `validation.py`.
- Recreate `cli.py` with `calibrate`, `generate`, and `benchmark`.
- Replace scenario YAML files and tests.
- Add small synthetic fixtures; unit tests must not depend on the downloaded raw directory.

### Task 1: Package scaffold and validated configuration

**Files:**
- Recreate: `.gitignore`
- Recreate: `pyproject.toml`
- Recreate: `src/prescriptive_capacity_sim/__init__.py`
- Recreate: `src/prescriptive_capacity_sim/config.py`
- Recreate: `configs/baseline.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import pytest
from prescriptive_capacity_sim.config import SimulationConfig

def test_defaults_use_documented_agent_pools():
    cfg = SimulationConfig.default()
    assert cfg.resources.regular_agents == 8
    assert cfg.resources.specialist_agents == 5
    assert cfg.behavior.action_levels == (0.0, 0.1, 0.2, 0.3)

def test_missing_calibration_is_rejected(tmp_path: Path):
    cfg = SimulationConfig.default().with_overrides(
        calibration_path=tmp_path / "missing.json"
    )
    with pytest.raises(FileNotFoundError):
        cfg.validate_for_simulation()

def test_invalid_cross_skill_efficiency_is_rejected():
    with pytest.raises(ValueError):
        SimulationConfig.default().with_overrides(
            resources={"cross_skill_efficiency": 1.5}
        )
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_config.py -v`

Expected: import or API failure because the replacement configuration is absent.

- [ ] **Step 3: Implement minimal configuration**

Create frozen `DataConfig`, `ResourceConfig`, `BehaviorConfig`, `CostConfig`, `SafetyConfig`, and `SimulationConfig`. Required API:

```python
cfg = SimulationConfig.default()
cfg = SimulationConfig.from_yaml("configs/baseline.yaml")
cfg.validate()
cfg.validate_for_simulation()
```

The default configuration uses 30-minute periods, 8 regular agents, 5 specialist agents, actions `(0.0, 0.1, 0.2, 0.3)`, and a calibration parameter path. It contains no hard-coded arrival means.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_config.py -v`

Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore pyproject.toml configs/baseline.yaml src/prescriptive_capacity_sim/__init__.py src/prescriptive_capacity_sim/config.py tests/test_config.py
git commit -m "build: scaffold calibrated call center simulator"
```

### Task 2: Parse, classify, and de-identify logs

**Files:**
- Create: `src/prescriptive_capacity_sim/ingest.py`
- Create: `tests/fixtures/calls_sample.txt`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Create a six-row fixture and failing tests**

The fixture includes `PS`, `IN`, `TT`, an unknown type, a pre-queue hang-up, and a `PHANTOM` row, including the mirror's leading row number.

```python
from prescriptive_capacity_sim.ingest import load_month, load_raw_directory

def test_parser_maps_three_classes(fixture_dir):
    frame, quality = load_month(fixture_dir / "calls_sample.txt")
    assert set(frame["service_class"]) == {
        "regular", "specialist", "callback_special"
    }
    assert quality.unknown_type_rows == 1

def test_analysis_frame_is_deidentified(fixture_dir):
    frame, _ = load_month(fixture_dir / "calls_sample.txt")
    assert {"customer_id", "server", "source_row"}.isdisjoint(frame.columns)

def test_directory_loader_is_deterministic(fixture_dir):
    left = load_raw_directory(fixture_dir)
    right = load_raw_directory(fixture_dir)
    assert left.calls.equals(right.calls)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_ingest.py -v`

Expected: missing `ingest` module.

- [ ] **Step 3: Implement ingestion**

```python
SERVICE_CLASS_MAP = {
    "PS": "regular", "PE": "regular", "NW": "regular",
    "IN": "specialist", "NE": "specialist",
    "TT": "callback_special",
}

@dataclass(frozen=True)
class DataQualityCounts:
    total_rows: int
    phantom_rows: int
    prequeue_exit_rows: int
    unknown_type_rows: int
    invalid_time_rows: int

LOAD_MONTH_SIGNATURE = "load_month(path: Path) -> tuple[pd.DataFrame, DataQualityCounts]"
LOAD_DIRECTORY_SIGNATURE = "load_raw_directory(path: Path) -> IngestedCalls"
```

Handle the mirror row-number field and `day.of.week`, strip type whitespace, build full timestamps with midnight rollover, and exclude unknown/phantom rows from queue calibration while counting them. The returned frame contains no identifiers.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_ingest.py -v`

Expected: all ingestion tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/ingest.py tests/fixtures/calls_sample.txt tests/test_ingest.py
git commit -m "feat: add deidentified call log ingestion"
```

### Task 3: Calibrate arrivals, service, patience, and demand states

**Files:**
- Create: `src/prescriptive_capacity_sim/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

```python
import json
from prescriptive_capacity_sim.calibration import calibrate_calls, write_calibration

def test_split_is_chronological(sample_calls):
    result = calibrate_calls(sample_calls, train_fraction=0.70)
    assert max(result.train_dates) < min(result.validation_dates)
    assert result.parameters["fit_scope"] == "train_only"

def test_aggregation_conserves_training_arrivals(sample_calls):
    result = calibrate_calls(sample_calls, train_fraction=0.70)
    assert result.intervals["arrivals"].sum() == len(result.train_calls)

def test_artifacts_contain_three_class_distributions(sample_calls, tmp_path):
    result = calibrate_calls(sample_calls, train_fraction=0.70)
    paths = write_calibration(result, tmp_path)
    params = json.loads(paths.parameters.read_text(encoding="utf-8"))
    assert set(params["classes"]) == {
        "regular", "specialist", "callback_special"
    }
    assert "service_seconds" in params["empirical"]
    assert "patience_seconds" in params["empirical"]
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_calibration.py -v`

Expected: missing calibration module.

- [ ] **Step 3: Implement calibration**

Implement chronological date splitting, 30-minute aggregation, negative-binomial moment estimates, hierarchical fallback, empirical samples stored as fixed quantile grids, 75th/95th-percentile daily demand states, and a Laplace-smoothed 3-by-3 transition matrix.

Parameter contract:

```json
{
  "schema_version": 1,
  "fit_scope": "train_only",
  "period_minutes": 30,
  "classes": ["regular", "specialist", "callback_special"],
  "arrival": {"regular": {"monday:14": {"mean": 1.0, "dispersion": 2.0}}},
  "empirical": {"service_seconds": {}, "patience_seconds": {}},
  "disruption": {"thresholds": {}, "transition_matrix": []}
}
```

Write `calibration_intervals.csv`, `calibration_parameters.json`, `data_quality_report.json`, and `calibration_manifest.json`. Use temporary sibling files and rename on success so partial artifacts are never mistaken for complete calibration.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_calibration.py -v`

Expected: all calibration tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/calibration.py tests/test_calibration.py
git commit -m "feat: calibrate simulator from call logs"
```

### Task 4: Calibrated demand process and queue state

**Files:**
- Recreate: `src/prescriptive_capacity_sim/state.py`
- Recreate: `src/prescriptive_capacity_sim/demand.py`
- Create: `tests/test_demand.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from prescriptive_capacity_sim.demand import CalibratedDemandProcess

def test_same_seed_reproduces_customer_draws(calibration_fixture):
    process = CalibratedDemandProcess(calibration_fixture)
    a = process.sample(4, "monday", 0, np.random.default_rng(7))
    b = process.sample(4, "monday", 0, np.random.default_rng(7))
    assert a == b

def test_severe_state_increases_expected_arrivals(calibration_fixture):
    process = CalibratedDemandProcess(calibration_fixture)
    assert sum(process.expected_arrivals(4, "monday", 2)) > sum(
        process.expected_arrivals(4, "monday", 0)
    )
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_demand.py -v`

Expected: missing calibrated demand and state API.

- [ ] **Step 3: Implement the state and draws**

Use cohort counts indexed by service class, priority, and waiting-age bucket. `ExogenousDraw` contains arrival cohorts, service-minute requirements, patience buckets, next demand state, pool-availability multipliers, and hidden manager alarm. `CalibratedDemandProcess` loads only the parameter JSON and samples from negative-binomial and empirical quantile distributions.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_demand.py -v`

Expected: all demand tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/state.py src/prescriptive_capacity_sim/demand.py tests/test_demand.py
git commit -m "feat: add calibrated call demand process"
```

### Task 5: Two-pool service, abandonment, and costs

**Files:**
- Recreate: `src/prescriptive_capacity_sim/environment.py`
- Replace: `tests/test_environment.py`

- [ ] **Step 1: Write failing tests**

```python
def test_transition_conserves_each_queue(environment, state, draw):
    result = environment.transition(state, action=1, draw=draw)
    for klass in range(3):
        assert result.next_state.queue_total(klass) == (
            state.queue_total(klass)
            + result.arrivals[klass]
            - result.served[klass]
            - result.abandoned[klass]
        )

def test_specialists_cross_serve_regular_at_reduced_efficiency(
    environment, regular_only_state, no_arrival_draw
):
    result = environment.transition(
        regular_only_state, action=0, draw=no_arrival_draw
    )
    assert result.specialist_minutes_cross_served > 0
    assert result.served[0] > 0

def test_more_capacity_does_not_increase_capacity_abandonment(
    environment, congested_state, fixed_draw
):
    low = environment.transition(congested_state, action=0, draw=fixed_draw)
    high = environment.transition(congested_state, action=3, draw=fixed_draw)
    assert sum(high.abandoned) <= sum(low.abandoned)

def test_priority_precedes_nonpriority_within_class(
    environment, mixed_priority_state, one_call_capacity_draw
):
    result = environment.transition(
        mixed_priority_state, action=0, draw=one_call_capacity_draw
    )
    assert result.served_priority[0] == 1
    assert result.served_nonpriority[0] == 0
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_environment.py -v`

Expected: the old supply-chain transition API fails the call-center contract.

- [ ] **Step 3: Implement queue dynamics**

Allocate regular and specialist pool minutes oldest-first within priority strata. Specialist capacity serves specialist demand first and may cross-serve regular demand using `cross_skill_efficiency`. Callback/special demand uses configured pool shares. Remove cohorts whose patience expires, calculate outcomes, then age the remaining cohorts.

Return a `PeriodResult` with these concrete fields and types:

```python
@dataclass(frozen=True)
class PeriodResult:
    base_staff_cost: float
    augmentation_cost: float
    waiting_cost: float
    abandonment_cost: float
    service_level_cost: float
    total_cost: float
    service_level: float
    p95_wait_minutes: float
    safety_violation: bool
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_environment.py -v`

Expected: conservation, skills, priority, capacity monotonicity, and safety tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/environment.py tests/test_environment.py
git commit -m "feat: simulate call queues and staffing pools"
```

### Task 6: Biased factual logs and oracle counterfactuals

**Files:**
- Recreate: `src/prescriptive_capacity_sim/behavior.py`
- Recreate: `src/prescriptive_capacity_sim/oracle.py`
- Recreate: `src/prescriptive_capacity_sim/logging.py`
- Replace: `tests/test_logging.py`

- [ ] **Step 1: Write failing tests**

```python
def test_observed_log_has_no_potential_outcomes(dataset):
    assert "action" in dataset.observed
    assert not any(c.startswith("potential_") for c in dataset.observed)

def test_oracle_has_four_actions_and_correct_minimizer(dataset):
    costs = [f"potential_cost_a{i}" for i in range(4)]
    assert set(costs).issubset(dataset.oracle)
    expected = dataset.oracle[costs].to_numpy().argmin(axis=1)
    assert (dataset.oracle["oracle_action"].to_numpy() == expected).all()

def test_hidden_alarm_is_not_an_observed_feature(hidden_dataset):
    assert "manager_alarm" not in hidden_dataset.observed.columns
    assert hidden_dataset.metadata["hidden_confounding_strength"] > 0

def test_lower_temperature_reduces_action_overlap(state, base_config):
    low = HistoricalBehaviorPolicy(
        base_config.with_overrides(behavior={"temperature": 0.25})
    ).probabilities(state, manager_alarm=0.0)
    high = HistoricalBehaviorPolicy(
        base_config.with_overrides(behavior={"temperature": 2.0})
    ).probabilities(state, manager_alarm=0.0)
    assert -(low * np.log(low)).sum() < -(high * np.log(high)).sum()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_logging.py -v`

Expected: missing replacement behavior/logging API.

- [ ] **Step 3: Implement behavior and logging**

The historical softmax uses queue totals, specialist/callback share, priority share, time period, demand state, and optional hidden alarm. Sample the exogenous draw once per decision and reuse it across all four oracle branches.

The observed row contains state, chosen action, exact propensity, realized outcomes, and split. It excludes manager alarm, unselected outcomes, identifiers, and oracle action. The oracle row contains potential cost, service level, abandonment, p95 wait, safety violation, and next queue for every action.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_logging.py -v`

Expected: all bias, overlap, reproducibility, oracle, and leakage tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/behavior.py src/prescriptive_capacity_sim/oracle.py src/prescriptive_capacity_sim/logging.py tests/test_logging.py
git commit -m "feat: generate factual and oracle capacity logs"
```

### Task 7: Policies, paired evaluation, and calibration validation

**Files:**
- Recreate: `src/prescriptive_capacity_sim/policies.py`
- Recreate: `src/prescriptive_capacity_sim/evaluation.py`
- Create: `src/prescriptive_capacity_sim/validation.py`
- Replace: `tests/test_evaluation.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing tests**

```python
def test_policy_metrics_cover_decision_and_service_outcomes(
    simulation_config, calibration_fixture
):
    metrics = evaluate_policies(
        simulation_config,
        calibration_fixture,
        default_policies(simulation_config),
        days=2,
        seed=11,
    )
    assert {
        "total_cost", "service_level", "abandonment_rate",
        "mean_wait_minutes", "p95_wait_minutes", "safety_violation_rate",
        "cost_gap_vs_oracle", "cost_improvement_vs_history",
    }.issubset(metrics.columns)

def test_validation_compares_heldout_real_and_simulated(
    heldout_intervals, simulated_intervals, heldout_calls, simulated_calls
):
    report = compare_real_and_simulated(
        heldout_intervals,
        simulated_intervals,
        heldout_calls,
        simulated_calls,
    )
    assert {"metric", "service_class", "real", "simulated", "error"}.issubset(
        report.columns
    )
    assert {"arrival_mae", "service_wasserstein",
            "abandonment_rate_error"}.issubset(set(report["metric"]))
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_evaluation.py tests/test_validation.py -v`

Expected: missing call-center metrics and validation module.

- [ ] **Step 3: Implement policies and diagnostics**

Provide historical, random, four fixed-action, myopic-oracle, and three-period rolling-oracle policies. Evaluate policies using the same episode draws. Compare held-out real intervals with simulations by class using arrival MAE, mean/variance error, empirical service-time Wasserstein distance, abandonment-rate error, and wait-quantile relative error.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_evaluation.py tests/test_validation.py -v`

Expected: all evaluation and validation tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/policies.py src/prescriptive_capacity_sim/evaluation.py src/prescriptive_capacity_sim/validation.py tests/test_evaluation.py tests/test_validation.py
git commit -m "feat: evaluate and validate capacity policies"
```

### Task 8: CLI, scenarios, and documentation

**Files:**
- Recreate: `src/prescriptive_capacity_sim/cli.py`
- Replace: `configs/*.yaml`
- Recreate: `README.md`
- Recreate: `runs.md`
- Replace: `tests/test_cli.py`

- [ ] **Step 1: Write failing end-to-end tests**

```python
def test_calibrate_cli_writes_four_artifacts(fixture_dir, tmp_path):
    output = tmp_path / "calibration"
    assert main(["calibrate", "--raw-dir", str(fixture_dir),
                 "--output", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {
        "calibration_intervals.csv", "calibration_parameters.json",
        "data_quality_report.json", "calibration_manifest.json",
    }

def test_generate_cli_writes_six_artifacts_and_manifest(
    calibrated_config_path, tmp_path
):
    output = tmp_path / "simulation"
    assert main(["generate", "--config", str(calibrated_config_path),
                 "--days", "2", "--seed", "17",
                 "--output", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {
        "observed_log.csv", "oracle_counterfactuals.csv",
        "episode_summary.csv", "policy_metrics.csv",
        "simulation_validation.csv", "run_manifest.json",
    }

def test_cli_outputs_never_contain_identifiers(calibrated_output):
    for csv_path in calibrated_output.glob("*.csv"):
        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        assert {"customer_id", "server"}.isdisjoint(columns)
```

Required commands:

```powershell
uv run python -m prescriptive_capacity_sim.cli calibrate `
  --raw-dir data/raw/technion_anonymous_bank/extracted `
  --output data/processed

uv run python -m prescriptive_capacity_sim.cli generate `
  --config configs/baseline.yaml --days 365 --seed 20260921 `
  --output outputs/baseline
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_cli.py -v`

Expected: the old CLI lacks `calibrate` and the required artifacts.

- [ ] **Step 3: Implement CLI, scenarios, and docs**

`calibrate` emits the four calibration artifacts. `generate` and `benchmark` emit `observed_log.csv`, `oracle_counterfactuals.csv`, `episode_summary.csv`, `policy_metrics.csv`, `simulation_validation.csv`, and `run_manifest.json`.

Reject simulation if the calibration manifest is absent or hashes do not match. Record Git revision when available, raw hashes, calibration hash, config, seed, environment, row counts, and `status: exploratory`.

README must call the environment semisynthetic, distinguish real quantities from synthetic decisions/counterfactuals, and state privacy restrictions. Preserve old exploratory results only under an archived heading in `runs.md`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_cli.py -v`

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/cli.py configs README.md runs.md tests/test_cli.py
git commit -m "feat: expose calibrated simulator workflow"
```

### Task 9: Calibrate real data and final verification

**Files:**
- Generate, do not commit: `data/processed/*`
- Generate, do not commit: `outputs/smoke/*`
- Modify production files only for defects first reproduced by a failing test.

- [ ] **Step 1: Synchronize dependencies**

Run: `uv sync --extra dev`

Expected: exit code 0.

- [ ] **Step 2: Run all tests**

Run: `uv run pytest -q`

Expected: zero failures and errors.

- [ ] **Step 3: Calibrate all twelve monthly files**

```powershell
uv run python -m prescriptive_capacity_sim.cli calibrate `
  --raw-dir data/raw/technion_anonymous_bank/extracted `
  --output data/processed
```

Expected: four calibration artifacts, 12 source files in the manifest, known raw row count, and no identifier columns.

- [ ] **Step 4: Run the smoke simulation**

```powershell
uv run python -m prescriptive_capacity_sim.cli generate `
  --config configs/baseline.yaml --days 30 --seed 20260921 `
  --output outputs/smoke
```

Expected: six artifacts; observed/oracle row counts match; all four actions occur under baseline overlap; zero oracle minimizer mismatches; no potential-outcome columns in the factual log.

- [ ] **Step 5: Prove reproducibility**

Repeat the run into `outputs/smoke_repeat` and compare SHA-256 hashes of CSV files, excluding manifests. Expected: exact matches.

- [ ] **Step 6: Inspect validation**

Verify every metric and service class appears in `simulation_validation.csv`. Record values in `runs.md` as exploratory; do not tune against held-out dates without reclassifying the run.

- [ ] **Step 7: Run final checks**

```powershell
uv run pytest -q
git diff --check
git status --short
```

Expected: tests pass, diff check is clean, raw/generated data are ignored, and only intended source, test, config, and documentation files are tracked.

- [ ] **Step 8: Commit run record**

```powershell
git add runs.md
git commit -m "docs: record calibrated simulator smoke run"
```

## Completion criteria

- Raw monthly files remain unchanged and untracked.
- Calibration artifacts are de-identified and reproducible.
- Demand, service, patience, and disruption parameters use training dates only.
- Staffing actions, historical propensities, and counterfactuals are explicitly synthetic.
- The simulator models three classes, priority, two skill pools, abandonment, waiting, and four augmentation actions.
- Factual and oracle outputs are separated and protected by leakage tests.
- Held-out real-versus-simulated diagnostics are generated automatically.
- Full tests and both end-to-end commands pass.
