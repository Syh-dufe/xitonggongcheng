# Capacity Calibration Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a reproducible, factual-only capacity-scenario validation workflow that selects a documented baseline using held-out call-center diagnostics.

**Architecture:** A new capacity_validation.py module owns candidate-grid parsing, factual-only simulation, metric aggregation, and report writing. It uses the existing demand process, historical behavior policy, and queueing kernel but never calls evaluate_actions, generate_dataset, or Oracle policy evaluation. The CLI validates a YAML grid against processed calibration artifacts.

**Implementation correction:** `supervisor_emergency_agents` remains a future-facing general configuration field, but the current queueing kernel does not consume it. Candidate calibration therefore varies and reports only regular-agent count, specialist-agent count, and cross-skill efficiency; candidate YAML that includes the unused field is rejected.

**Audit correction:** Calibration writes a deidentified, exact held-out call table (`service_class`, `queue_seconds`, `service_seconds`) rather than reconstructing calls from per-class quantiles in the quality report. Both CLI validation paths read that table, report queue-flow conservation as a non-selection diagnostic, and distinguish exact, truncated, and cycled validation-weekday schedules in their manifests.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, pandas, PyYAML, pytest.

---

## File structure

- Create: src/prescriptive_capacity_sim/capacity_validation.py — candidate schema, factual-only runner, ranking and reports.
- Modify: src/prescriptive_capacity_sim/validation.py — add overall p90 waiting diagnostic.
- Modify: src/prescriptive_capacity_sim/cli.py — add validate-capacity command.
- Create: configs/capacity_candidates.yaml — baseline and sensitivity grid.
- Modify: tests/test_validation.py, tests/test_cli.py, README.md.
- Create: tests/test_capacity_validation.py.

### Task 1: Candidate schema and factual-only runner

**Files:**
- Create: src/prescriptive_capacity_sim/capacity_validation.py
- Create: tests/test_capacity_validation.py

- [ ] **Step 1: Write the failing candidate-grid test**

~~~
def test_candidate_grid_rejects_duplicate_names(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "candidates:\n"
        "  - name: baseline\n    regular_agents: 8\n    specialist_agents: 5\n"
        "  - name: baseline\n    regular_agents: 9\n    specialist_agents: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_candidates(path)
~~~

- [ ] **Step 2: Verify RED**

Run: uv run pytest tests/test_capacity_validation.py::test_candidate_grid_rejects_duplicate_names -q

Expected: FAIL because capacity_validation does not exist.

- [ ] **Step 3: Implement minimal candidate schema**

~~~
@dataclass(frozen=True)
class ResourceCandidate:
    name: str
    regular_agents: int
    specialist_agents: int
    cross_skill_efficiency: float = 0.75

    def resource_overrides(self) -> dict[str, int | float]:
        return {
            "regular_agents": self.regular_agents,
            "specialist_agents": self.specialist_agents,
            "cross_skill_efficiency": self.cross_skill_efficiency,
        }


def load_candidates(path: str | Path) -> tuple[ResourceCandidate, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = raw.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a nonempty list")
    candidates = tuple(ResourceCandidate(**row) for row in rows)
    if len({item.name for item in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        SimulationConfig.default().with_overrides(
            resources=candidate.resource_overrides()
        )
    return candidates
~~~

- [ ] **Step 4: Verify GREEN**

Run: uv run pytest tests/test_capacity_validation.py::test_candidate_grid_rejects_duplicate_names -q

Expected: PASS.

- [ ] **Step 5: Write the failing factual-only runner test**

~~~
def test_factual_runner_is_reproducible_and_excludes_oracle_columns():
    cfg = SimulationConfig.default().with_overrides(periods_per_day=2)
    parameters = _parameters(periods_per_day=2)
    left = simulate_historical_periods(cfg, parameters, days=2, seed=23)
    right = simulate_historical_periods(cfg, parameters, days=2, seed=23)
    assert left.equals(right)
    assert len(left) == 4
    assert not any(
        "potential_" in column or "oracle" in column for column in left.columns
    )
~~~

- [ ] **Step 6: Verify RED**

Run: uv run pytest tests/test_capacity_validation.py::test_factual_runner_is_reproducible_and_excludes_oracle_columns -q

Expected: FAIL because simulate_historical_periods does not exist.

- [ ] **Step 7: Implement factual-only runner**

Create simulate_historical_periods(config, parameters, *, days, seed). Reject days less than or equal to zero. For every episode and period initialize CalibratedDemandProcess, CallCenterEnvironment, and HistoricalBehaviorPolicy; use SeedSequence([seed, episode_id, 0]) for environmental draws and SeedSequence([seed, episode_id, 1]) for historical actions; invoke only demand.sample, behavior.act, and environment.transition. Return period-level factual arrivals, abandonment counts, mean class waits, p95 wait, service level, chosen action, and factual state fields. Do not import oracle.py or logging.generate_dataset.

- [ ] **Step 8: Verify GREEN**

Run: uv run pytest tests/test_capacity_validation.py -q

Expected: PASS.

- [ ] **Step 9: Commit**

Run:
git add src/prescriptive_capacity_sim/capacity_validation.py tests/test_capacity_validation.py
git commit -m "feat: add factual capacity validation runner"

### Task 2: Held-out metrics and predeclared selection score

**Files:**
- Modify: src/prescriptive_capacity_sim/validation.py
- Modify: src/prescriptive_capacity_sim/capacity_validation.py
- Modify: tests/test_validation.py
- Modify: tests/test_capacity_validation.py

- [ ] **Step 1: Write a failing p90 diagnostic test**

~~~
def test_validation_reports_overall_p90_wait_error():
    report = compare_real_and_simulated(
        _real_intervals_with_p90(), _simulated_periods(), _real_calls(), _parameters()
    )
    row = report.loc[
        report["metric"].eq("overall_p90_wait_relative_error")
    ].iloc[0]
    assert row["real"] == 2.0
    assert row["simulated"] == 3.0
    assert row["error"] == 0.5
~~~

- [ ] **Step 2: Verify RED**

Run: uv run pytest tests/test_validation.py::test_validation_reports_overall_p90_wait_error -q

Expected: FAIL because the metric is absent.

- [ ] **Step 3: Add overall p90 diagnostic**

Append one service_class=overall row in compare_real_and_simulated. Compute real p90 as the 90th percentile of positive real_calls.queue_seconds divided by 60. Compute simulated p90 as the 90th percentile of factual simulated_log.p95_wait_minutes. Use absolute relative error; use zero for empty inputs. Keep all existing class-level diagnostics unchanged.

- [ ] **Step 4: Verify GREEN**

Run: uv run pytest tests/test_validation.py -q

Expected: PASS.

- [ ] **Step 5: Write a failing scoring test**

~~~
def test_summary_ranks_lowest_predeclared_operational_error_first():
    runs = pd.DataFrame([
        {"candidate": "high", "seed": 1, "metric": "abandonment_rate_error", "error": 0.4},
        {"candidate": "high", "seed": 1, "metric": "mean_wait_relative_error", "error": 0.4},
        {"candidate": "high", "seed": 1, "metric": "overall_p90_wait_relative_error", "error": 0.4},
        {"candidate": "low", "seed": 1, "metric": "abandonment_rate_error", "error": 0.1},
        {"candidate": "low", "seed": 1, "metric": "mean_wait_relative_error", "error": 0.1},
        {"candidate": "low", "seed": 1, "metric": "overall_p90_wait_relative_error", "error": 0.1},
    ])
    summary = summarize_candidates(runs)
    assert summary.iloc[0]["candidate"] == "low"
    assert summary.iloc[0]["selection_score"] == pytest.approx(0.1)
~~~

- [ ] **Step 6: Verify RED**

Run: uv run pytest tests/test_capacity_validation.py::test_summary_ranks_lowest_predeclared_operational_error_first -q

Expected: FAIL because summarize_candidates does not exist.

- [ ] **Step 7: Implement score and summary**

Define:
~~~
SELECTION_WEIGHTS = {
    "abandonment_rate_error": 0.4,
    "mean_wait_relative_error": 0.4,
    "overall_p90_wait_relative_error": 0.2,
}
~~~

Average class-level values per candidate, seed, metric before weighting so classes do not receive extra weight. Score only the three named operational metrics; arrival and service diagnostics remain in output but are excluded because changing capacity cannot change their calibrated distribution. Return candidate, selection score, score standard deviation across seeds, seed count, rank, and fixed resource values sorted by score then candidate name.

- [ ] **Step 8: Verify GREEN**

Run: uv run pytest tests/test_validation.py tests/test_capacity_validation.py -q

Expected: PASS.

- [ ] **Step 9: Commit**

Run:
git add src/prescriptive_capacity_sim/capacity_validation.py src/prescriptive_capacity_sim/validation.py tests/test_capacity_validation.py tests/test_validation.py
git commit -m "feat: score held-out capacity scenarios"

### Task 3: CLI reports, manifest, candidate grid, documentation

**Files:**
- Modify: src/prescriptive_capacity_sim/cli.py
- Create: configs/capacity_candidates.yaml
- Modify: tests/test_cli.py
- Modify: README.md

- [ ] **Step 1: Write the failing CLI integration test**

~~~
def test_validate_capacity_writes_factual_reports(tmp_path):
    calibration = tmp_path / "calibration"
    assert main(["calibrate", "--raw-dir", str(FIXTURE_DIR), "--output", str(calibration)]) == 0
    config = tmp_path / "config.yaml"
    config.write_text(
        f"calibration_path: {calibration / 'calibration_parameters.json'}\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        "candidates:\n"
        "  - name: baseline\n    regular_agents: 8\n    specialist_agents: 5\n"
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
    assert "oracle" not in (output / "candidate_validation_runs.csv").read_text(encoding="utf-8")
~~~

- [ ] **Step 2: Verify RED**

Run: uv run pytest tests/test_cli.py::test_validate_capacity_writes_factual_reports -q

Expected: FAIL because validate-capacity is not a recognised command.

- [ ] **Step 3: Implement the command**

Add parser arguments --config Path, --candidates Path, --days int, --seeds int nargs="+", and --output Path. Dispatch to _validate_capacity. It must reuse the calibration manifest hash check, invoke the factual-only runner for every candidate/seed pair, call compare_real_and_simulated with validation intervals, and add candidate value/seed columns to every metric row.

Write:
- candidate_validation_runs.csv: candidate, seed, metric, service_class, real, simulated, error plus three implemented resource values;
- candidate_validation_summary.csv: ranked summary;
- candidate_validation_manifest.json: command, config, calibration hash, candidate YAML SHA-256, candidate values, days, seeds, selection weights, package/Python/platform/Git metadata, and counterfactual_data_used false.

- [ ] **Step 4: Add documented candidate grid**

Create configs/capacity_candidates.yaml:
~~~
# Scenario values, not recovered historical staffing.
candidates:
  - name: baseline_8_regular_5_specialist
    regular_agents: 8
    specialist_agents: 5
    cross_skill_efficiency: 0.75
  - name: lean_7_regular_4_specialist
    regular_agents: 7
    specialist_agents: 4
    cross_skill_efficiency: 0.65
  - name: flexible_8_regular_5_specialist
    regular_agents: 8
    specialist_agents: 5
    cross_skill_efficiency: 0.90
  - name: resilient_9_regular_6_specialist
    regular_agents: 9
    specialist_agents: 6
    cross_skill_efficiency: 0.75
~~~

- [ ] **Step 5: Document exact workflow**

Add this README command:
~~~
uv run python -m prescriptive_capacity_sim.cli validate-capacity --config configs/baseline.yaml --candidates configs/capacity_candidates.yaml --days 90 --seeds 20260921 20260922 20260923 --output outputs/capacity_validation
~~~

State that no Oracle data are read; the score is 40% abandonment-rate error, 40% mean-wait relative error, and 20% overall p90-wait error. State that the result is a semisynthetic baseline scenario, not recovered enterprise staffing.

- [ ] **Step 6: Verify GREEN**

Run: uv run pytest tests/test_cli.py::test_validate_capacity_writes_factual_reports -q

Expected: PASS.

- [ ] **Step 7: Commit**

Run:
git add src/prescriptive_capacity_sim/cli.py configs/capacity_candidates.yaml tests/test_cli.py README.md
git commit -m "feat: add capacity validation command"

### Task 4: Complete verification and smoke record

**Files:**
- Modify: runs.md

- [ ] **Step 1: Run all tests**

Run: uv run pytest -q

Expected: every test passes without warnings.

- [ ] **Step 2: Run factual-only smoke validation**

Run:
~~~
uv run python -m prescriptive_capacity_sim.cli calibrate --raw-dir data/raw/technion_anonymous_bank/extracted --output data/processed
uv run python -m prescriptive_capacity_sim.cli validate-capacity --config configs/baseline.yaml --candidates configs/capacity_candidates.yaml --days 10 --seeds 20260921 20260922 --output outputs/capacity_validation_smoke
~~~

Expected: three report files exist; the manifest says counterfactual_data_used false; rank starts at 1 and is consecutive.

- [ ] **Step 3: Inspect privacy and counterfactual boundaries**

Run:
rg -n "customer_id|server" outputs/capacity_validation_smoke data/processed
rg -n "oracle|potential_" outputs/capacity_validation_smoke/candidate_validation_runs.csv

Expected: no matches.

- [ ] **Step 4: Record only non-sensitive results**

Append command metadata, test result, selected baseline name, and limitations to runs.md. Do not commit raw data, processed data, or output reports because they are ignored.

- [ ] **Step 5: Commit**

Run:
git add runs.md
git commit -m "docs: record capacity validation smoke run"
