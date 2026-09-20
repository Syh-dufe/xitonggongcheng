# Real Call-Center Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, call-level discrete-event simulator calibrated from the Technion Anonymous Bank logs, with dynamic interval staffing and hidden counterfactual action evaluation.

**Architecture:** A privacy-preserving loader converts the Bern mirror's tab-separated monthly files into queue primitives without retaining customer identifiers. An empirical calibrator replays real queue-entry timestamps and samples service and patience primitives. A cloneable event engine advances one interval at a time so every staffing action can be evaluated from the same pre-decision state with common random numbers.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyYAML, pytest, standard-library heap/event simulation.

---

### Task 1: Parse the Anonymous Bank mirror safely

**Files:**
- Create: `src/prescriptive_capacity_sim/callcenter_data.py`
- Create: `tests/test_callcenter_data.py`

- [ ] **Step 1: Write failing parser tests**

Create a temporary mirror-format file whose header has 18 fields while data rows contain a leading row number plus the 18 documented fields. Assert that `load_callcenter_month()` returns typed timestamps, excludes `customer_id` and `server`, removes `PHANTOM`, excludes pre-queue abandonment, and retains queued `AGENT` and `HANG` calls.

- [ ] **Step 2: Run the parser tests and verify failure**

Run: `uv run pytest tests/test_callcenter_data.py -q`

Expected: import failure because `callcenter_data.py` does not exist.

- [ ] **Step 3: Implement the minimal parser**

Implement:

```python
@dataclass(frozen=True)
class QueueCall:
    call_key: str
    queue_entry: datetime
    call_type: str
    priority: int
    observed_wait_seconds: float
    outcome: str
    observed_service_seconds: float | None

def load_callcenter_month(path: Path, included_types: set[str] | None = None) -> list[QueueCall]:
    """Read the Bern mirror format without returning customer or server identifiers."""
```

Accept the mirror's leading row index and verify all documented columns. Use `q_start` for queued calls; for direct-to-agent calls whose `q_start` is `00:00:00`, use `vru_exit` (falling back to `ser_start`) as the queue-entry timestamp and record zero waiting. Reject malformed dates or negative durations with a clear `ValueError`.

- [ ] **Step 4: Run parser tests**

Run: `uv run pytest tests/test_callcenter_data.py -q`

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: parse anonymous bank call logs safely`

### Task 2: Calibrate empirical service and patience primitives

**Files:**
- Create: `src/prescriptive_capacity_sim/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing calibration tests**

Test that service sampling is stratified by call type with a pooled fallback, that identical seeds reproduce samples, and that the Kaplan-Meier patience estimator treats answered calls as right-censored and abandoned calls as observed events.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_calibration.py -q`

Expected: import failure because `calibration.py` does not exist.

- [ ] **Step 3: Implement the empirical models**

Implement:

```python
@dataclass(frozen=True)
class EmpiricalCalibration:
    service_seconds_by_type: dict[str, np.ndarray]
    pooled_service_seconds: np.ndarray
    patience_support_seconds: np.ndarray
    patience_survival: np.ndarray

    def sample_service(self, call_type: str, rng: np.random.Generator) -> float: ...
    def sample_patience(self, rng: np.random.Generator) -> float: ...

def fit_empirical_calibration(calls: Sequence[QueueCall]) -> EmpiricalCalibration: ...
```

Use only positive answered-call service durations. Build a Kaplan-Meier survival curve from queue waits, marking `HANG` as an event and `AGENT` as right-censored. Sampling must be deterministic under a fixed NumPy generator.

- [ ] **Step 4: Run calibration tests**

Run: `uv run pytest tests/test_calibration.py -q`

Expected: all calibration tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: calibrate empirical call primitives`

### Task 3: Build the cloneable interval event engine

**Files:**
- Create: `src/prescriptive_capacity_sim/callcenter_simulator.py`
- Create: `tests/test_callcenter_simulator.py`

- [ ] **Step 1: Write failing event-engine tests**

Cover immediate service, FIFO ordering, high-priority ordering, abandonment at the patience deadline, service completion across interval boundaries, flow conservation, deterministic replay, and weak monotonicity of abandonment under higher staffing for identical call primitives.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_callcenter_simulator.py -q`

Expected: import failure because `callcenter_simulator.py` does not exist.

- [ ] **Step 3: Implement the event engine**

Implement immutable call primitives and cloneable mutable state:

```python
@dataclass(frozen=True)
class SimulatedCall:
    call_key: str
    arrival_second: float
    service_seconds: float
    patience_seconds: float
    priority: int
    call_type: str

@dataclass
class CallCenterState:
    now_second: float
    waiting: list[WaitingCall]
    busy: list[BusyCall]

    def clone(self) -> "CallCenterState": ...

def simulate_interval(
    state: CallCenterState,
    arrivals: Sequence[SimulatedCall],
    staffing: int,
    interval_seconds: int = 1800,
    service_level_seconds: int = 20,
) -> IntervalResult: ...
```

Process arrivals, service completions, abandonments, and interval end in chronological order. Calls already in service finish after staffing reductions; new service begins only when busy calls are below the requested staffing. Return arrivals, started service, completed service, abandoned, waiting-time summaries, service level, ending queue, busy seconds, staffed seconds, and a cloned next state.

- [ ] **Step 4: Run event-engine tests**

Run: `uv run pytest tests/test_callcenter_simulator.py -q`

Expected: all event-engine tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add cloneable call center event simulator`

### Task 4: Replay real days and generate factual/oracle logs

**Files:**
- Create: `src/prescriptive_capacity_sim/callcenter_scenarios.py`
- Create: `tests/test_callcenter_scenarios.py`

- [ ] **Step 1: Write failing scenario tests**

Test that a real day is converted into 30-minute call primitives, the same seed produces identical service and patience draws, the behavior policy exposes positive propensity for every staffing action, and the oracle branches all actions from identical pre-decision state and exogenous calls.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_callcenter_scenarios.py -q`

Expected: import failure because `callcenter_scenarios.py` does not exist.

- [ ] **Step 3: Implement replay and branching**

Implement `build_replay_day()`, a softmax `StaffingBehaviorPolicy`, and `generate_semisynthetic_day()`. The observed row contains only the selected action and realized result. The oracle row contains potential cost and service outcomes for every action but is returned separately. Cost combines staffing minutes, waiting seconds, abandonment, SLA violations, and terminal queue using an explicit configuration object.

- [ ] **Step 4: Run scenario tests**

Run: `uv run pytest tests/test_callcenter_scenarios.py -q`

Expected: all scenario tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: generate factual and counterfactual staffing logs`

### Task 5: Add a real-data command line workflow

**Files:**
- Modify: `src/prescriptive_capacity_sim/cli.py`
- Create: `configs/callcenter_baseline.yaml`
- Create: `tests/test_callcenter_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write a failing CLI test**

Run the CLI against a temporary two-day fixture and assert creation of `calibration_summary.json`, `observed_log.csv`, `oracle_counterfactuals.csv`, `daily_metrics.csv`, and `run_manifest.json`. Assert the observed log has no customer/server identifiers and no potential-outcome columns.

- [ ] **Step 2: Verify the CLI test fails**

Run: `uv run pytest tests/test_callcenter_cli.py -q`

Expected: the `callcenter-generate` subcommand is missing.

- [ ] **Step 3: Implement the command**

Add:

```text
capacity-sim callcenter-generate --data-dir PATH --config configs/callcenter_baseline.yaml --output PATH --seed 20260920
```

The command validates twelve monthly files when running the full dataset, records data paths and hashes in the manifest, splits days chronologically into calibration/validation/test partitions, and never writes raw identifiers.

- [ ] **Step 4: Run CLI and full tests**

Run: `uv run pytest -q`

Expected: all legacy and new tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add real call center simulation workflow`

### Task 6: Validate against the downloaded year and document the simulator

**Files:**
- Modify: `README.md`
- Create: `docs/callcenter-data-and-simulator.md`
- Modify: `runs.md`

- [ ] **Step 1: Run the full-year data audit**

Run the loader against `C:\Users\13384\Desktop\系统\data\raw\technion_anonymous_bank\extracted` and record total raw rows, included queue calls, exclusions by reason, dates, call types, service duration summaries, and abandonment summaries.

- [ ] **Step 2: Run a deterministic smoke simulation**

Generate at least seven replay days under the baseline staffing action set. Re-run with the same seed and verify byte-identical factual and oracle CSV outputs.

- [ ] **Step 3: Document provenance and limitations**

Explain that real logs calibrate arrivals, service, and patience; staffing actions and counterfactuals are semi-synthetic; busy-agent observations are not treated as full historical staffing; and raw identifiers must not be committed.

- [ ] **Step 4: Run all verification**

Run: `uv run pytest -q`

Run: `git status --short`

Expected: tests pass; no raw data or generated outputs are tracked.

- [ ] **Step 5: Commit**

Commit message: `docs: document calibrated call center simulator`

### Task 7: Publish to the user's new repository

**Files:** none

- [ ] **Step 1: Add the new GitHub repository as a dedicated remote**

Use remote name `xitonggongcheng` and URL `https://github.com/Syh-dufe/xitonggongcheng.git` without changing the existing origin.

- [ ] **Step 2: Verify the destination and branch**

Run `git remote -v`, `git status --short --branch`, and the complete test suite.

- [ ] **Step 3: Push the implementation branch**

Push `codex/real-callcenter-simulator` to the new repository. Do not force-push and do not overwrite an unrelated remote branch.
