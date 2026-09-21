# A-Herzog Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin and audit A-Herzog Callcenter-Simulator 6.2.235, define the 30-minute state/action protocol, export Technion-calibrated model inputs, and prove the Python controller against a deterministic reference backend before patching the Java event loop.

**Architecture:** The repository keeps upstream source outside tracked application code and records immutable provenance in a manifest. Python owns calibration, protocol validation, process control, manifests, and a reference backend implementing the same contract expected from Java. The source audit produced by this plan supplies exact A-Herzog class and method locations for the follow-on Java-hook plan; no unverified Java internals are guessed in this phase.

**Tech Stack:** Python 3.11+, dataclasses, JSON Lines, subprocess, NumPy, pandas, SciPy, pytest, PowerShell, Git, A-Herzog Callcenter-Simulator 6.2.235 / Java 11+.

---

## Scope boundary

This plan is the first independently testable subsystem from the approved design. It does not train the paper's AI models and does not yet alter A-Herzog's event loop. It finishes when the project can reproducibly acquire the fixed upstream version, export calibrated inputs, run a 48-period state/action episode through the production Python controller, and identify the exact upstream hook points. The next plan will implement and verify the Java bridge using those audited locations.

## File map

- `third_party/a_herzog/UPSTREAM.json`: immutable upstream identity and license metadata.
- `third_party/a_herzog/README.md`: acquisition, license, build, and citation instructions.
- `scripts/fetch_a_herzog.ps1`: fetch and verify the pinned upstream tag without committing downloaded source.
- `src/prescriptive_capacity_sim/protocol.py`: versioned state, action, and result messages.
- `src/prescriptive_capacity_sim/bridge.py`: backend protocol plus JSON Lines subprocess controller.
- `src/prescriptive_capacity_sim/reference_backend.py`: deterministic in-process backend used for contract tests and migration comparison.
- `src/prescriptive_capacity_sim/a_herzog_export.py`: neutral calibrated model specification for the Java adapter.
- `src/prescriptive_capacity_sim/patience.py`: right-censored patience estimator.
- `src/prescriptive_capacity_sim/upstream_audit.py`: source-tree audit that reports candidate entry points and random-seed/event hooks.
- `tests/test_protocol.py`, `tests/test_bridge.py`, `tests/test_a_herzog_export.py`, `tests/test_patience.py`, `tests/test_upstream_audit.py`: focused tests.
- `docs/a-herzog-integration.md`: reproducible setup and current integration status.

### Task 1: Record and verify upstream provenance

**Files:**
- Create: `third_party/a_herzog/UPSTREAM.json`
- Create: `third_party/a_herzog/README.md`
- Create: `scripts/fetch_a_herzog.ps1`
- Modify: `.gitignore`

- [ ] **Step 1: Add the immutable upstream manifest**

Create `third_party/a_herzog/UPSTREAM.json` with:

```json
{
  "name": "A-Herzog/Callcenter-Simulator",
  "repository": "https://github.com/A-Herzog/Callcenter-Simulator.git",
  "version": "6.2.235",
  "tag": "6.2.235",
  "commit_prefix": "2119d92",
  "license": "Apache-2.0",
  "project_doi": "10.5281/zenodo.12792568",
  "source_directory": ".cache/upstream/Callcenter-Simulator-6.2.235"
}
```

- [ ] **Step 2: Add a fetch script that fails closed**

Create `scripts/fetch_a_herzog.ps1` with parameters `ManifestPath` and `Destination`. The script must initialize an empty repository, fetch only `refs/tags/6.2.235`, detach-checkout `FETCH_HEAD`, and reject any revision whose short hash does not start with `2119d92`:

```powershell
param(
  [string]$ManifestPath = "third_party/a_herzog/UPSTREAM.json",
  [string]$Destination = ".cache/upstream/Callcenter-Simulator-6.2.235"
)
$ErrorActionPreference = "Stop"
$manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$target = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $target) {
  $actual = git -C $target rev-parse --short HEAD
  if (-not $actual.StartsWith($manifest.commit_prefix)) {
    throw "Existing upstream checkout has revision $actual"
  }
  exit 0
}
New-Item -ItemType Directory -Path $target | Out-Null
git init $target
git -C $target remote add origin $manifest.repository
git -C $target -c http.version=HTTP/1.1 fetch --depth 1 origin "refs/tags/$($manifest.tag)"
git -C $target checkout --detach FETCH_HEAD
$actual = git -C $target rev-parse --short HEAD
if (-not $actual.StartsWith($manifest.commit_prefix)) {
  throw "Fetched revision $actual does not match $($manifest.commit_prefix)"
}
```

- [ ] **Step 3: Keep fetched source out of Git**

Add this exact entry to `.gitignore`:

```gitignore
.cache/upstream/
```

- [ ] **Step 4: Document attribution and build prerequisites**

In `third_party/a_herzog/README.md`, record the upstream repository, tag, commit prefix, Apache 2.0 obligations, DOI, `powershell -File scripts/fetch_a_herzog.ps1` command, and Java 11+/Maven requirements. State that no upstream source is redistributed by this repository at this phase.

- [ ] **Step 5: Verify acquisition**

Run:

```powershell
powershell -File scripts/fetch_a_herzog.ps1
git -C .cache/upstream/Callcenter-Simulator-6.2.235 rev-parse --short HEAD
```

Expected: the second command begins with `2119d92`; a network failure must leave the task incomplete rather than silently selecting another version.

- [ ] **Step 6: Commit provenance files**

```powershell
git add .gitignore third_party/a_herzog scripts/fetch_a_herzog.ps1
git commit -m "build: pin A-Herzog simulator upstream"
```

### Task 2: Define the versioned state/action protocol

**Files:**
- Create: `tests/test_protocol.py`
- Create: `src/prescriptive_capacity_sim/protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Create tests covering round-trip serialization, four legal actions, period mismatch rejection, unknown schema rejection, and non-finite numeric rejection:

```python
import pytest

from prescriptive_capacity_sim.protocol import ActionMessage, StateMessage


def sample_state() -> StateMessage:
    return StateMessage(
        schema_version="1.0", episode_id=7, period=3, seed=41,
        weekday="monday", demand_forecast=(12.0, 4.0, 1.0),
        arrivals=(11, 5, 1), queues=(3, 2, 0),
        mean_wait_minutes=(1.5, 2.0, 0.0), max_wait_minutes=(4.0, 5.0, 0.0),
        staffed_agents=(8, 5, 0), idle_agents=(1, 1, 0),
        utilization=(0.875, 0.8, 0.0), previous_action=1,
        cumulative_reinforcement=2, served=(8, 3, 1), abandoned=(0, 0, 0),
    )


def test_state_round_trip():
    state = sample_state()
    assert StateMessage.from_json(state.to_json()) == state


@pytest.mark.parametrize("action", range(4))
def test_action_accepts_four_levels(action):
    message = ActionMessage("1.0", 7, 3, action)
    message.validate_for(sample_state())


def test_action_rejects_wrong_period():
    with pytest.raises(ValueError, match="period"):
        ActionMessage("1.0", 7, 4, 0).validate_for(sample_state())
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `uv run pytest tests/test_protocol.py -q`

Expected: collection fails because `prescriptive_capacity_sim.protocol` does not exist.

- [ ] **Step 3: Implement immutable protocol messages**

Implement frozen `StateMessage`, `ActionMessage`, and `ResultMessage` dataclasses. `ResultMessage` contains `schema_version`, `episode_id`, `periods_completed`, `total_served`, `total_abandoned`, `final_queues`, and `exit_status`. Give every message `to_json` and `from_json`; add structural length checks for all three-element tuples, non-negative count validation, finite numeric validation, and `ActionMessage.validate_for(state)`. Use sorted JSON keys and compact separators so protocol transcripts are deterministic.

- [ ] **Step 4: Run protocol tests**

Run: `uv run pytest tests/test_protocol.py -q`

Expected: all protocol tests pass.

- [ ] **Step 5: Commit the protocol**

```powershell
git add tests/test_protocol.py src/prescriptive_capacity_sim/protocol.py
git commit -m "feat: define simulator bridge protocol"
```

### Task 3: Add a backend contract and process controller

**Files:**
- Create: `tests/test_bridge.py`
- Create: `tests/fixtures/fake_bridge.py`
- Create: `src/prescriptive_capacity_sim/bridge.py`

- [ ] **Step 1: Write failing lifecycle tests**

Test a successful two-period exchange, malformed JSON, wrong period, timeout, non-zero exit, and diagnostics on stderr. The happy-path assertion must use a policy callable:

```python
from prescriptive_capacity_sim.bridge import JsonLineBridge


def test_bridge_exchanges_state_and_action(fake_bridge_command):
    bridge = JsonLineBridge(fake_bridge_command, timeout_seconds=2.0)
    transcript = bridge.run(lambda state: 2)
    assert [item.action for item in transcript.actions] == [2, 2]
    assert transcript.result.periods_completed == 2
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `uv run pytest tests/test_bridge.py -q`

Expected: collection fails because `prescriptive_capacity_sim.bridge` does not exist.

- [ ] **Step 3: Implement the bridge contract**

Define the backend contract plus immutable `PeriodExchange` and `EpisodeTranscript` containers. `EpisodeTranscript` contains ordered `states`, ordered `actions`, and one final `ResultMessage`:

```python
class Backend(Protocol):
    def run(self, policy: Callable[[StateMessage], int]) -> EpisodeTranscript: ...
```

Implement `JsonLineBridge` with `subprocess.Popen(..., text=True, bufsize=1)`, one reader thread per output stream, a bounded queue, deadline-based reads, strict state/action validation, graceful stdin close, exit-code checking, and captured stderr. Never use `shell=True`.

- [ ] **Step 4: Implement the deterministic fake bridge**

`tests/fixtures/fake_bridge.py` must emit two valid `StateMessage` JSON lines, read one action after each state, then emit a final result message. Command-line flags select malformed, wrong-period, timeout, and non-zero-exit behavior.

- [ ] **Step 5: Run lifecycle tests**

Run: `uv run pytest tests/test_bridge.py -q`

Expected: all bridge tests pass and no child process remains running.

- [ ] **Step 6: Commit bridge infrastructure**

```powershell
git add tests/test_bridge.py tests/fixtures/fake_bridge.py src/prescriptive_capacity_sim/bridge.py
git commit -m "feat: add headless simulator process bridge"
```

### Task 4: Estimate patience with censoring

**Files:**
- Create: `tests/test_patience.py`
- Create: `src/prescriptive_capacity_sim/patience.py`
- Modify: `src/prescriptive_capacity_sim/calibration.py`

- [ ] **Step 1: Write failing Kaplan-Meier tests**

Use a small hand-calculated sample in which abandoned calls are events and served calls are right-censored:

```python
import numpy as np
from prescriptive_capacity_sim.patience import kaplan_meier_quantiles


def test_patience_quantiles_treat_service_as_censoring():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    abandoned = np.array([True, False, True, False])
    result = kaplan_meier_quantiles(times, abandoned, probabilities=(0.25, 0.5))
    assert result[0.25] == 1.0
    assert result[0.5] == 3.0
```

Add tests for ties, zero times, no abandonment events, and all-abandonment samples.

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_patience.py -q`

Expected: collection fails because `prescriptive_capacity_sim.patience` does not exist.

- [ ] **Step 3: Implement the estimator**

Sort unique positive times, compute risk-set sizes immediately before each time, apply event factors `1 - d_i / n_i`, and invert the estimated abandonment CDF `1 - S(t)` for requested probabilities. Return an explicit `PatienceEstimate` containing quantiles, event count, censored count, and an `identifiable` flag. When no event occurs, mark the estimate non-identifiable and let calibration use its documented class-level fallback.

- [ ] **Step 4: Connect calibration output**

Replace the existing queue-time-as-patience export with per-class censored estimates. Preserve raw waiting-time summaries separately for validation. Add `patience_estimation` metadata to `calibration_parameters.json`.

- [ ] **Step 5: Run focused and regression tests**

Run:

```powershell
uv run pytest tests/test_patience.py tests/test_calibration.py tests/test_ingest.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit patience calibration**

```powershell
git add tests/test_patience.py src/prescriptive_capacity_sim/patience.py src/prescriptive_capacity_sim/calibration.py
git commit -m "feat: calibrate censored customer patience"
```

### Task 5: Export an A-Herzog-neutral calibrated model specification

**Files:**
- Create: `tests/test_a_herzog_export.py`
- Create: `src/prescriptive_capacity_sim/a_herzog_export.py`
- Modify: `src/prescriptive_capacity_sim/cli.py`

- [ ] **Step 1: Write failing exporter tests**

The exporter test must assert three customer types, three agent groups, 48 periods, the documented skill matrix, disabled unsupported behaviors, and provenance:

```python
def test_export_contains_research_model(calibration_parameters):
    spec = build_model_spec(calibration_parameters, SimulationConfig.default())
    assert [item["id"] for item in spec["customer_types"]] == [
        "regular", "specialist", "callback_special"
    ]
    assert len(spec["periods"]) == 48
    assert spec["behaviors"] == {"redial": False, "forwarding": False, "recall": False}
    assert spec["upstream"]["version"] == "6.2.235"
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_a_herzog_export.py -q`

Expected: collection fails because `a_herzog_export` does not exist.

- [ ] **Step 3: Implement deterministic export**

Implement `build_model_spec(parameters, config)` and `write_model_spec(...)`. The JSON must contain customer types, interval arrival parameters, service distributions, censored patience distributions, agent groups, compatibility and service-time multipliers, action levels, decision interval, disabled unsupported behaviors, Technion calibration hash, and A-Herzog upstream identity. Write atomically using a temporary sibling file and `Path.replace`.

- [ ] **Step 4: Add an export command**

Extend the CLI with:

```text
capacity-sim export-a-herzog --config configs/baseline.yaml --output outputs/a_herzog/model_spec.json
```

The command must call `validate_for_simulation`, reject missing calibration, and print only the written path.

- [ ] **Step 5: Run exporter and CLI tests**

Run:

```powershell
uv run pytest tests/test_a_herzog_export.py tests/test_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the exporter**

```powershell
git add tests/test_a_herzog_export.py src/prescriptive_capacity_sim/a_herzog_export.py src/prescriptive_capacity_sim/cli.py
git commit -m "feat: export calibrated A-Herzog model spec"
```

### Task 6: Adapt the current simulator as a protocol-conformance backend

**Files:**
- Create: `tests/test_reference_backend.py`
- Create: `src/prescriptive_capacity_sim/reference_backend.py`

- [ ] **Step 1: Write failing conformance tests**

Test exactly 48 decisions, deterministic replay, action application, and queue conservation:

```python
def test_reference_backend_runs_48_decision_periods(config, parameters):
    backend = ReferenceBackend(config, parameters, episode_id=1, seed=23)
    transcript = backend.run(lambda state: state.period % 4)
    assert transcript.result.periods_completed == 48
    assert [a.action for a in transcript.actions[:4]] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_reference_backend.py -q`

Expected: collection fails because `reference_backend` does not exist.

- [ ] **Step 3: Implement the adapter**

Wrap `CalibratedDemandProcess` and `CallCenterEnvironment` behind the `Backend` protocol. Convert each pre-decision state to `StateMessage`, validate the returned action through `ActionMessage`, call the existing transition, and append protocol messages plus period outcomes to `EpisodeTranscript`. Do not duplicate queueing logic.

- [ ] **Step 4: Run conformance and existing environment tests**

Run:

```powershell
uv run pytest tests/test_reference_backend.py tests/test_environment.py tests/test_demand.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the reference backend**

```powershell
git add tests/test_reference_backend.py src/prescriptive_capacity_sim/reference_backend.py
git commit -m "feat: expose reference backend contract"
```

### Task 7: Audit the pinned A-Herzog source tree

**Files:**
- Create: `tests/test_upstream_audit.py`
- Create: `src/prescriptive_capacity_sim/upstream_audit.py`
- Create: `docs/a-herzog-source-audit.md`

- [ ] **Step 1: Write failing audit tests with a fixture tree**

Build a temporary Java tree containing a `main` method, event loop, agent count mutation, random generator construction, and statistics exporter. Assert that the audit classifies each file and reports missing categories as errors.

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_upstream_audit.py -q`

Expected: collection fails because `upstream_audit` does not exist.

- [ ] **Step 3: Implement structural source audit**

Implement a deterministic recursive scan of `.java` and Maven files. Record files and line numbers matching these categories:

```python
PATTERNS = {
    "entry_points": ("public static void main",),
    "event_execution": ("runNext", "nextEvent", "EventList", "schedule"),
    "agent_capacity": ("setNumber", "setCapacity", "numAgents", "freeAgents"),
    "randomness": ("Random", "RandomGenerator", "seed", "Seed"),
    "statistics": ("Statistics", "export", "saveStatistics"),
    "command_line": ("args[]", "args.length", "CommandLine"),
}
```

Emit JSON for machine use and Markdown grouped by category for review. Include upstream revision from `git rev-parse` and fail if it does not begin with `2119d92`.

- [ ] **Step 4: Run the audit on the pinned checkout**

Run:

```powershell
uv run python -m prescriptive_capacity_sim.upstream_audit `
  --source .cache/upstream/Callcenter-Simulator-6.2.235 `
  --json outputs/a_herzog/source_audit.json `
  --markdown docs/a-herzog-source-audit.md
```

Expected: the command succeeds only when every required category has at least one candidate; the Markdown report contains exact repository-relative paths and one-based line numbers.

- [ ] **Step 5: Review and narrow candidates**

Read each reported location and edit `docs/a-herzog-source-audit.md` to mark one primary hook and one fallback for entry point, decision-boundary event, capacity mutation, seed injection, and statistics extraction. Every selection must quote a method or class name and explain why it preserves the original queueing logic.

- [ ] **Step 6: Run audit tests and commit**

```powershell
uv run pytest tests/test_upstream_audit.py -q
git add tests/test_upstream_audit.py src/prescriptive_capacity_sim/upstream_audit.py docs/a-herzog-source-audit.md
git commit -m "docs: audit A-Herzog integration hooks"
```

### Task 8: Document and verify the foundation

**Files:**
- Create: `docs/a-herzog-integration.md`
- Modify: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add documentation**

Document prerequisites, upstream fetch, calibration, export, protocol semantics, reference-backend smoke run, source audit, privacy boundary, license attribution, and the explicit statement that the Java decision hook is the next implementation phase.

- [ ] **Step 2: Add project markers and test command**

Register an `upstream` pytest marker for tests that require the downloaded A-Herzog checkout. Keep network-dependent acquisition out of the default unit suite.

- [ ] **Step 3: Run the complete local verification**

Run:

```powershell
uv run pytest -q
uv run python -m prescriptive_capacity_sim.cli export-a-herzog `
  --config configs/baseline.yaml `
  --output outputs/a_herzog/model_spec.json
git diff --check
```

Expected: all tests pass, the export exists, and `git diff --check` prints no errors.

- [ ] **Step 4: Confirm no sensitive or downloaded source entered Git**

Run:

```powershell
git status --short
git ls-files data/raw .cache/upstream
```

Expected: `git ls-files` prints no raw-data or fetched-upstream paths.

- [ ] **Step 5: Commit documentation and configuration**

```powershell
git add README.md pyproject.toml docs/a-herzog-integration.md
git commit -m "docs: describe A-Herzog backend foundation"
```

## Follow-on plan gate

Do not design the Java patch from guesses. After Task 7 identifies exact classes and methods, create a second implementation plan containing the precise upstream files, minimal patch hunks, Java tests, portable JDK setup, headless bridge build, 48-period integration test, and paired-seed validation against the Python reference backend.
