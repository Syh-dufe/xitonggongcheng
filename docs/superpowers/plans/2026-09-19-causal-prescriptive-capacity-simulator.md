# Causal Prescriptive Capacity Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python simulator that generates biased observational capacity-allocation logs and leakage-safe oracle counterfactuals for dynamic supply-chain fulfillment exceptions.

**Architecture:** A small installable package separates configuration, state transitions, stochastic demand, historical behavior, oracle branching, logging, policies, evaluation, and CLI concerns. All randomness flows through explicit NumPy generators, and oracle branches consume pre-sampled exogenous shocks so action comparisons use common random numbers.

**Tech Stack:** Python 3.13, NumPy, pandas, PyYAML, pytest, uv

---

### Task 1: Project scaffold and validated configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/prescriptive_capacity_sim/__init__.py`
- Create: `src/prescriptive_capacity_sim/config.py`
- Create: `configs/baseline.yaml`
- Test: `tests/test_config.py`

- [ ] Write failing tests proving defaults are valid and invalid transition matrices, costs, and action levels are rejected.
- [ ] Run `uv run pytest tests/test_config.py -v` and verify import/config failures.
- [ ] Implement frozen dataclasses `SimulationConfig`, `DemandConfig`, `BehaviorConfig`, `CostConfig`, `SafetyConfig`, YAML loading, and validation.
- [ ] Re-run the test and verify it passes.

The public API is:

```python
cfg = SimulationConfig.default()
cfg = SimulationConfig.from_yaml("configs/baseline.yaml")
cfg.validate()
```

### Task 2: State, stochastic demand, and historical behavior

**Files:**
- Create: `src/prescriptive_capacity_sim/state.py`
- Create: `src/prescriptive_capacity_sim/demand.py`
- Create: `src/prescriptive_capacity_sim/behavior.py`
- Test: `tests/test_stochastic_components.py`

- [ ] Write failing tests for deterministic seeded shocks, larger severe-state arrival means, valid softmax probabilities, action sampling, and lower-temperature overlap concentration.
- [ ] Run the focused test and confirm the missing-module failures.
- [ ] Implement `SystemState`, `ExogenousShock`, `PeriodResult`, `DemandProcess.sample_shock`, and `HistoricalBehaviorPolicy.probabilities/act`.
- [ ] Re-run and verify all focused tests pass.

The behavior score is an ordered-action softmax whose risk index uses backlog, urgent share, critical share, disruption severity, capacity availability, and optional manager alarm.

### Task 3: Dynamic environment and conservation laws

**Files:**
- Create: `src/prescriptive_capacity_sim/environment.py`
- Test: `tests/test_environment.py`

- [ ] Write failing tests for reproducible reset, nonnegative state, flow conservation, service bounded by available work, monotone temporary staffing cost, aging, and safety violation calculation.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement `CapacityEnvironment.reset`, `sample_exogenous`, `transition`, and `step` with class-priority service allocation and explicit cost components.
- [ ] Re-run and verify the focused tests pass.

The transition invariant is:

```text
next_backlog[class] = prior_backlog[class] + arrivals[class] - completed[class]
```

### Task 4: Oracle branching and leakage-safe historical logs

**Files:**
- Create: `src/prescriptive_capacity_sim/oracle.py`
- Create: `src/prescriptive_capacity_sim/logging.py`
- Test: `tests/test_logging.py`

- [ ] Write failing tests that each observed row contains one realized outcome, oracle rows contain all four potential outcomes, oracle action minimizes potential cost, and generated data are seed-reproducible.
- [ ] Run the test and confirm failure because the generator is absent.
- [ ] Implement one-step common-random-number branching and `generate_dataset` returning observed, oracle, and episode-summary DataFrames.
- [ ] Re-run and verify the tests pass.

The generator must never merge oracle potential outcomes into the observed frame.

### Task 5: Policies and paired out-of-sample evaluation

**Files:**
- Create: `src/prescriptive_capacity_sim/policies.py`
- Create: `src/prescriptive_capacity_sim/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] Write failing tests for fixed, random, history, myopic-oracle, and rolling-oracle policy interfaces; paired evaluation must report total cost, regret, service levels, backlog, temporary capacity, and safety violations.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement the policy protocol and paired evaluator using fixed episode seeds across policies.
- [ ] Re-run and verify the focused tests pass.

### Task 6: CLI, scenario configs, documentation, and end-to-end run

**Files:**
- Create: `src/prescriptive_capacity_sim/cli.py`
- Create: `configs/bias_low.yaml`
- Create: `configs/bias_high.yaml`
- Create: `configs/overlap_low.yaml`
- Create: `configs/hidden_confounding.yaml`
- Create: `configs/shock_compound.yaml`
- Create: `README.md`
- Create: `runs.md`
- Test: `tests/test_cli.py`

- [ ] Write a failing CLI test that invokes a two-day run in a temporary directory and checks all five required output files and manifest contents.
- [ ] Run the test and verify the CLI is missing.
- [ ] Implement `generate` and `benchmark` subcommands, scenario YAMLs, README commands, and the exploratory run-log template.
- [ ] Re-run the focused test and verify it passes.
- [ ] Run `uv run pytest -q` and verify the full suite passes.
- [ ] Run a small smoke experiment and inspect row counts, schemas, nonnegative values, action coverage, and oracle minimization.

### Task 7: Packaging and completion verification

**Files:**
- Modify as needed only to fix verification failures.

- [ ] Run `uv sync`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run python -m prescriptive_capacity_sim.cli generate --config configs/baseline.yaml --days 3 --seed 20260919 --output outputs/smoke`.
- [ ] Run `uv run python -m prescriptive_capacity_sim.cli benchmark --config configs/baseline.yaml --days 3 --seed 20260919 --output outputs/smoke_benchmark`.
- [ ] Inspect generated manifests and CSV schemas.
- [ ] Record commands, seed, config, runtime, and exploratory status in `runs.md`.
