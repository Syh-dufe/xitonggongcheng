# policytree 因果策略树适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从半合成事实日志导出无泄漏的日期切分策略学习数据，并用开源 `grf`/`policytree` 训练可解释的四行动因果产能策略。

**Architecture:** Python 保持事实日志、清单校验和仿真评价的唯一入口；R 脚本仅接受 Python 导出的白名单数据。训练阶段按 `episode_id` 保留日期切分，永不读取 Oracle；R 输出的建议行动由 Python 校验后再进入既有策略评价。

**Tech Stack:** Python 3.11、pandas、pytest、R、`grf`、`policytree`。

---

## File structure

- Create: `src/prescriptive_capacity_sim/policy_data.py` — 事实日志白名单导出、成本构造和清单生成。
- Create: `tests/test_policy_data.py` — 导出、日期切分、防泄漏与行动支持度单元测试。
- Create: `r/policytree_train.R` — 用 `grf`/`policytree` 训练四行动双重稳健策略树并导出建议。
- Create: `r/README.md` — R 依赖安装、训练命令、输入输出契约。
- Modify: `src/prescriptive_capacity_sim/cli.py` — 新增 `export-policy-data` 子命令。
- Modify: `README.md` — 添加训练数据导出和 R 训练说明。
- Modify: `pyproject.toml` — 不新增 Python 因果学习依赖；R 依赖在 R 文档中声明。

### Task 1: 建立事实日志导出契约

**Files:**
- Create: `tests/test_policy_data.py`
- Create: `src/prescriptive_capacity_sim/policy_data.py`

- [ ] **Step 1: Write the failing test for the public column whitelist**

```python
from pathlib import Path

import pandas as pd

from prescriptive_capacity_sim.policy_data import export_policy_data


def test_export_policy_data_keeps_only_factual_columns(tmp_path: Path) -> None:
    log = pd.DataFrame({
        "episode_id": [0, 1, 0, 1], "split": ["train", "train", "test", "test"],
        "period": [0, 0, 1, 1], "queue_regular": [1, 2, 3, 4],
        "queue_specialist": [0, 1, 1, 0], "queue_callback_special": [0, 0, 0, 1],
        "queue_priority": [0, 1, 0, 1], "max_waited_periods": [0, 1, 1, 2],
        "demand_state": ["normal", "high", "normal", "severe"], "action": [0, 1, 2, 3],
        "total_cost": [10.0, 11.0, 12.0, 13.0], "next_queue": [2, 3, 4, 5],
        "potential_cost_a0": [1.0] * 4, "oracle_action": [0] * 4,
    })
    source = tmp_path / "observed_log.csv"
    log.to_csv(source, index=False)

    result = export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)

    train = pd.read_csv(result.training_path)
    assert train.columns.tolist() == [
        "episode_id", "split", "period", "queue_regular", "queue_specialist",
        "queue_callback_special", "queue_priority", "max_waited_periods",
        "demand_state", "action", "cost",
    ]
    assert train["cost"].tolist() == [14.0, 17.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_policy_data.py::test_export_policy_data_keeps_only_factual_columns -q`

Expected: FAIL because module `prescriptive_capacity_sim.policy_data` does not exist.

- [ ] **Step 3: Implement the minimal factual export API**

```python
FEATURE_COLUMNS = (
    "period", "queue_regular", "queue_specialist", "queue_callback_special",
    "queue_priority", "max_waited_periods", "demand_state",
)
POLICY_COLUMNS = ("episode_id", "split", *FEATURE_COLUMNS, "action", "cost")


@dataclass(frozen=True)
class PolicyDataExport:
    training_path: Path
    test_path: Path
    manifest_path: Path


def export_policy_data(source: Path, output: Path, *, queue_penalty: float) -> PolicyDataExport:
    frame = pd.read_csv(source)
    _validate_factual_log(frame)
    policy = frame.loc[:, ["episode_id", "split", *FEATURE_COLUMNS, "action", "total_cost", "next_queue"]].copy()
    policy["cost"] = policy.pop("total_cost") + queue_penalty * policy.pop("next_queue")
    return _write_policy_splits(policy, output, queue_penalty=queue_penalty, source=source)
```

`_write_policy_splits` writes `policy_training.csv` and `policy_test.csv`, filters by the existing `split` column, and returns `PolicyDataExport`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_policy_data.py::test_export_policy_data_keeps_only_factual_columns -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/policy_data.py tests/test_policy_data.py
git commit -m "feat: export factual policy learning data"
```

### Task 2: 添加防泄漏和重叠性诊断

**Files:**
- Modify: `tests/test_policy_data.py`
- Modify: `src/prescriptive_capacity_sim/policy_data.py`

- [ ] **Step 1: Write failing tests for date disjointness, action support, and manifest**

```python
import json
import pytest


def _write_log(tmp_path: Path, *, splits: list[str], episodes: list[int], actions: list[int]) -> Path:
    rows = len(actions)
    frame = pd.DataFrame({
        "episode_id": episodes, "split": splits, "period": list(range(rows)),
        "queue_regular": [1] * rows, "queue_specialist": [0] * rows,
        "queue_callback_special": [0] * rows, "queue_priority": [0] * rows,
        "max_waited_periods": [0] * rows, "demand_state": ["normal"] * rows,
        "action": actions, "total_cost": [10.0] * rows, "next_queue": [1] * rows,
    })
    source = tmp_path / "observed_log.csv"
    frame.to_csv(source, index=False)
    return source


def _write_valid_log(tmp_path: Path) -> Path:
    return _write_log(
        tmp_path,
        splits=["train"] * 4 + ["test"] * 4,
        episodes=list(range(8)),
        actions=[0, 1, 2, 3, 0, 1, 2, 3],
    )


def test_export_policy_data_rejects_overlapping_episode_ids(tmp_path: Path) -> None:
    source = _write_log(tmp_path, splits=["train", "test"], episodes=[7, 7], actions=[0, 1])
    with pytest.raises(ValueError, match="episode_id"):
        export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)


def test_export_policy_data_rejects_missing_action_support(tmp_path: Path) -> None:
    source = _write_log(tmp_path, splits=["train"] * 4 + ["test"] * 4, episodes=list(range(8)), actions=[0, 1, 2, 0, 0, 1, 2, 3])
    with pytest.raises(ValueError, match="four actions"):
        export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)


def test_export_policy_data_manifest_declares_no_counterfactuals(tmp_path: Path) -> None:
    source = _write_valid_log(tmp_path)
    result = export_policy_data(source, tmp_path / "policy", queue_penalty=2.0)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counterfactual_data_used"] is False
    assert manifest["action_counts_train"] == {"0": 1, "1": 1, "2": 1, "3": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy_data.py -q`

Expected: FAIL because the initial export has no overlap/action/manifest validation.

- [ ] **Step 3: Implement only the tested validation and manifest fields**

```python
def _validate_factual_log(frame: pd.DataFrame) -> None:
    # Forbidden columns are permitted in the source but are never selected; require factual columns instead.
    required = {"episode_id", "split", *FEATURE_COLUMNS, "action", "total_cost", "next_queue"}
    if not required.issubset(frame.columns):
        raise ValueError("observed log is missing required factual columns")
    if frame.loc[:, list(required)].isna().any().any():
        raise ValueError("observed log contains missing policy-learning values")
    if set(frame["split"]) != {"train", "test"}:
        raise ValueError("observed log must contain train and test splits")
    if set(frame.loc[frame["split"].eq("train"), "episode_id"]).intersection(frame.loc[frame["split"].eq("test"), "episode_id"]):
        raise ValueError("episode_id occurs in both train and test splits")
    actions = set(frame.loc[frame["split"].eq("train"), "action"])
    if actions != {0, 1, 2, 3}:
        raise ValueError("training data must contain all four actions")
```

`_write_policy_splits` writes `policy_manifest.json` with SHA-256 of the source file, feature list, `queue_penalty`, cost definition, train/test episode counts, train action counts, and `counterfactual_data_used: false`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy_data.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/policy_data.py tests/test_policy_data.py
git commit -m "feat: validate factual policy exports"
```

### Task 3: 暴露可复现命令行接口

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/prescriptive_capacity_sim/cli.py`

- [ ] **Step 1: Write a failing CLI parser test**

```python
from prescriptive_capacity_sim.cli import build_parser


def test_cli_parses_export_policy_data() -> None:
    args = build_parser().parse_args([
        "export-policy-data", "--observed-log", "outputs/historical_factual_730d/observed_log.csv",
        "--queue-penalty", "2.0", "--output", "outputs/policy_data",
    ])
    assert args.command == "export-policy-data"
    assert args.queue_penalty == 2.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_parses_export_policy_data -q`

Expected: FAIL because `export-policy-data` is not a recognized command.

- [ ] **Step 3: Add the parser and dispatch path**

```python
export_policy = commands.add_parser("export-policy-data")
export_policy.add_argument("--observed-log", required=True, type=Path)
export_policy.add_argument("--queue-penalty", required=True, type=float)
export_policy.add_argument("--output", required=True, type=Path)

if args.command == "export-policy-data":
    export_policy_data(args.observed_log, args.output, queue_penalty=args.queue_penalty)
    return 0
```

Import `export_policy_data` from `.policy_data`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_cli_parses_export_policy_data -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/prescriptive_capacity_sim/cli.py tests/test_cli.py
git commit -m "feat: add policy data export command"
```

### Task 4: 加入官方 R 策略树训练脚本

**Files:**
- Create: `r/policytree_train.R`
- Create: `r/README.md`

- [ ] **Step 1: Write an executable input-contract check in R**

```r
required_columns <- c(
  "episode_id", "split", "period", "queue_regular", "queue_specialist",
  "queue_callback_special", "queue_priority", "max_waited_periods",
  "demand_state", "action", "cost"
)
stopifnot(identical(names(train), required_columns))
stopifnot(identical(sort(unique(train$action)), 0:3))
stopifnot(!anyNA(train))
```

- [ ] **Step 2: Run the script without inputs to verify it fails clearly**

Run: `Rscript r/policytree_train.R`

Expected: nonzero exit with `Usage: Rscript r/policytree_train.R <policy-data-dir> <output-dir> <tree-depth>`.

- [ ] **Step 3: Implement the minimal official-package training script**

```r
if (!requireNamespace("grf", quietly = TRUE) || !requireNamespace("policytree", quietly = TRUE)) {
  stop("Install required packages: install.packages(c('grf', 'policytree'))")
}

forest <- grf::multi_arm_causal_forest(
  X = model.matrix(~ . - 1, data = train[feature_columns]),
  Y = -train$cost,
  W = train$action + 1
)
gamma_hat <- policytree::double_robust_scores(forest)
tree <- policytree::policy_tree(X_train, gamma_hat, depth = tree_depth)
recommended_action <- as.integer(predict(tree, X_test)) - 1L
```

The script writes `recommendations.csv` with only `episode_id`, `period`, and `recommended_action`, then writes `policy_tree.txt`, `training_diagnostics.json`, and a copy of the input manifest. It rejects any input directory containing `oracle_counterfactuals.csv` as a defensive guard.

- [ ] **Step 4: Run R syntax validation**

Run: `Rscript -e "parse(file='r/policytree_train.R'); cat('R syntax OK\\n')"`

Expected: `R syntax OK`.

- [ ] **Step 5: Document exact installation and invocation**

```markdown
Rscript -e "install.packages(c('grf','policytree'), repos='https://cloud.r-project.org')"
Rscript r/policytree_train.R outputs/policy_data outputs/policytree_causal 3
```

State explicitly that this trains only on `policy_training.csv`; `policy_test.csv` receives recommendations and Oracle stays outside the training directory.

- [ ] **Step 6: Commit**

```powershell
git add r/policytree_train.R r/README.md
git commit -m "feat: add policytree causal training adapter"
```

### Task 5: 文档化完整流程并开展回归验证

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the README workflow**

```powershell
uv run python -m prescriptive_capacity_sim.cli export-policy-data `
  --observed-log outputs/historical_factual_730d/observed_log.csv `
  --queue-penalty 2.0 `
  --output outputs/policy_data
Rscript r/policytree_train.R outputs/policy_data outputs/policytree_causal 3
```

Explain that train/test are split by simulated date and the four action labels map to 0%, 10%, 20%, 30% temporary capacity augmentation.

- [ ] **Step 2: Run full Python test suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run a factual-only smoke export**

Run: `uv run python -m prescriptive_capacity_sim.cli export-policy-data --observed-log outputs/historical_factual_730d/observed_log.csv --queue-penalty 2.0 --output outputs/policy_data_smoke`

Expected: `policy_training.csv`, `policy_test.csv`, and `policy_manifest.json` exist; manifest declares `counterfactual_data_used: false`.

- [ ] **Step 4: Commit**

```powershell
git add README.md tests/test_policy_data.py
git commit -m "docs: document causal policytree workflow"
```
