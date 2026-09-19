# 因果规范性服务产能调度仿真环境

本项目模拟供应链履约异常服务中心，生成带历史决策偏差的产能调度日志，并把训练可用的事实日志与研究者专用的Oracle反事实结果严格分离。它适合检验“模仿历史决策”“预测后优化”“因果规范性决策”和“安全处方”等方法。

环境不使用深度学习。随机需求、历史行为策略、动态积压、供应链冲击、反事实分支和基准评价均可独立替换。

## 安装

```powershell
uv sync --extra dev
```

## 生成仿真历史日志

```powershell
uv run python -m prescriptive_capacity_sim.cli generate `
  --config configs/baseline.yaml `
  --days 365 `
  --seed 20260919 `
  --output outputs/baseline
```

输出：

- `observed_log.csv`：算法可使用的历史事实日志；
- `oracle_counterfactuals.csv`：四种行动的潜在结果，只用于评价；
- `episode_summary.csv`：每日汇总；
- `policy_metrics.csv`：内置基准策略的配对样本外结果；
- `run_manifest.json`：完整配置、种子、运行环境和数据规模。

## 场景

`configs/` 包含基准、低/高历史偏差、高/低重叠、未观测混杂、需求冲击、产能冲击和复合冲击配置。YAML只需写相对默认值的变化。

## 内置策略

- 历史带偏策略；
- 随机策略；
- 四个固定产能等级；
- 单期Oracle；
- 三期滚动Oracle。

新方法实现以下接口即可接入 `evaluate_policies`：

```python
class MyPolicy:
    name = "my_policy"

    def act(self, state, shock, environment, rng) -> int:
        return 0
```

普通学习策略不应读取 `shock.manager_alarm`；该字段只用于仿真历史管理者和Oracle安全性实验，以模拟日志中未记录的管理者信息。

## 测试

```powershell
uv run pytest -q
```

测试覆盖配置校验、随机种子复现、需求冲击、历史策略重叠、流量守恒、积压老化、成本、安全事件、数据防泄漏、Oracle最优行动和命令行端到端输出。

## 研究使用注意事项

本项目生成的是合成历史决策日志，不能描述为企业真实日志。Oracle文件不得用于训练或调参。确认性实验应预先固定场景、数据划分、指标和随机种子，并报告对未观测混杂及决策重叠的敏感性。
