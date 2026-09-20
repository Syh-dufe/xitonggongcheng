# 因果规范性服务产能调度仿真环境

本项目包含两套环境：原有的供应链履约异常聚合仿真器，以及基于 Technion Anonymous Bank 真实到达日志校准的呼叫中心离散事件仿真器。二者都把训练可用的事实日志与研究者专用的 Oracle 反事实结果严格分离，适合检验“从历史规律直接处方”与“以因果推断为决策基础的规范性 AI”之间的差异。

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

## 生成真实数据校准的半合成日志

原始数据应放在仓库外，且不得提交。当前本机数据目录为
`C:\Users\13384\Desktop\系统\data\raw\technion_anonymous_bank\extracted`。

```powershell
uv run capacity-sim callcenter-generate `
  --data-dir "C:\Users\13384\Desktop\系统\data\raw\technion_anonymous_bank\extracted" `
  --config configs/callcenter_baseline.yaml `
  --seed 20260920 `
  --output outputs/callcenter_baseline
```

该流程使用真实的逐通电话到达时刻，并从校准期估计分业务类型服务时长及带右删失的 Kaplan–Meier 耐心分布。坐席数、历史行为策略以及每个未采取动作的结果是半合成的。输出增加 `calibration_summary.json` 和 `daily_metrics.csv`；`observed_log.csv` 只含所选动作和事实结果，反事实仅存在于独立的 `oracle_counterfactuals.csv` 中。

数据来源、审计数字、事件顺序和论文表述边界见 [docs/callcenter-data-and-simulator.md](docs/callcenter-data-and-simulator.md)。

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

测试覆盖配置校验、随机种子复现、需求冲击、历史策略重叠、流量守恒、积压老化、成本、安全事件、数据防泄漏、Oracle最优行动、匿名银行格式解析、Kaplan–Meier 校准、逐通事件顺序、共同随机数反事实分支和命令行端到端输出。

## 研究使用注意事项

聚合环境生成的是纯合成历史决策日志；呼叫中心环境是“真实到达过程 + 校准分布 + 合成调度决策”的半合成数据，均不能描述为企业真实排班结果。Oracle 文件不得用于训练或调参。确认性实验应预先固定场景、数据划分、指标和随机种子，并报告对未观测混杂及决策重叠的敏感性。
