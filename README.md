# 基于真实日志校准的半合成呼叫中心服务产能配置环境

本项目使用Technion匿名银行呼叫中心1999年逐通话日志校准需求到达、服务时间、客户耐心和需求状态，再合成历史产能行动及其反事实结果。它用于比较“从历史行动规律直接学习”的规范性方法与“校正历史选择偏差”的因果规范性方法。

## 哪些是真实的，哪些是合成的

来自真实日志并由校准期估计：

- 星期和30分钟时段的分业务到达分布；
- 服务时长经验分布；
- 排队放弃者的耐心时间经验分布；
- 优先客户比例；
- 正常、高需求、严重高需求状态及其转移。

人工设定或仿真生成：

- 每个时段的真实在岗人数和增援行动；
- 历史管理者的行动倾向；
- 未选择产能行动的潜在结果；
- 人员、等待、放弃和服务水平成本。

因此本项目是半合成实验环境，不能被描述为企业真实排班数据或真实管理决策。

## 业务和资源结构

- `regular`：`PS/PE/NW`普通业务；
- `specialist`：`IN/NE`互联网与证券专业业务；
- `callback_special`：`TT`回拨或特殊业务；
- 8个普通坐席工位、5个专业坐席工位；
- 四级临时增援行动：0%、10%、20%、30%；
- 专业坐席空闲时可按折损效率支援普通业务。

## 安装

```powershell
uv sync --extra dev
```

## 校准真实数据

```powershell
uv run python -m prescriptive_capacity_sim.cli calibrate `
  --raw-dir data/raw/technion_anonymous_bank/extracted `
  --output data/processed
```

校准输出：

- `calibration_intervals.csv`：脱敏的训练期与留出期半小时汇总；
- `calibration_parameters.json`：训练期估计参数；
- `data_quality_report.json`：异常、排除和IVR阶段流失计数；
- `calibration_manifest.json`：输入文件哈希和参数哈希。

## 生成观测日志与反事实

```powershell
uv run python -m prescriptive_capacity_sim.cli generate `
  --config configs/baseline.yaml `
  --days 365 `
  --seed 20260921 `
  --output outputs/baseline
```

运行输出：

- `observed_log.csv`：只包含历史事实行动与实现结果；
- `oracle_counterfactuals.csv`：四种行动的潜在结果，只能用于评价；
- `episode_summary.csv`：每日运行摘要；
- `policy_metrics.csv`：内置策略的配对评价；
- `simulation_validation.csv`：真实留出期与模拟分布诊断；
- `run_manifest.json`：配置、种子、哈希、代码版本和运行环境。

## 隐私与防泄漏

原始记录含客户编号和坐席标识，已通过 `.gitignore` 排除。处理后的表不会保存 `customer_id`、`server` 或原始行号。训练接口不读取Oracle文件；任何使用Oracle调参的结果都不能作为确认性实验。

## 测试

```powershell
uv run pytest -q
```

测试覆盖原始格式解析、脱敏、时间切分、到达与经验分布校准、队列守恒、技能匹配、优先服务、放弃、历史行动重叠、隐藏混杂、防反事实泄漏、配对评价、留出期验证和命令行端到端流程。
