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
- `validation_reference_calls.csv`：仅含业务类别、排队秒数和服务秒数的逐通话脱敏留出期诊断表；
- `calibration_manifest.json`：输入文件哈希和参数哈希。

## 验证产能候选情景

```powershell
uv run python -m prescriptive_capacity_sim.cli validate-capacity `
  --config configs/baseline.yaml `
  --candidates configs/capacity_candidates.yaml `
  --days 90 `
  --seeds 20260921 20260922 20260923 `
  --output outputs/capacity_validation
```

该命令仅运行历史管理者实际选择的一项增援行动，不读取 Oracle 或潜在结果文件。它从 `validation_reference_calls.csv` 读取逐通话留出期记录，以保留不同业务的真实样本量；总体正等待 p90 因而按全部原始留出期通话计算，而不是按业务类别分位点等权重重建。预先固定的选择得分为：放弃率误差权重 40%、平均等待时间相对误差权重 40%、总体正等待 p90 误差权重 20%。队列流量守恒误差会报告为机制检查，但不参与候选排序。

命令会输出候选—随机种子层面的诊断、候选排序和运行清单。清单以 `validation_dates_exact`、`validation_dates_truncated` 或 `validation_dates_cycled` 明确记录运行天数与留出期日期的关系；因此不会错误地把截断或循环运行称为逐日严格对齐。`configs/capacity_candidates.yaml` 中的普通/专业坐席规模和跨技能效率均是透明的半合成情景值，用于选择基准情景与开展敏感性分析，**不是**从企业历史日志中恢复的真实排班。`supervisor_emergency_agents` 虽保留在通用仿真配置中以支持后续机制开发，但当前队列内核尚未使用它；因此它被明确排除在候选校准、报告和排序之外。

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
- `simulation_validation.csv`：由精确逐通话留出期参考表和模拟日志计算的分布、等待与队列守恒诊断；
- `run_manifest.json`：配置、种子、哈希、代码版本和运行环境。

## 隐私与防泄漏

原始记录含客户编号和坐席标识，已通过 `.gitignore` 排除。处理后的表（含逐通话留出期参考表）不会保存 `customer_id`、`server` 或原始行号。训练接口不读取Oracle文件；任何使用Oracle调参的结果都不能作为确认性实验。

## 训练因果规范性策略树

`r/policytree_train.R` 复用开源的 `grf` 与 `policytree` R 包：前者拟合四行动因果森林并计算双重稳健收益分数，后者在固定深度下生成可解释的策略树。它对应观察性策略学习和多行动离线策略学习的已发表方法；实现细节见 `r/README.md`。

先从**事实**历史日志导出训练/测试数据。该步骤保留原有按 `episode_id` 的日期切分，成本目标固定为 `total_cost + 2.0 × next_queue`，并将 `counterfactual_data_used: false` 写入清单：

```powershell
uv run python -m prescriptive_capacity_sim.cli export-policy-data `
  --observed-log outputs/historical_factual_730d/observed_log.csv `
  --queue-penalty 2.0 `
  --output outputs/policy_data
```

安装 R 包后，运行深度为 3 的因果规范性树：

```powershell
$env:POLICYTREE_R_LIB = "$env:USERPROFILE/R/win-library/4.6"
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' r/policytree_train.R `
  outputs/policy_data outputs/policytree_causal 3
```

输出包含留出日期每个决策时点的 `recommendations.csv`、可解释的 `policy_tree.txt` 和重叠性诊断。该脚本会拒绝含 Oracle 文件的输入目录，且不读取潜在结果文件。

若 Windows R 无法读取包含中文字符的工作目录，可临时将当前项目映射为英文盘符后运行：

```powershell
subst P: (Get-Location).Path
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' r/policytree_train.R `
  P:/outputs/policy_data P:/outputs/policytree_causal 3
subst P: /d
```

## 测试

```powershell
uv run pytest -q
```

测试覆盖原始格式解析、脱敏、时间切分、到达与经验分布校准、队列守恒、技能匹配、优先服务、放弃、历史行动重叠、隐藏混杂、防反事实泄漏、配对评价、留出期验证和命令行端到端流程。
