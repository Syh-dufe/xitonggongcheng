# 实验运行记录

所有支持论文主张的运行都必须记录。当前输出属于探索性结果，不能直接表述为确认性证据。

| 日期 | 假设 | 命令 | Git提交 | 种子 | 配置 | 数据版本 | 指标 | 结果 | 运行时间 | 状态 | 备注 |
|---|---|---|---|---:|---|---|---|---|---|---|---|
| 2026-09-19 | 默认环境能生成有重叠的四行动历史日志 | `uv run python -m prescriptive_capacity_sim.cli generate --config configs/baseline.yaml --days 30 --seed 20260919 --output outputs/final_baseline` | 未提交（本机未配置Git身份） | 20260919 | `configs/baseline.yaml` | simulator 0.1.0 | 行动覆盖、泄漏、Oracle一致性 | 1,440行；四行动均出现；Oracle不一致0；泄漏列0 | 2.576秒 | exploratory | 21个训练episode、9个测试episode；最小已选行动概率0.008988 |
| 2026-09-19 | 复合冲击应激活积压与安全指标 | `uv run python -m prescriptive_capacity_sim.cli benchmark --config configs/shock_compound.yaml --days 30 --seed 20260919 --output outputs/final_shock` | 未提交（本机未配置Git身份） | 20260919 | `configs/shock_compound.yaml` | simulator 0.1.0 | 总成本、服务率、积压、安全违反率 | 历史策略安全违反率0.05347；固定零增援0.08403；滚动Oracle经验成本最低 | 2.507秒 | exploratory | 仅用于环境健全性检查，不支持算法优越性主张 |

## 确认性实验冻结项

- 数据划分：按episode划分，训练期早于测试期；
- 主要指标：样本外总成本，越低越好；
- 安全指标：关键客户安全违反率，越低越好；
- 随机种子：在确认性运行前预先登记；
- 禁止根据测试集结果更改生成机制、指标或基准而不重新标记为探索性实验。
