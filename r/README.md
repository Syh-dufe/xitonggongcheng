# 因果策略树训练

本目录使用正式发表的观察性策略学习方法的开源实现：`grf` 估计多行动因果森林和双重稳健得分，`policytree` 在固定深度限制下学习可解释的行动树。

先安装 R，并在 R 中安装依赖：

```powershell
$env:POLICYTREE_R_LIB = "$env:USERPROFILE/R/win-library/4.6"
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' -e "dir.create(Sys.getenv('POLICYTREE_R_LIB'), recursive=TRUE, showWarnings=FALSE); install.packages(c('grf','policytree','jsonlite'), lib=Sys.getenv('POLICYTREE_R_LIB'), repos='https://cloud.r-project.org')"
```

先由 Python 导出只含事实的策略学习数据：

```powershell
uv run python -m prescriptive_capacity_sim.cli export-policy-data `
  --observed-log outputs/historical_factual_730d/observed_log.csv `
  --queue-penalty 2.0 `
  --output outputs/policy_data
```

再训练深度为 3 的因果规范性策略树：

```powershell
$env:POLICYTREE_R_LIB = "$env:USERPROFILE/R/win-library/4.6"
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' r/policytree_train.R `
  outputs/policy_data outputs/policytree_causal 3
```

输入目录只能包含 `policy_training.csv`、`policy_test.csv` 和 `policy_manifest.json`。训练脚本拒绝含 Oracle 文件的目录，并要求清单明确声明 `counterfactual_data_used: false`。输出的 `recommendations.csv` 是对留出日期中每个 30 分钟状态给出的四档增援行动建议；`policy_tree.txt` 和 `training_diagnostics.json` 分别保存可解释规则与重叠性诊断。
