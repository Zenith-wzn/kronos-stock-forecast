# 实验结果

这里仅提交少量可审阅的 HTML 曲线和聚合摘要。实验配置、模型输出、持续学习日志和审计材料统一以 `experiments/` 为归档入口；完整预测明细、checkpoint、隐藏状态缓存和大规模 dashboard 数据按对应实验目录管理，并由 `.gitignore` 排除。

结果文件必须同时记录数据集、时间边界、模型口径和组合规则。结果页面不新增实验口径，也不引用仓库外的旧训练目录。生成脚本位于 `scripts/experiments/` 与 `figures/`。

建议的结果布局：

```text
results/cumulative_return_curves/
├── csi300/                 # CSI300 模型与组合曲线
├── continual/              # 持续学习曲线
├── sp500/                  # S&P500 对照曲线
└── odestream_vs_existing/  # ODEStream 独立对比曲线
```
