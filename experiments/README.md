# 实验目录索引

`experiments/` 是当前项目的实验归档总目录。实验按数据集、任务和方法分层保存；不同的对比实验使用不同的子目录，不以旧的训练轮次目录名混放。根目录之外的本地缓存、下载目录和历史工作目录不属于本实验归档。

## 目录结构

```text
experiments/
├── csi300_aligned/
│   ├── reference/                         # 日历、样本键、数据清单和审计文件
│   └── models/                            # 同一 CSI300 对齐任务下的模型对照
│       ├── zero_shot/                     # Kronos 零样本
│       ├── linear_head/                   # 线性预测头
│       ├── prediction_adapter/            # 预测 Adapter
│       ├── ranking_adapter/               # 排序 Adapter
│       ├── adapter_continual/             # Adapter 加持续学习
│       ├── continual_abc/                 # Continual ABC
│       ├── continual_abc_mae_gate_tol001/ # 带 MAE 门控的 Continual ABC
│       ├── odestream_direct_24x1/         # 独立 ODEStream 正式实验
│       └── odestream_direct_24x1_smoke/   # ODEStream 小规模连通性检查
├── csi300_backtest/
│   └── top10pct_drop3/                    # CSI300 Top10%/Drop3 组合对照
│       ├── page1_abcd_top30_drop3/        # 零样本、线性头和 Adapter
│       ├── page2_continual_abc_top30_drop3/ # Continual ABC
│       └── page3_adapter_continual_top30_drop3/ # Adapter 持续学习
├── baselines/                             # 非主模型基线和统一比较
│   ├── comparison_recent50/
│   ├── lstm_recent50/
│   ├── rolling_window_recent50/
│   ├── swvmd_kronos_recent50/
│   ├── swvmd_gain_recent50/
│   ├── swvmd_snr_recent10/
│   ├── swvmd_snr_recent10_clipped/
│   └── hidden_fusion_swvmd_recent10/
├── continual/                             # 持续学习和排序奖励实验
│   ├── adapter_metrics/
│   ├── sorting_reward/                    # CSI300、CSI800、S&P500 多种子
│   └── yahoo_formal/
├── sp500/                                  # S&P500 参考面板和 Adapter 持续学习
├── smoke/                                  # 独立的小规模验证实验
├── references/                             # 跨实验合并面板和来源清单
└── notes/                                  # 当前实验说明和结果注记
```

## 归档规则

- 同一数据集、同一任务下的模型对照放在同一个任务目录下的独立方法子目录中，例如 `csi300_aligned/models/`。
- 组合回测按股票池、持仓规则和交易规则单独分层，例如 `csi300_backtest/top10pct_drop3/`；不同页面或方法不混写到同一目录。
- 持续学习实验统一放入 `continual/`，并在数据集或随机种子层继续分目录。
- 每个正式实验目录保存 `run_config.json` 或等价配置、摘要指标和必要日志；预测明细、模型权重和缓存按 `.gitignore` 规则管理。
- 小规模 smoke test 与正式实验分开，不能用 smoke 结果替代正式结果。
- 结果展示页放在 `results/cumulative_return_curves/`，按 `csi300/`、`continual/`、`sp500/` 和 `odestream_vs_existing/` 分类；展示页应能回溯到这里的实验目录和配置。

## 当前主要对照

CSI300 对齐任务的主要模型目录为：

- `models/zero_shot/`：Kronos 零样本；
- `models/linear_head/`：线性头；
- `models/prediction_adapter/`：预测 Adapter；
- `models/ranking_adapter/`：排序 Adapter；
- `models/adapter_continual/`：Adapter 持续学习；
- `models/continual_abc/` 和 `models/continual_abc_mae_gate_tol001/`：Continual ABC 及门控变体；
- `models/odestream_direct_24x1/`：独立 ODEStream 24→1 对比，不接入 Kronos 组件。

ODEStream 的 24→1 单变量任务与 Kronos 的 90→10 或 90→20 任务不是完全同口径，比较时必须同时查看对应的配置和日期边界。
