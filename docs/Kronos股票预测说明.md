# Kronos 股票预测实验说明

本文档说明当前项目如何运行 Kronos 股票预测实验，以及主要输入、输出和参数含义。

## 1. 项目环境

项目使用独立虚拟环境：

```powershell
D:\Desktop\时间序列预测\股票预测\.venv_kronos
```

当前环境已安装 CUDA 版 PyTorch，可使用 GPU 推理：

```text
torch==2.5.1+cu124
device=cuda:0
```

项目中的 PowerShell 启动脚本会自动调用该虚拟环境，无需手动激活。

## 2. 主要脚本

| 文件 | 作用 |
|---|---|
| `kronos_stock_forecast.py` | 单次预测脚本，对每只股票取一个预测窗口并计算逐股票误差指标。 |
| `rolling_window_comparison.py` | 滑动窗口对比实验脚本，运行 Kronos 和 Naive，并计算 IC、RankIC、TopK、BottomK、方向准确率等指标。 |
| `run_rolling_window_kronos.ps1` | 推荐使用的启动脚本，默认使用 GPU 和已解压的 Kronos 官方源码。 |
| `merge_rolling_chunks.py` | 合并分块滚动实验结果。 |

## 3. 输入数据

股票数据目录：

```text
D:\Desktop\时间序列预测\股票预测\data\csi300_daily
```

每个 CSV 文件对应一只股票。核心字段包括：

| 字段 | 含义 |
|---|---|
| `date` | 交易日期 |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额；如果缺失，代码会自动近似生成 |

## 4. 运行滑动窗口实验

推荐使用：

```powershell
cd "D:\Desktop\时间序列预测\股票预测"

.\run_rolling_window_kronos.ps1 `
  -Device cuda:0 `
  -MaxWindowsPerStock 10 `
  -BatchSize 8 `
  -OutputDir rolling_window_outputs_gpu_recent10
```

如果显存稳定，可以把 `BatchSize` 提到 `16`。如果出现显存不足，则降到 `4`。

## 5. 主要参数

| 参数 | 含义 |
|---|---|
| `Device` | 推理设备，GPU 使用 `cuda:0`，CPU 使用 `cpu`。 |
| `MaxWindowsPerStock` | 每只股票取最近多少个滑动窗口；`10` 表示每只股票取最近 10 个窗口。 |
| `Step` | 滑动窗口每次向后移动多少个交易日，默认 `20`。 |
| `BatchSize` | 每次送入模型并行预测的窗口数量，只影响速度和显存，不改变指标含义。 |
| `MaxStocks` | 限制股票数量；`0` 表示不限制。 |
| `OutputDir` | 输出目录。 |

底层默认预测参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `model_name` | `NeoQuasar/Kronos-small` | Kronos 模型版本 |
| `tokenizer_name` | `NeoQuasar/Kronos-Tokenizer-base` | Kronos tokenizer |
| `lookback` | `400` | 每个窗口参考过去 400 个交易日 |
| `pred_len` | `20` | 每个窗口预测未来 20 个交易日 |
| `sample_count` | `1` | 采样次数 |
| `temperature` | `1.0` | 采样温度 |
| `top_p` | `0.9` | nucleus sampling 参数 |

## 6. 输出文件

滑动窗口实验会在 `OutputDir` 中生成：

| 文件 | 内容 |
|---|---|
| `rolling_predictions_all.csv` | 所有股票、窗口、预测步长的预测明细。 |
| `rolling_metrics_summary.csv` | 汇总指标。 |
| `rolling_cross_section_metrics.csv` | 逐截面的 IC、RankIC、TopK、BottomK 指标。 |
| `rolling_metrics_summary.md` | 汇总指标的 Markdown 版本。 |
| `run_config.json` | 本次运行参数。 |
| `skipped_stocks.json` | 被跳过的股票及原因。 |

## 7. 指标口径

预测收益：

```text
pred_return = pred_close / current_close - 1
```

真实收益：

```text
actual_return = actual_close / current_close - 1
```

TopK 和 BottomK 的计算方式：

1. 在同一个 `as_of_date + step` 截面内比较所有股票。
2. 按 `pred_return` 从高到低排序。
3. 前 10% 为 TopK，后 10% 为 BottomK。
4. 用真实收益计算 TopK、BottomK 和多空收益。

当前代码已经修正为按 `as_of_date + step` 做截面分组，避免把不同真实日期的股票混在一起比较。

## 8. 注意事项

1. `rolling_window_comparison.py` 输出的是截面评估，不是严格交易回测。
2. 若要汇报真实投资表现，应进一步构建日频持仓、调仓和交易成本。
3. `Naive` 是持平预测基线，主要用于流程对照，不代表可交易模型。
4. 当前正式结果建议优先看 `train3` 下的滑动窗口实验输出。

