<<<<<<< HEAD
# Kronos Stock Forecasting

面向 A 股日线数据的 Kronos 时间序列预测、Adapter 和持续学习实验仓库。项目同时保留了独立的 ODEStream 对比实验，以及收益率曲线和横截面排序评估脚本。

本仓库当前版本的重点是**可复现实验代码与实验口径**。原始行情文件、模型权重、缓存和大规模逐样本预测明细不随 Git 提交；获取数据后可以按文档中的路径放置并运行脚本。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```
=======
# Kronos A 股时间序列预测实验

本项目使用 Kronos 金融时间序列基础模型对本地 A 股日线数据进行 zero-shot 预测，并与 Naive、LSTM、SW-VMD 前端去噪和 hidden-state 门控融合方案进行统一比较。

仓库保留了从单窗口验证到 50 窗口正式基线的多轮实验数据和结果，主要评估价格误差、涨跌方向、横截面排序与简化组合表现。

## 主要内容

- 约 300 只 A 股的 OHLCV 日线数据，正式实验中 298 只有效；
- Kronos 单窗口与批量滚动窗口预测；
- Naive 和 LSTM 基线；
- SW-VMD 因果时频分解、信噪比诊断和门控融合实验；
- MAE、RMSE、R²、方向准确率、IC、RankIC、Top10、LongShort、回撤和 Sharpe 等指标；
- `train1`、`train2`、`train3` 三轮实验及完整预测明细、配置与报告。
>>>>>>> origin/main

## 项目结构

```text
.
<<<<<<< HEAD
├── continual_abcs/             # ABC 持续排序头、面板、评估和组合逻辑
├── data/                       # 股票池与数据元信息；原始日线文件不入库
├── docs/                       # 实验说明、指标定义、审计和研究记录
├── figures/                    # 结果图和 dashboard 生成脚本
├── results/                    # 少量可审阅的 HTML 曲线和摘要
├── scripts/                    # 数据、实验、评估和审计入口
├── src/                        # 可复用的模型、运行器和 CPU 基线
├── tests/                      # 单元测试和实验协议测试
└── third_party/                # Kronos 官方源码快照及其许可证
```

常用脚本按用途分类：

- `scripts/experiments/`：零样本、Adapter、Continual Learning 与回测实验；
- `scripts/data/`：面板、缓存和数据准备；
- `scripts/evaluation/`：指标计算与复测；
- `scripts/audit/`：数据泄漏、缓存对齐和结果审计。

脚本默认从仓库根目录解析 `data/`、`results/` 和 `third_party/`，因此请在仓库根目录执行命令。

## 数据准备

原始日线文件应放置在以下目录，目录名和 CSV 字段需保持不变：

```text
data/csi300_daily/csi300_daily/*.csv
data/csi800_daily/csi800_daily/*.csv
data/yahoo_sp500/yahoo_sp500/*.csv
```

股票池、字段、数据来源和最近一次校验记录见 `data/`。行情文件被 `.gitignore` 排除，以避免把本地下载副本直接提交到 GitHub。

## 主要实验入口

| 实验 | 入口 |
| --- | --- |
| Kronos CSI300 零样本 | `scripts/experiments/run_kronos_csi300_paper_zero_shot.py` |
| Adapter 持续学习 | `scripts/experiments/run_adapter_continual_csi300_90x10.py` |
| Continual ABC | `scripts/experiments/run_continual_csi300_abc_experiment.py` |
| 独立 ODEStream 24→1 | `scripts/experiments/run_odestream_csi300_24x1.py` |
| Top30/Drop3 回测 | `scripts/experiments/run_csi300_top10pct_backtest.py` |
| ODEStream 对比曲线 | `scripts/experiments/build_odestream_comparison_html.py` |

所有脚本支持 `--help`。脚本不会把 ODEStream 改造成 Kronos、Adapter、Replay 或排序头模块。

## 重要说明

不同实验的测试窗口、标签和组合评估口径可能不同，不能直接混合比较。当前成分股回溯实验存在 survivorship bias / 当前成分股前视偏差；论文或报告应明确限定为固定可评价面板上的离线结果。

## 已保存结果

`results/cumulative_return_curves/` 中保留了少量可直接浏览的结果页面和摘要，重点包括 `odestream_vs_existing/odestream_vs_existing_cumulative_return_curve.html`。完整预测明细、checkpoint 和 dashboard 数据体积较大，默认只保留在本地实验目录。

## 实验口径与限制

- Kronos/Adapter/Continual 的主要项目实验采用项目内定义的 90→10 或 90→20 路径；ODEStream 对比采用原始 close 单变量 24→1 口径，不能视为严格同任务排名。
- CSI300 采用当前成分股数据时存在 survivorship bias 和当前成分股前视偏差，结果不等同于可执行的历史成分股回测。
- 当前组合评估是研究型简化回测，不能替代包含交易成本、涨跌停、停牌、流动性和资金约束的生产级回测。

Kronos 官方源码放在 `third_party/Kronos_official_src/`，请同时遵守其原始许可证。贡献和复现实验约定见 `CONTRIBUTING.md`，指标定义和实验说明见 `docs/`。
=======
├─ stocks(1)/                  # A 股日线数据
├─ Kronos_official_src/        # Kronos 官方源码快照
├─ train1/                     # 单窗口和早期验证
├─ train2/                     # 10 窗口滚动实验
├─ train3/                     # 50 窗口基线与改进实验
├─ kronos_stock_forecast.py
├─ rolling_window_comparison.py
├─ lstm_baseline.py
├─ swvmd_kronos_experiment.py
├─ kronos_hidden_fusion_experiment.py
└─ 项目目录大纲.md
```

完整文件说明见 [项目目录大纲](项目目录大纲.md)，代码流程见 [实验代码框架说明](实验代码框架说明.md)。

## 环境安装

项目已验证环境为 Python、PyTorch 2.5.1 CUDA 12.4、Pandas 2.2.2。建议在 Windows PowerShell 中运行：

```powershell
python -m venv .venv_kronos
.\.venv_kronos\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kronos 模型和 tokenizer 默认通过 Hugging Face 标识加载：

- `NeoQuasar/Kronos-small`
- `NeoQuasar/Kronos-Tokenizer-base`

首次运行可能需要联网下载模型权重。

## 快速运行

### 正式 50 窗口 Kronos/Naive 基线

```powershell
.\train3\run_train3_baseline.ps1
```

### LSTM 基线

```powershell
.\train3\run_train3_lstm.ps1
```

### SW-VMD 前端去噪

```powershell
.\train3\run_train3_swvmd_kronos.ps1
```

### Frozen Kronos hidden-state 门控融合

```powershell
.\train3\run_hidden_fusion_swvmd.ps1
```

通用 Kronos 滚动实验也可以从根目录启动：

```powershell
.\run_rolling_window_kronos.ps1 `
  -Device cuda:0 `
  -MaxWindowsPerStock 50 `
  -Step 20 `
  -BatchSize 8 `
  -OutputDir train3\rolling_window_outputs_recent50
```

## 实验口径

正式 `train3` 基线固定使用：

| 参数 | 设置 |
|---|---:|
| 有效股票数 | 298 |
| lookback | 400 |
| pred_len | 20 |
| step | 20 |
| 最大测试窗口数/股票 | 50 |
| Kronos batch size | 8 |

项目同时评估两类收益信号：

- `single_step_20`：第 20 日预测收盘价相对当前价格的收益；
- `future_mean_1_20`：未来 1–20 日预测收盘均价相对当前价格的收益。

IC、RankIC、Top10 和 Bottom10 均按 `as_of_date + step` 构造真实横截面。

## 现有结果概览

- Naive 与 LSTM 在价格点位误差上优于 Kronos；
- Kronos 的方向准确率约为 51%，能产生非持平信号，但优势较弱；
- `future_mean_1_20` 通常比单独使用第 20 日收盘价更稳定；
- LSTM 的 Top10 多头表现较好，但 RankIC 和 LongShort 仍未证明稳定排序能力；
- 当前 SW-VMD 前端去噪及简单 hidden-state 门控融合没有带来稳定的横截面选股增益。

详细结果见：

- [train1 实验总结](train1/train1实验总结.md)
- [train2 实验总结](train2/train2实验总结.md)
- [train3 实验结果汇总](train3/train3实验结果汇总与四维分析.md)
- [train3 基线统一比较](train3/baseline_comparison_recent50/comparison_summary.md)
- [SW-VMD 增益比较](train3/swvmd_gain_comparison_recent50/swvmd_gain_summary.md)

## 实验数据与 Git LFS

本仓库有意保留多轮实验结果。以下大文件由 Git LFS 管理：

- 压缩归档 `*.zip`；
- PyTorch 权重 `*.pt`；
- PDF、Word 研究材料和实验图片；
- `train2` 和 `train3` 中的全量 `rolling_predictions_all.csv`。

首次使用：

```powershell
git lfs install
git clone <repository-url>
cd <repository-directory>
git lfs pull
```

虚拟环境和缓存不会提交，但实验结果不会被 `.gitignore` 排除。

## 研究边界

当前结果属于研究型预测与简化组合评估，并非包含交易成本、涨跌停、停牌、成交约束和严格资金曲线管理的生产级回测。仓库中的结果不构成投资建议。

## 许可与引用

本仓库暂未指定开源许可证。在公开发布或复用 Kronos 官方源码、论文、模型权重及数据前，请分别核对其原始许可证和数据使用条款。
>>>>>>> origin/main
