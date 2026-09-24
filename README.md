# Kronos A 股时间序列预测实验

本仓库用于研究 Kronos 在 A 股日线数据上的时间序列预测、Adapter、持续学习、横截面排序和简化组合回测。仓库还保留一个独立的 ODEStream 对比实验；ODEStream 使用自己的 close 单变量 24→1 流式方案，不接入 Kronos、Adapter、Replay 或排序头。

文档和目录只对应当前仓库中的代码、数据说明、实验记录和结果页面。原始行情文件、模型权重、缓存以及大规模逐样本预测明细由 `.gitignore` 排除；准备好数据后可按下文路径运行实验。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

Kronos 官方源码快照位于 `third_party/Kronos_official_src/`。首次运行 Kronos 实验时，模型和 tokenizer 可能需要从 Hugging Face 下载。

## 项目结构

```text
.
├── continual_abcs/       # Continual ABC 持续排序头、面板、评估和组合逻辑
├── data/                 # 股票池、字段说明和数据校验；原始行情不入库
├── docs/                 # 实验协议、指标定义、审计和研究记录
├── experiments/          # 所有正式实验记录的统一根目录
│   ├── csi300_aligned/   # CSI300 对齐任务：参考数据和模型对照
│   ├── csi300_backtest/  # CSI300 组合回测及分方法结果
│   ├── baselines/        # Naive、LSTM、滚动窗口和 SW-VMD 基线
│   ├── continual/        # 持续学习、排序奖励和多种子结果
│   ├── references/       # 跨实验合并面板和指标恢复文件
│   ├── smoke/            # 小规模连通性、路径和形状检查
│   ├── sp500/            # S&P500 对照实验
│   ├── notes/            # 当前实验记录和结果说明
│   └── README.md         # 实验目录索引和归档约定
├── figures/              # 图表和 dashboard 生成脚本
├── results/              # 可直接浏览的 HTML 曲线和聚合摘要
├── scripts/              # 数据准备、实验运行、评估和审计入口
├── src/                  # 可复用模型、运行器和 CPU 基线
├── tests/                # 单元测试和实验协议测试
└── third_party/          # 第三方源码及许可证
```

`experiments/` 是实验归档的总目录。不同数据集、任务、模型和回测规则使用独立子目录，目录名表达实验含义。详细索引和放置规则见 [`experiments/README.md`](experiments/README.md)。`results/` 只保存适合直接浏览的 HTML 曲线和聚合摘要，不替代实验归档。仓库根目录保留的 Python/PowerShell 文件是通用入口或分析工具，不作为实验结果目录使用。

## 根目录脚本

当前仓库根目录保留以下通用入口和分析代码：

- `kronos_stock_forecast.py`、`rolling_window_comparison.py`：Kronos 预测和滚动窗口统一评估；
- `naive_baseline.py`、`lstm_baseline.py`：Naive 与 LSTM 基线；
- `swvmd_kronos_experiment.py`、`kronos_hidden_fusion_experiment.py`：SW-VMD 和 hidden-state 融合实验；
- `compare_baselines.py`、`compare_naive_kronos_metrics.py`、`compare_swvmd_gain.py`：基线和增益分析；
- `analyze_swvmd_snr.py`、`kronos_direction_probabilities.py`、`merge_rolling_chunks.py`：诊断、方向概率和结果合并工具。

正式实验运行器位于 `scripts/experiments/`，数据、评估和审计脚本分别位于 `scripts/data/`、`scripts/evaluation/` 和 `scripts/audit/`。

## 数据准备

原始日线文件按以下路径放置，目录名和 CSV 字段保持不变：

```text
data/csi300_daily/csi300_daily/*.csv
data/csi800_daily/csi800_daily/*.csv
data/yahoo_sp500/yahoo_sp500/*.csv
```

CSI 日线 CSV 至少应包含 `date`、`open`、`high`、`low`、`close` 和 `volume` 字段。股票池、字段说明、数据来源和最近一次校验记录见 `data/`。原始行情文件由 `.gitignore` 排除，避免将本地下载副本直接提交到 GitHub。

## 实验入口

| 实验 | 入口 |
| --- | --- |
| Kronos CSI300 零样本 | `scripts/experiments/run_kronos_csi300_paper_zero_shot.py` |
| Adapter 持续学习 | `scripts/experiments/run_adapter_continual_csi300_90x10.py` |
| Continual ABC | `scripts/experiments/run_continual_csi300_abc_experiment.py` |
| 独立 ODEStream 24→1 | `scripts/experiments/run_odestream_csi300_24x1.py` |
| CSI300 Top10%/Drop3 回测 | `scripts/experiments/run_csi300_top10pct_backtest.py` |
| ODEStream 对比曲线 | `scripts/experiments/build_odestream_comparison_html.py` |

所有脚本支持 `--help`。建议从仓库根目录执行，并将输出目录显式指定到 `experiments/` 的相应子目录。

## 当前实验布局

实验目录的完整清单和各层职责见 [`experiments/README.md`](experiments/README.md)。当前 CSI300 模型对照集中在 `experiments/csi300_aligned/models/`，CSI300 组合对照集中在 `experiments/csi300_backtest/top10pct_drop3/`，通用基线集中在 `experiments/baselines/`，持续学习和多种子排序结果集中在 `experiments/continual/`。小规模检查、参考面板、S&P500 对照和实验说明分别位于对应的独立子目录。

每个正式实验目录应至少包含 `run_config.json` 或等价配置、摘要指标和必要的日志。大规模预测明细、checkpoint 和缓存可只保留在本地；实验目录中的说明文件应记录数据集、时间边界、窗口、标签、更新规则、交易成本和文件含义。`odestream_direct_24x1_smoke/` 是实际存在的小规模 ODEStream 连通性结果，正式结果在 `odestream_direct_24x1/`。

## 结果与比较口径

`results/cumulative_return_curves/` 保存少量可直接浏览的累计收益率 HTML 页面和聚合摘要，并按 `csi300/`、`continual/`、`sp500/` 和 `odestream_vs_existing/` 分类。ODEStream 与当前项目已有模型的独立比较位于 `results/cumulative_return_curves/odestream_vs_existing/`。

不同实验的测试窗口、标签、股票池、交易成本和组合规则可能不同，不能把指标直接混合排序。Kronos、Adapter 和 Continual 的主要路径采用项目内定义的 90→10 或 90→20 任务；ODEStream 采用原始 close 单变量 24→1 任务，因此只能作为明确标注口径的独立对比。

当前成分股回溯实验存在 survivorship bias 和当前成分股前视偏差；结果应描述为固定可评价面板上的离线研究结果，而不是历史实时成分股回测。组合结果也不替代包含涨跌停、停牌、流动性、资金约束和完整交易成本模型的生产级回测。

## 复现与贡献

指标定义和实验协议见 `docs/`，脚本职责见 `scripts/README.md`，结果目录约定见 `results/README.md`。提交代码时请保留运行命令、配置和数据边界，并避免将生成的全量预测文件混入源代码变更。

Kronos 官方源码及其许可证位于 `third_party/Kronos_official_src/`。使用第三方源码、模型权重和数据时，请分别遵守其原始许可证与数据使用条款。
