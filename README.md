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

## 项目结构

```text
.
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
