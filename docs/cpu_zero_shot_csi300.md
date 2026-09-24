# CSI300：Chronos-2 / TimesFM 3.0 本机 CPU 零样本实验

## 状态与执行位置

本说明描述实现与复现口径，不代表两个正式实验已经完成。验收以各目录的
`results/summary.json` 中 `status=complete` 为准；只有 `progress.json` 不能视为完成。
本机运行进度请查看各模型 results/progress.json 和 chronos2/logs/active_sequence.json 指向的 stdout 日志。
两个模型顺序执行；任一运行失败停止序列，不会跳过错误继续宣称成功。
无 SSH、远程 GPU、训练、微调或量化。

所有路径以下均相对于项目根目录
`C:\Users\huawei_wzn\Desktop\kronos`。

## 固定版本与独立目录

| 项目 | Chronos-2 | TimesFM 3.0 |
|---|---|---|
| 独立目录 | chronos2 | timesfm3 |
| 官方源码 | amazon-science/chronos-forecasting | google-research/timesfm |
| 源码 commit | 4dbf163c2734c089cdf7da2b86fde48862ff9c6f | e31dadd84cb26bd5153fde6687502b8312e918fb |
| HF checkpoint | amazon/chronos-2 | google/timesfm-3.0-pytorch |
| 固定 revision | 29ec3766d36d6f73f0696f85560a422f50e8498c | 43046b85ec22d584a13f8098c2ed39c889e129c2 |
| 权重 SHA256 | ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 | a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da |

各目录：upstream（干净官方源码）、.venv（独立 CPython 3.11.15）、weights、config.json、
provenance.json、requirements.lock.txt、logs、results。仅共享只读原始数据和本项目编排代码。
TimesFM 源码包版本 3.0.2，使用的仍是官方 **TimesFM 3.0 checkpoint**，不是替换模型。
TimesFM 权重的 LICENSE 保存在 weights/LICENSE；仅按非商业、非生产科研实验使用。

## 数据口径

- 输入 data/csi300_daily/csi300_daily，300 CSV；SHA256 必须与历史 dataset_manifest 一致。
- 参考 train3/csi300_csi800_aligned_seed100；读取其交易日历、有效股票日期对、日期窗口及 raw_signal。
- 训练截止 2024-12-15；验证 2025-01-02—2025-06-14；测试信号日 2025-07-01—2026-06-05。
- 这些是历史区间对齐记录，**不会在训练期或验证期拟合本实验模型**。
- 90 个真实交易日，预测其后 10 个真实交易日。最后信号日的标签允许在测试信号截止日之后。
- 恰好 226 个信号日、66,103 个唯一股票/日期键；不按模型输出筛选测试样本。
- seed=100。close 是唯一目标，open/high/low/volume 是同股票的历史协变量。
- 未来量价不传入模型；真实未来价格仅在推理后用于标签与评估。
- 每只股票是独立任务；Chronos 明确 cross_learning=False；TimesFM 单股票五变量任务。
- 两模型读取的真实历史长度都是 90；官方模型内部的 masked patch padding 不是额外真实数据。
- float32 CPU，9 个原始分位数，q0.5 作为点价格路径；不排序、不裁零、不外加归一化。
  模型内部自带的上下文归一化保留，不用全数据拟合 scaler。

收益公式：`r_h = price_h/current_close - 1`；路径信号 `mean(r_1,...,r_10)`；终点信号 `r_10`。
原 Kronos 使用已有采样预测聚合，历史 temperature=.6、top_p=.9、sample_count=10；
本实验不是按该采样方案重跑 Kronos，新模型这三个参数均为 null / 不适用。

## CPU 性能与断点续跑

固定性能样本是验证期最早五个信号日，每日按股票排序最多 32 个有效样本，共 160。
每次先单模型 batch4/线程4，然后验证 batch4/8/16 × 线程4/8；每种设置预热后测速。
吞吐量包含输入构造与模型前向，不包含正式逐日落盘和评估；全量时间估算不是完成时间保证。
RSS 每 0.1 秒采样，并记录系统最低可用内存；非常短的瞬时内存峰值可能漏采。
比较完整 9 分位数，rtol=1e-4、atol=1e-5，同时反转股票顺序验证跨样本独立性。
OOM 时按规则缩小 batch，最小为1；不删除正式样本。输出异常保存原始预测并停止。

每日期 CSV 使用临时文件+os.replace；配套 .meta.json 记录完整指纹与 CSV 哈希。
指纹含配置、原始数据、历史样本文件、权重版本及哈希、记录的依赖、运行代码与评估源码。
恢复要求身份、日期键及 CSV 内容全部匹配。代码/配置改变后不能强行复用旧输出。
run.lock 使用操作系统锁，崩溃后自动释放；锁文件存在不等于任务运行。
开发期中断结果保存在 results-archive，不覆盖也不拼入最终实验。

## 评估与已发现的历史问题

1. 原 Kronos 逐日原始文件有 **67,390 行**，历史有效测试键为 **66,103 行**。
   对照先检查无重复且没有缺失必需键，再显式按固定有效测试键对齐；其余 **1,287 行**
   保存为 kronos_outside_fixed_test_keys.csv，并写 kronos_alignment.json。
   这不是删除新模型异常样本；原始数据和原实验输出完全不改。
2. 比较前检查两个新模型与 Kronos 的真实路径/终点收益一致。
3. 复用原标量、方向、每日 IC/RankIC、五分组、Top50、NW lag9、block bootstrap10×5000 等公式。
4. 原 compute_csi800_aligned_retest_metrics.py 的组合逻辑：MIN_HOLD=5 未严格执行，
   DROP_N=5 是排名缓冲而非卖出数量上限；AER 的 excess 字段未减费用，top_net_return 已减费用；
   risk_degree=.95 未实际应用。旧口径文件明确标记 historical_proxy，不能作为严格交易仿真。
5. 另输出 strict_* 文件：复用已有 _target_holdings 实施最短持有5日、主动卖出最多5只，
   保持旧 Top50、等权单日收益与交易费用公式。离开当日有效股票池的强制退出另列审计，
   可能早于5日；strict_* 仍是单日收益代理回测，不是具有涨跌停、滑点、停牌撮合的市场仿真。
   新旧口径不可混报。ordinary cost=0.0015×原换手，AER open=.001、close=.0015、
   minimum_fee=5、notional=1,000,000、annualization=238；风险仓位0.95仍不追溯改写。
6. 当期成分股回溯有存活/前视偏差；checkpoint 是现在版本，不能声称当时可用；
   预训练数据与目标历史重叠未能排除，因此零样本仅指本项目未训练/微调。

## 输出

- results/benchmark.json、benchmark_samples.csv、model_load.json：CPU测速与内存。
- results/data_audit.json、run_identity.json：数据和运行指纹。
- results/predictions/by_date/*.csv：每股信号日、10个标签日、实际/预测价格和累计收益、
  路径均值及终点信号、全部9分位数、模型revision、状态。
- results/predictions.csv：完整汇总（66,103行）。
- results/evaluation/{model,Kronos}/：点预测/方向/路径/终点/IC/RankIC/分组/组合结果。
- results/evaluation/comparison.{json,csv,md}：同键对照；后完成模型会纳入先完成模型。
- results/failure.json 或 results/failures/：失败/异常明细，不会宣称完成。
- results/summary.json：只有推理和评估都成功才写 status=complete。

## 复现命令（PowerShell）

```powershell
Set-Location 'C:\Users\huawei_wzn\Desktop\kronos'
# 已准备环境：全流程顺序执行；不要和当前运行重复启动。
& 'C:\Users\huawei_wzn\Desktop\kronos\scripts\baselines\run_cpu_sequence.ps1' -Stage all
# 单模型数据校验或断点恢复
& 'C:\Users\huawei_wzn\Desktop\kronos\chronos2\run.ps1' -Stage preflight
& 'C:\Users\huawei_wzn\Desktop\kronos\chronos2\run.ps1' -Stage run
& 'C:\Users\huawei_wzn\Desktop\kronos\timesfm3\run.ps1' -Stage run
# 全新环境才用 setup；已有 .venv 时故意拒绝覆盖
& 'C:\Users\huawei_wzn\Desktop\kronos\scripts\baselines\setup_cpu.ps1' -Model chronos2
& 'C:\Users\huawei_wzn\Desktop\kronos\scripts\baselines\setup_cpu.ps1' -Model timesfm3
# 测试；basetemp 每次选新的工作区目录
& 'C:\Users\huawei_wzn\Desktop\kronos\chronos2\.venv\Scripts\python.exe' -m pytest tests/test_cpu_baselines.py -q -p no:cacheprovider --basetemp .runtime/test-new-run
```

TimesFM HF 大文件连接不稳定时，download_timesfm_ranged.py 从官方固定 revision 分段续传，
最终必须通过固定 SHA256；下载后在 TimesFM 环境运行 download_weights.py 记录 provenance。
分段脚本可用已安装 requests 的 Chronos 环境下载，TimesFM 推理仍只在自身环境执行。
不要删除有效预测来绕过指纹错误；协议变化应另开独立结果目录。

## 已执行测试

新增测试12项在两个独立环境均通过（未来数据隔离、日期窗口、收益转换、异常保留、
断点/哈希/键校验、batch比较、官方适配器参数、严格持仓规则等）。
完整旧测试集加 scripts/experiments 到 PYTHONPATH 后42通过、10失败；失败集中在
原 smoke schedule 条件、Windows mmap临时文件清理及缺失旧dashboard产物。未为本任务修改旧代码。
完整旧测试集并未全绿，不能将新增测试通过等同全仓库通过。


## 2026-09-17 本机实测（最终代码指纹）

| 模型 | 线程 | batch | 窗口/秒 | 峰值RSS GiB | 66,103窗口推理外推 |
|---|---:|---:|---:|---:|---:|
| Chronos-2 | 8 | 16 | 28.0881 | 0.9920 | 39.22分钟 |
| TimesFM 3.0 | 8 | 16 | 13.2530 | 1.7400 | 83.13分钟 |

六个配置全部稳定通过，Chronos最大batch/线程数值差约3.81e-6，TimesFM约5.72e-6。
数值精度一致性不是逐位完全一致保证。验证样本只用于性能选择，不用标签调参。
TimesFM完整权重1,322,898,824字节，已通过官方SHA256校验；参数330,710,976个。
Chronos参数119,477,664个。
正式序列在北京时间2026-09-17 13:38恢复，先复核Chronos已有78日期断点，随后运行TimesFM。
此记录不是最终完成声明；实时状态以progress.json与最终summary.json为准。
