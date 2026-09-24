# Kronos CSI300 零样本与 Adapter 实验交接

**更新时间**：2026-08-20  
**项目目录（本地工作区）**：`C:\Users\huawei_wzn\Desktop\Desktop\时间序列预测\股票预测`  
**远程项目目录**：`D:\Desktop\时间序列预测\股票预测`

> 本文只记录本次对话中最重要的 Kronos CSI300 零样本实验结果，以及下一步共享 Adapter 实验设计。请明确区分“已完成实验”和“拟执行实验”。

---

## 1. 当前结论

1. Kronos-small 的 CSI300 零样本推理已经完整结束，进程正常退出。
2. 本轮完成 `225/225` 个信号日，时间为 `2024-07-01—2025-06-05`，平均每个截面 `296.5822` 只股票。
3. 论文 Eq.13 主信号下呈现弱正向横截面排序能力，但稳健统计尚未达到显著：
   - RankIC `0.02603`
   - RankIC Newey-West t `1.2316`
   - 移动块 bootstrap 95% CI `[-0.01419, 0.06749]`
4. 当前结果只能称为“论文参数对齐的近似复现”，不能称为严格复现论文 Table 10 的 AER/IR。
5. 下一步计划做“全市场共享 Adapter”，但尚未启动；正式运行前必须先实现统一交易日历和公平对照样本。
6. 下一轮训练/推理的 `batch_size` 已明确要求设为 `16`。

---

## 2. 远程连接

上次可用 SSH：

```powershell
ssh -i "C:\Users\huawei_wzn\.ssh\id_ed25519_gpu4060" -o IdentitiesOnly=yes -p 13195 WangZ@25.tcp.cpolar.top
```

说明：cpolar 端口可能变化；若无法连接，应先让用户提供新的隧道地址，不要假定端口永久有效。

远程 GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`，显存约 8 GB。

---

# 第一部分：已完成的 Kronos 零样本实验

## 3. 完成状态

远程运行目录：

```text
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605
```

完成记录：

- `completed_dates = 225`
- `selected_dates = 225`
- `status = finished_selected_dates`
- `exit_code = 0`
- 完成时间：`2026-08-20 16:08:16 +08:00`
- 总推理耗时：`20470.77 秒`，约 `5.69 小时`
- 完成后 GPU 已空闲，零样本 Python 任务不再运行。

完成标志：

```text
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\process_exit.json
```

## 4. 零样本实验配置

| 项目 | 配置 |
|---|---|
| 模型 | `NeoQuasar/Kronos-small` |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` |
| 设备 | `cuda:0` |
| 数据目录 | `D:\Desktop\时间序列预测\股票预测\data\data\csi300_daily` |
| 信号日 | `2024-07-01—2025-06-05` |
| 信号日数量 | 225 |
| Lookback | 90 条该股票有效交易记录 |
| 预测长度 | 10 条未来有效交易记录 |
| Temperature | 0.6 |
| top-p | 0.9 |
| top-k | 0 |
| sample_count | 10 |
| seed | 100 |
| batch_size | 2 |
| 分组数 | 五分位 |
| Top-N | 50 |

主预测信号（论文 Eq.13）：

```text
mean(pred_close_1 ... pred_close_10) / close_t - 1
```

对称真实标签：

```text
mean(actual_close_1 ... actual_close_10) / close_t - 1
```

辅助终点信号与标签：

```text
pred_close_10 / close_t - 1
actual_close_10 / close_t - 1
```

## 5. 最终横截面指标

最终指标已在实验全部完成后重新运行统计脚本，覆盖全部 225 个日期，`errors=[]`。

### 5.1 论文 Eq.13 主口径：未来10日平均收盘收益

| 指标 | 最终结果 |
|---|---:|
| Pearson IC | 0.02211699 |
| Pearson IC 正比例 | 55.11% |
| Pearson IC Newey-West t（lag=9） | 1.0212 |
| Spearman RankIC | 0.02603374 |
| RankIC 中位数 | 0.00294748 |
| RankIC 标准差 | 0.16688967 |
| RankICIR，未年化 mean/std | 0.15599373 |
| RankICIR，sqrt(252) 年化 | 2.47632366 |
| RankIC 正比例 | 50.67% |
| RankIC IID t | 2.3399 |
| RankIC Newey-West t（lag=9） | 1.2316 |
| RankIC 移动块 bootstrap 95% CI | [-0.01419, 0.06749] |
| Q5-Q1 平均收益 | +0.07996% |
| Q5-Q1 Newey-West t | 0.3434 |
| Top50 相对全池平均超额 | +0.09944% |
| Top50 超额 Newey-West t | 0.7345 |
| Top50-Bottom50 | +0.13318% |
| Top50-Bottom50 Newey-West t | 0.5219 |
| Top50 真实赢家命中率 | 18.6756% |
| 随机命中率基线 | 约 16.85% |

主口径五分位均值：

| 分位 | 平均真实收益 |
|---|---:|
| Q1（最低预测信号） | 0.5094% |
| Q2 | 0.5754% |
| Q3 | 0.5181% |
| Q4 | 0.4461% |
| Q5（最高预测信号） | 0.5894% |

解释：方向为正，但五分位并不严格单调；稳健 t 值偏低，bootstrap 区间跨 0，因此目前只能说存在弱正向排序迹象。

### 5.2 辅助口径：第10日终点收益

| 指标 | 最终结果 |
|---|---:|
| Pearson IC | 0.04122099 |
| Spearman RankIC | 0.04812016 |
| RankICIR，未年化 mean/std | 0.28224319 |
| RankIC 正比例 | 57.78% |
| RankIC Newey-West t（lag=9） | 1.9408 |
| RankIC 移动块 bootstrap 95% CI | [-0.00073, 0.09740] |
| Q5-Q1 | +0.35777% |
| Top50 相对全池超额 | +0.24017% |
| Top50-Bottom50 | +0.32006% |

终点收益的排序表现强于 Eq.13 平均路径口径，但仍不是论文 Table 10 的组合回测指标。

## 6. 指标文件

主要输出：

```text
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\metrics\summary.json

D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\summary.json
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\metric_summary.csv
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\daily_metrics.csv
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\quantile_summary.csv
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\quantile_daily.csv
D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_paper_zero_shot_small_20240701_20250605\ranking_metrics_detailed\errors.csv
```

脚本：

```text
D:\Desktop\时间序列预测\股票预测\run_kronos_csi300_paper_zero_shot.py
D:\Desktop\时间序列预测\股票预测\compute_kronos_csi300_cross_section_metrics.py
```

## 7. 与原论文的比较

论文 Table 10 中 Kronos-small 的 CSI300 指标：

- AER：`0.1805`，即 18.05%
- IR：`1.2394`

本轮不能直接与上述数值作一一比较，原因：

1. 论文使用 Qlib 数据和 Qlib 回测引擎；本轮使用本地 CSI300 日线 CSV。
2. 论文策略为 `top-k/drop-n`、每日交易、限制每日换股数量，并要求最短持有 5 天。
3. 本轮只统计逐日横截面 IC、RankIC、分位数组合和 Top50，不是完整净值回测。
4. 本轮未按论文口径计算交易成本、指数基准超额、AER 和 IR。
5. 本轮股票池是 2026-08-20 下载的当前 CSI300 成分股，不是历史 point-in-time 成分股，存在生存偏差。
6. 本轮行情未确认与论文 Qlib 数据同源，且为原始/未复权数据。

因此结论应表述为：

> 排序方向与论文结论一致，但强度较弱；复现了弱正向横截面排序迹象，尚未复现论文 Table 10 的 AER/IR。

## 8. 已知时间对齐问题

本轮每个横截面的 `as_of_date` 一致，但输入窗口按每只股票自己的最近 90 条有效记录截取。

这意味着：

- 所有股票信号日一致；
- 但如果某股票停牌或缺失交易日，其第 k 个输入位置可能不对应其他股票相同的自然交易日；
- 因而不是严格的 90 个位置逐日历对齐。

零样本脚本包含未来扰动自检，可保证未来标签变化不会修改过去输入，但没有完成统一交易日历重索引。

---

# 第二部分：拟执行的共享 Adapter 实验

## 9. 当前状态

**Adapter 新实验尚未启动。**

用户已选择：

> 方案1：全市场所有 CSI300 股票共享同一个 Adapter。

用户还明确要求：

- 使用统一交易日历重新索引；
- 确保每只股票输入的 90 个位置对应完全相同的交易日；
- 缺失日期使用掩码或剔除；
- `batch_size=16`。

现有旧结果目录：

```text
D:\Desktop\时间序列预测\股票预测\train3\kronos_adapter_recent10_train20_ep5_seed20260521
```

旧实验仅为 `recent10/train20` 口径，不能代替新的 CSI300 完整测试期严格对照。

远程现有相关脚本：

```text
D:\Desktop\时间序列预测\股票预测\kronos_adapter_experiment.py
D:\Desktop\时间序列预测\股票预测\train3\run_kronos_adapter_experiment.ps1
```

正式运行前必须审计并修改脚本，不要直接复用旧参数启动。

## 10. Adapter 的训练对象

冻结 Kronos-small 主干，只训练轻量共享 Adapter 和输出预测头。

建议残差瓶颈结构：

```text
h' = h + W_up(GELU(W_down(LayerNorm(h))))
```

训练参数包括：

- Adapter LayerNorm 参数；
- `W_down`；
- `W_up`；
- 与线性头对照组完全相同的预测头。

含义：

- 普通线性头只学习“如何读取 Kronos 已有特征”；
- Adapter 学习“如何先修正 Kronos 特征，再读取预测结果”；
- 全市场股票共用同一组参数，不做逐股票 Adapter。

## 11. 推荐公平对照矩阵

| 组别 | Kronos主干 | 可训练部分 | 损失 | 目的 |
|---|---|---|---|---|
| A | 冻结 | 无 | 无 | 原始零样本主基线 |
| B | 冻结 | 普通线性预测头 | 预测损失 | 控制监督训练/新增头的效果 |
| C | 冻结 | Adapter + 与B相同预测头 | 预测损失 | 检验 Adapter 结构本身 |
| D | 冻结 | Adapter + 与B相同预测头 | 预测损失 + 排序损失 | 检验排序损失是否额外有效 |

公平性要求：

- 相同训练、验证和测试日期；
- 相同股票-日期样本；
- 相同隐藏特征位置；
- B/C/D 使用完全相同的输出头；
- 相同 epoch、batch、seed 和评价脚本；
- C 与 D 唯一差异是是否使用排序损失。

## 12. 统一交易日历设计

建立唯一 CSI300 主交易日历 `T`。

对每个信号日 `t`：

```text
输入日期：T[t-89] ... T[t]，共90日
标签日期：T[t+1] ... T[t+10]，共10日
```

所有股票同一位置必须对应完全相同的主交易日。

运行模型前先生成固定样本清单：

```text
eligible_stock_date_pairs.csv
```

建议字段：

- `as_of_date`
- `stock`
- 90 个输入日期或窗口起止/日历哈希
- 10 个标签日期或标签起止/日历哈希
- 输入完整性
- 标签完整性
- 缺失比例
- 是否入选
- 剔除原因

所有 A/B/C/D 组必须使用该清单的交集，避免模型间样本不同造成虚假提升。

## 13. 缺失数据规则

### 主实验：严格剔除（推荐）

如果股票在以下任一位置缺失，则剔除该股票-日期样本：

- 90日输入窗口任一主交易日缺失；
- 信号日缺失；
- 未来10日标签任一主交易日缺失；
- OHLCV 含非有限值；
- 价格非正；
- 成交量为负。

原因：官方 Kronos 推理接口没有原生暴露行情缺失掩码；直接填充可能制造虚假价格路径。严格剔除更容易审计，应作为主结果。

### 稳健性附加实验：掩码 + 有限填充

仅作为次要实验：

- 停牌/缺失日 OHLC 以前值填充；
- volume 填 0；
- 额外保留 `missing_mask`；
- 90日窗口缺失比例超过 5% 时仍剔除。

这需要修改 Kronos 输入或 Adapter 结构，已偏离官方零样本口径，不应作为主结论。

## 14. 时间切分方案

严格按时间切分，不能随机混合股票-日期样本。

| 数据段 | 信号日范围 | 用途 |
|---|---|---|
| 训练集 | 数据最早满足90日历史的信号日—`2023-12-15` | 训练共享 Adapter/预测头 |
| 第一次隔离期 | `2023-12-18—2023-12-29` | 防止10日标签与验证期重叠 |
| 验证集 | `2024-01-02—2024-06-14` | 早停和超参数选择 |
| 第二次隔离期 | `2024-06-17—2024-06-28` | 防止验证标签与测试期重叠 |
| 测试集 | `2024-07-01—2025-06-05` | 最终冻结评估 |

注意：以上日期应按主交易日历复核。隔离的实质要求是相邻数据段的未来10日标签不重叠，而不是机械依赖自然日数量。

推荐训练方式：

> 在测试期开始前训练一次共享 Adapter，使用验证集早停；进入测试期后完全冻结，不在测试期滚动重训。

该建议在本次对话中提出，但尚未收到最终“启动执行”确认。

## 15. 推荐训练参数

| 参数 | 建议值 |
|---|---:|
| Adapter bottleneck | 64 |
| Adapter dropout | 0.1 |
| batch_size | **16，用户明确要求** |
| epochs | 5，上限；配合早停 |
| 优化器 | AdamW |
| Adapter learning rate | `1e-4` |
| Head learning rate | `3e-4` |
| weight decay | `1e-4` |
| gradient clipping | 1.0 |
| early stopping patience | 2 epochs，按验证 RankIC |
| 初始 seed | 100 |
| 正式 seeds | 100 / 200 / 300 |

如果 `batch_size=16` 显存不足：

- 先停止并报告；
- 不应未经用户确认擅自降 batch；
- 可建议保持有效 batch=16，使用 micro-batch + gradient accumulation。

## 16. 损失设计

### C组：纯预测损失

```text
L_C = Huber(predicted_target, actual_target)
```

### D组：预测 + 横截面排序损失

```text
L_D = L_Huber + lambda_rank * L_rank
lambda_rank 初始建议 0.2
```

关键要求：

- `L_rank` 必须在同一个 `as_of_date` 的股票之间计算；
- 不能将不同日期股票混在一个排序损失中；
- 推荐 sampler 先选择一个信号日，再从该日有效股票中选择16只；
- `batch_size=16` 表示同一横截面内的16只股票。

主目标建议仍与 Eq.13 对齐：

```text
mean(future_close_1 ... future_close_10) / close_t - 1
```

如果输出完整10日路径，则预测头输出10个收益/价格值，再计算 Eq.13 信号；如果直接输出标量收益，必须明确其为监督排序适配，不再是原生Kronos生成路径的纯零样本形式。

## 17. 无泄漏要求

正式启动前至少加入以下断言：

1. 输入只含信号日及以前数据；
2. 标签只含信号日之后10个统一交易日；
3. 标准化参数只由训练集计算；
4. 验证/测试不得重新拟合标准化参数；
5. Adapter训练不访问验证/测试标签；
6. Early stopping只使用验证集；
7. 测试集不得选择 bottleneck、学习率、epoch 或 lambda；
8. A/B/C/D 使用同一测试样本交集；
9. 扰动未来标签后，模型输入和测试预测保持不变；
10. 每个训练排序 batch 的 `as_of_date` 必须唯一。

## 18. Adapter 最终评价指标

主信号继续使用 Eq.13。至少报告：

- Pearson IC；
- Spearman RankIC；
- RankICIR（未年化与年化分开）；
- RankIC 正比例；
- Newey-West lag=9 t 值；
- moving-block bootstrap，block=10；
- Q5-Q1；
- Top50 相对全池超额；
- Top50-Bottom50；
- Top50 命中率；
- 多随机种子均值和标准差。

成功标准建议：

1. C组 Adapter 的 RankIC 高于 B组普通线性头和 A组零样本；
2. D组高于C组，才支持“排序损失有效”；
3. 多随机种子提升方向一致；
4. RankIC 提升不能以 Q5-Q1 和 Top50 超额同时恶化为代价；
5. 对共享日期做配对 bootstrap，而不是比较两组互不相关的均值。

---

## 19. 可选的2026年近期测试

远程检查显示当前300只股票的数据均更新到 `2026-08-19`。

可新增近期样本外区间：

```text
2026-01-05—2026-08-05
```

结束于8月5日前后是为了给最后信号日保留完整未来10日标签。具体最后可用信号日应由统一主交易日历自动计算，不要手工假定。

近期测试建议作为独立报告：

1. 历史论文参数对齐区间：`2024-07-01—2025-06-05`；
2. 近期有效性区间：2026年可用完整标签区间。

不要用2026年结果反向调整2024—2025主测试的超参数。

---

## 20. 下一次接手的推荐执行顺序

1. 重新验证 SSH/cpolar 地址。
2. 确认零样本最终文件仍完整，勿重跑已完成任务。
3. 审计远程 `kronos_adapter_experiment.py`，确认旧代码训练的究竟是隐藏特征 Adapter 还是外接回归头。
4. 编写并测试统一主交易日历与 `eligible_stock_date_pairs.csv`。
5. 先跑一日/少量股票 smoke test，断言90日逐位置完全对齐。
6. 跑 A 组重新对齐后的零样本基线；不要直接将旧的非严格日历对齐结果作为唯一基线。
7. 跑 B 组线性头。
8. 跑 C 组 Adapter + 预测损失。
9. 通过 C 组后再跑 D 组 Adapter + 排序损失。
10. 最终对 A/B/C/D 做同日期、同股票交集的配对统计。
11. 如需与论文 Table 10 直接比较，另行实现 Qlib `top-k/drop-n + 最短持有5日` 回测并计算 AER/IR。

## 21. 禁止混淆的事项

- 不要把33日期的旧阶段性指标当成最终结果；最终结果为225日期版本。
- 不要把终点收益 RankIC `0.04812` 当成 Eq.13 主 RankIC；主 RankIC 是 `0.02603`。
- 不要把年化 `mean/std*sqrt(252)` 直接当成论文组合 IR。
- 不要声称已经复现论文 AER `18.05%` 或 IR `1.2394`。
- 不要把旧 `recent10/train20` Adapter 结果当成新的完整 CSI300 Adapter 对照。
- 不要在测试期上训练、早停或选择 lambda。
- 新实验未经用户确认不要直接启动；用户要求先确认实验细节。
