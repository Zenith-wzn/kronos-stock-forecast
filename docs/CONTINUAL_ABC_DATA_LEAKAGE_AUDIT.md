# 持续学习 ABC 实验时间泄漏审计

**审计日期**：2026-08-25  
**审计对象**：`D:\Desktop\时间序列预测\股票预测\train3\kronos_csi300_continual_abc_abcd_strict90x10_20260825_seed100_v3\run`  
**核心问题**：是否把尚未完成测试、或标签尚未成熟的未来测试样本放入训练集。

## 总体结论：PASS（无须因未来测试集泄漏而重跑）

对运行代码、缓存生成代码、远程 train/val/test 元数据、90→10 日期窗口以及三次持续更新训练池进行了交叉核验。未发现尚未测试的未来测试样本进入训练，也未发现标签结束日期晚于更新日期的样本进入 replay。

## 1. 固定实验口径

- 测试区间锁定为 `2026-05-27` 至 `2026-08-05`。
- 输入窗口锁定为 90 个交易日，输出窗口锁定为 10 个交易日。
- 正式运行要求恰好 50 个测试日期。
- 面板实际为 50 个测试日期、14,842 个 `(as_of_date, stock)` 样本键。

代码证据：

- `run_continual_csi300_abc_experiment.py:34-37`
- `run_continual_csi300_abc_experiment.py:654-658`
- `run_continual_csi300_abc_experiment.py:698-703`
- `continual_abcs/panel.py:120-150`

## 2. 初始 train / val / test 时间隔离

远程数据实测：

| split | 行数 | as_of_date | label_end_date | 日期数 |
|---|---:|---|---|---:|
| train | 455,166 | 2016-05-17 ～ 2023-12-15 | 2016-05-31 ～ 2023-12-29 | 1,846 |
| val | 31,575 | 2024-01-02 ～ 2024-06-14 | 2024-01-16 ～ 2024-06-28 | 107 |
| test | 14,842 | 2026-05-27 ～ 2026-08-05 | 2026-06-10 ～ 2026-08-19 | 50 |

检查结果：

- train 中 `as_of_date >= 2026-05-27`：0 行。
- train 中 `label_end_date >= 2026-05-27`：0 行。
- val 中 `as_of_date >= 2026-05-27`：0 行。
- val 中 `label_end_date >= 2026-05-27`：0 行。
- train/val、train/test、val/test 的 `(as_of_date, stock)` 重叠均为 0。
- 初始 B 只使用 train 拟合；固定 val 只用于初始评价和持续更新候选模型的门控。

代码证据：

- `run_continual_csi300_abc_experiment.py:666-673`
- `run_continual_csi300_abc_experiment.py:707-717`
- `run_continual_csi300_abc_experiment.py:778-787`

## 3. 持续更新严格遵循 test-then-train

每天的执行顺序为：

1. 使用当天更新前的 C checkpoint 产生 A/B/C 预测；
2. 只有当日属于更新日时，才在预测之后检查可用标签；
3. 测试样本必须同时满足：
   - `as_of_date < update_date`：样本已经在更早的测试日完成预测；
   - `label_end_date <= update_date`：完整 10 日未来标签在更新日已经成熟；
4. 候选模型被接受后，新版本只从下一测试日生效。

代码证据：

- 当日先预测：`run_continual_csi300_abc_experiment.py:724-748`
- 预测后才更新：`run_continual_csi300_abc_experiment.py:750-778`
- 成熟标签条件：`run_continual_csi300_abc_experiment.py:753-756`
- replay 测试池条件：`run_continual_csi300_abc_experiment.py:557-564`
- 接受后切换版本：`run_continual_csi300_abc_experiment.py:779-803`

## 4. 三次更新的训练池复核

| 更新日期 | 可进入 replay 的测试行 | 测试 as_of_date 范围 | `as_of_date >= 更新日` | `label_end_date > 更新日` | 结果 |
|---|---:|---|---:|---:|---|
| 2026-05-29 | 0 | 无 | 0 | 0 | 跳过，无成熟新标签 |
| 2026-06-30 | 4,151 | 2026-05-27 ～ 2026-06-15 | 0 | 0 | 合法；更新接受，版本 0→1 |
| 2026-07-31 | 10,969 | 2026-05-27 ～ 2026-07-17 | 0 | 0 | 合法；候选被固定 val 门控拒绝 |

实际 `update_log.csv` 与复算一致：

- 2026-06-30：4,151 个已成熟测试行，选择 replay 120,000 行；新 checkpoint 从 2026-07-01 生效。
- 2026-07-31：10,969 个已成熟测试行，其中新增 6,818 行；候选未通过验证门控，版本保持 1。

实际 C 版本区间：

- version 0：2026-05-27 ～ 2026-06-30；
- version 1：2026-07-01 ～ 2026-08-05。

这证明 2026-06-30 当天预测仍使用 version 0，更新后的 version 1 到 2026-07-01 才开始用于预测。

## 5. replay 索引对齐

- train 在股票筛选后被重置索引，`_source_row=np.arange(len(train_meta))` 与筛选后的 `train_h/train_y` 行位置一致。
- test 选取正式区间后重置索引，`_source_row=test_meta.index` 与 `test_h/test_y` 行位置一致。
- `_materialize_replay()` 分别按 `_source` 取对应 hidden/labels，再用同一排序顺序同步重排特征、标签和元数据。

代码证据：

- split 筛选保持数组和 metadata 同步：`run_continual_csi300_abc_experiment.py:195-212`
- train/test 重置索引：`run_continual_csi300_abc_experiment.py:675-688`
- replay source row：`run_continual_csi300_abc_experiment.py:554-564`
- replay materialize：`run_continual_csi300_abc_experiment.py:610-634`

结论：未发现 `_source_row` 造成 hidden、label 与 metadata 错配。

## 6. 缓存标签构造

缓存生成脚本按面板中的 `input_dates` 读取 90 日历史输入，按 `label_dates` 读取之后 10 个交易日的未来价格，并将 10 日未来收益写入 labels。hidden 仅由输入窗口生成，不读取未来 label 窗口。

代码证据：

- `tmp_remote_prepare_hidden_cache_20260820.py:40-46`
- `continual_abcs/panel.py:120-150`

## 7. 过程合理性边界：未来可用性筛选

面板构造函数会检查完整的 `90 + 10` 日窗口是否都有有效行情，再决定某个 `(as_of_date, stock)` 是否进入样本集合。因此，测试样本集合的“可评价资格”使用了未来 10 日数据是否完整这一信息。这不会把未来价格值送入模型输入，也不会把未来测试样本放进训练集；但从严格在线交易模拟角度看，它属于**事后可评价样本筛选**，可能带来退市、停牌或缺失数据相关的样本选择偏差。

本实验又将测试键锁定到参考 Adapter 的同一批键，因此 A/B/C 之间的比较仍然公平。建议论文或报告将其描述为“固定可评价面板上的离线回测”，不要表述成完全无幸存者偏差的实时交易仿真。

代码证据：

- 完整 100 日窗口有效性筛选：`continual_abcs/panel.py:120-139`
- Adapter 测试键锁定：`prepare_continual_csi300_abc_panel.py:136-163`

## 注意事项（不构成当前直接泄漏）

1. `updated_after_prediction=True` 表示“该日预测完成后运行过更新”，不是“该日预测使用了更新后的模型”。建议以后改名为 `update_ran_after_prediction`，避免误读。
2. 当前代码依赖输入缓存本身保持严格时间切分。此次远程缓存已通过日期范围、标签范围、键重叠和 panel 精确键检查。
3. 建议在运行器中加入 fail-fast 断言，永久防止未来替换缓存时发生泄漏：
   - `train/val label_end_date < test_start`；
   - 每次 replay 都断言 `test as_of_date < update_date`；
   - 每次 replay 都断言所有 `label_end_date <= update_date`；
   - 记录每次实际选中 replay 行的最大 `as_of_date` 和最大 `label_end_date`。

## 最终判定

**PASS**：本次实验的时间顺序合理，没有把尚未进行测试的未来测试集放入训练集；测试日当天先预测后更新，新 checkpoint 下一测试日才生效。现有结果不需要因为该类数据泄漏问题重跑。

