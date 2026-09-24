# 数据目录

本目录提交股票池清单、字段说明和最近一次完整性校验结果。原始日线 CSV 被 `.gitignore` 排除，原因是它们体积较大且需要按数据源条款单独获取。

脚本默认目录：

```text
data/csi300_daily/csi300_daily/*.csv
data/csi800_daily/csi800_daily/*.csv
data/yahoo_sp500/yahoo_sp500/*.csv
```

CSI 日线 CSV 至少应包含 `date`、`open`、`high`、`low`、`close`、`volume` 字段。完整字段、复权方式和来源见 `csi_daily_metadata.json`；文件数量、日期范围和校验错误见 `csi_daily_validation.json`。
