from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cumulative_return_curves" / "csi300_three_model_cumulative_return_data.json"
OUT_DIR = ROOT / "results" / "cumulative_return_curves"
data = json.loads(DATA.read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 14,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for s in data["series"].values():
    dates, values = s["dates"], s["values"]
    x = list(range(len(dates)))
    ax.plot(x, values, label=f'{s["label"]} ({s["final_cumulative_return"]*100:.2f}%)', color=s["color"], linewidth=1.8)
ax.axhline(0, color="#667085", linewidth=0.8)
ax.set_title("CSI300 三模型累计收益率")
ax.set_xlabel("日期")
ax.set_ylabel("累计收益率")
common_dates = next(iter(data["series"].values()))["dates"]
tick_idx = [0, round((len(common_dates)-1)*0.25), round((len(common_dates)-1)*0.5), round((len(common_dates)-1)*0.75), len(common_dates)-1]
ax.set_xticks(tick_idx, [common_dates[i] for i in tick_idx])
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(axis="y", color="#e5eaf2", linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.text(0.5, -0.01, "Top30 等权 · 每日最多调出3只 · 最短持有5个交易日 · 单边换手成本0.15% · 2025-07-01—2026-06-05", ha="center", fontsize=8, color="#667085")
fig.savefig(OUT_DIR / "csi300_three_model_cumulative_return_curve.png")
fig.savefig(OUT_DIR / "csi300_three_model_cumulative_return_curve.pdf")
print(OUT_DIR / "csi300_three_model_cumulative_return_curve.png")
print(OUT_DIR / "csi300_three_model_cumulative_return_curve.pdf")
