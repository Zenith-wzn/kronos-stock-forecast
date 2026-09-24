from __future__ import annotations

"""Generate the current continual-learning dashboard and realized return curves."""

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only
from continual_abcs.evaluate import compute_daily_metrics
from src.cpu_baselines.common import Panel

SEEDS = (200, 300, 400)
GROUPS = ("A", "B", "C")
START, END = "2025-07-01", "2026-06-05"
ROWS, DATES = 66103, 226
DATA_DIR = ROOT / "results" / "continual_multiseed"
TEMPLATE = ROOT / "results" / "dashboards" / "all_experiments_dashboard_226d.html"
DASHBOARD = ROOT / "results" / "dashboards" / "continual_multiseed_dashboard_226d.html"
DASHBOARD_DATA = ROOT / "results" / "dashboards" / "continual_multiseed_dashboard_226d_data.json"
CURVE_DIR = ROOT / "results" / "cumulative_return_curves"
CURVE_HTML = CURVE_DIR / "continual_multiseed_cumulative_return_curve.html"
CURVE_DATA = CURVE_DIR / "continual_multiseed_cumulative_return_data.json"
CURVE_DAILY = CURVE_DIR / "continual_multiseed_daily_returns.csv"
CURVE_SUMMARY = CURVE_DIR / "continual_multiseed_cumulative_return_summary.csv"

CONFIG = PortfolioConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=0.0015,
                         initial_capital=1_000_000.0, annualization_days=252.0)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 12) if math.isfinite(value) else None


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return finite(sum(values) / len(values)) if values else None


def audit_inputs() -> dict[int, dict]:
    sources = {}
    expected_keys = None
    for seed in SEEDS:
        directory = DATA_DIR / f"seed{seed}"
        common = pd.read_csv(directory / "common_predictions.csv")
        required = {"as_of_date", "stock", "actual_signal", "label_end_date", *GROUPS}
        if not required.issubset(common.columns):
            raise ValueError(f"seed{seed}: common_predictions missing {sorted(required - set(common.columns))}")
        keys = common[["as_of_date", "stock"]].astype(str)
        if keys.duplicated().any() or len(common) != ROWS or common.as_of_date.nunique() != DATES:
            raise ValueError(f"seed{seed}: expected {ROWS} unique rows across {DATES} dates")
        if common.as_of_date.min() != START or common.as_of_date.max() != END:
            raise ValueError(f"seed{seed}: unexpected date boundary")
        key_set = set(map(tuple, keys.to_numpy()))
        if expected_keys is None:
            expected_keys = key_set
        elif key_set != expected_keys:
            raise ValueError(f"seed{seed}: key mismatch against seed200")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        metrics = {}
        for group in GROUPS:
            metrics_path = directory / f"daily_metrics_{group}.csv"
            if metrics_path.exists():
                metrics[group] = pd.read_csv(metrics_path)
            else:
                # seed400 was delivered with the complete common prediction
                # table but without the derived daily-metrics exports. Keep
                # the existing result directory untouched and rebuild these
                # audited diagnostics in memory.
                source = common[["as_of_date", "stock", group, "actual_signal"]].rename(
                    columns={group: "pred_signal"}
                )
                rebuilt = compute_daily_metrics(source, top_n=50)
                metrics[group] = rebuilt.rename(columns={"n": "n_stocks"})
        for group, daily in metrics.items():
            daily["as_of_date"] = pd.to_datetime(daily["as_of_date"], errors="raise").dt.strftime("%Y-%m-%d")
            if len(daily) != DATES or daily.as_of_date.min() != START or daily.as_of_date.max() != END:
                raise ValueError(f"seed{seed} group {group}: invalid daily metrics window")
        sources[seed] = {"common": common, "summary": summary, "config": config, "daily": metrics}
    return sources


def metrics_from_daily(daily: pd.DataFrame, common: pd.DataFrame, group: str) -> dict:
    pred = pd.to_numeric(common[group], errors="coerce").to_numpy(float)
    actual = pd.to_numeric(common["actual_signal"], errors="coerce").to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    err = pred - actual
    tp = int(np.sum((pred > 0) & (actual > 0)))
    tn = int(np.sum((pred <= 0) & (actual <= 0)))
    fp = int(np.sum((pred > 0) & (actual <= 0)))
    fn = int(np.sum((pred <= 0) & (actual > 0)))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    rankic = pd.to_numeric(daily.rankic, errors="coerce")
    ic = pd.to_numeric(daily.pearson_ic, errors="coerce")
    q5q1 = pd.to_numeric(daily.q5_q1, errors="coerce")
    # The source daily metrics contain the audited Path-signal diagnostics.
    return {
        "n": int(len(pred)),
        "mae_return": finite(np.mean(np.abs(err))),
        "rmse_return": finite(np.sqrt(np.mean(err ** 2))),
        "mase_vs_zero_return_persistence": finite(np.sum(np.abs(err)) / np.sum(np.abs(actual))) if np.sum(np.abs(actual)) else None,
        "oos_r2_vs_zero_return": finite(1.0 - np.sum(err ** 2) / np.sum(actual ** 2)) if np.sum(actual ** 2) else None,
        "direction_accuracy": finite((tp + tn) / len(pred)),
        "balanced_accuracy": finite(balanced_accuracy),
        "precision": finite(tp / (tp + fp)) if tp + fp else None,
        "recall": finite(recall) if tp + fn else None,
        "f1": finite(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "specificity": finite(specificity) if tn + fp else None,
        "mcc": finite((tp * tn - fp * fn) / den) if den else None,
        "ic": {"count": int(ic.notna().sum()), "mean": finite(ic.mean()), "positive_ratio": finite((ic > 0).mean())},
        "rankic": {"count": int(rankic.notna().sum()), "mean": finite(rankic.mean()), "positive_ratio": finite((rankic > 0).mean())},
        "q5_q1": finite(q5q1.mean()),
        "mean_turnover": None,
    }


def build_dashboard_data(sources: dict[int, dict]) -> dict:
    rows = {}
    for group in GROUPS:
        per_seed = [metrics_from_daily(sources[seed]["daily"][group], sources[seed]["common"], group) for seed in SEEDS]
        rows[group] = {
            "point_forecast": {"path_signal_overall": {key: mean([x[key] for x in per_seed]) for key in (
                "n", "mae_return", "rmse_return", "mase_vs_zero_return_persistence", "oos_r2_vs_zero_return")}},
            "direction_forecast": {"path_mean": {
                key: mean([x[key] for x in per_seed]) for key in (
                    "direction_accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "mcc")},
                "cross_section_path_mean": {
                    "ic": {"count": DATES, "mean": mean([x["ic"]["mean"] for x in per_seed]), "positive_ratio": mean([x["ic"]["positive_ratio"] for x in per_seed])},
                    "rankic": {"count": DATES, "mean": mean([x["rankic"]["mean"] for x in per_seed]), "positive_ratio": mean([x["rankic"]["positive_ratio"] for x in per_seed])},
                }},
            "portfolio": {"path_mean": {
                "mean_q5_q1_path_mean": mean([x["q5_q1"] for x in per_seed]),
                "mean_top30_path_mean_raw": None,
                "mean_bottom30_path_mean": None,
                "mean_top30_excess_path_mean": None,
                "mean_top30_path_mean_net": None,
                "mean_top30_bottom30_path_mean": None,
                "mean_turnover_path_mean": None,
            }},
            "seed_summaries": {str(seed): sources[seed]["summary"] for seed in SEEDS},
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strict_boundary": {"dataset": "CSI300", "train_end": "2024-12-15", "val_start": "2025-01-02", "val_end": "2025-06-14", "test_start": START, "test_end": END, "lookback": 90, "pred_len": 10, "test_dates": DATES, "test_rows": ROWS, "seed": "200/300/400"},
        "ref": {"metrics": {"CSI300": {"config": {"dataset": "CSI300", "test_start": START, "test_end": END, "test_dates": DATES, "test_rows": ROWS, "lookback": 90, "pred_len": 10, "top_k": 30, "drop_n": 3, "min_holding_days": 5, "cost_rate": 0.0015}, "groups": {}}}},
        "continual": {"datasets": {"CSI300": {"config": {"test_start": START, "test_end": END, "resolved_test_dates": sources[SEEDS[0]]["config"]["resolved_test_dates"], "groups": list(GROUPS), "top_k": 30, "drop_n": 3, "min_holding_days": 5, "cost_rate": 0.0015, "backend": "torch", "experiment_type": "continual_learning_ranking_head", "update_policy": "month_end_test_then_train_next_test_date", "replay_policy": "70pct_recent_24m_30pct_historical_year_stratified", "seed_scope": "seed200/seed300/seed400; arithmetic mean"}, "run_summary": {"test_start": START, "test_end": END, "test_dates": DATES, "test_rows": ROWS, "seed_count": len(SEEDS), "seeds": list(SEEDS)}, "groups": rows}}},
        "experiments": [],
        "dashboard_update": {"updated_at": datetime.now(timezone.utc).isoformat(), "scope": "continual_learning_ranking_head; seed200/300/400; A/B/C arithmetic mean"},
    }


def replace_template_data(data: dict) -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    marker = "var D = "
    start = text.index(marker) + len(marker)
    end = text.index(";\nvar metricCatalog", start)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    DASHBOARD.write_text(text[:start] + payload + text[end:], encoding="utf-8")
    DASHBOARD_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_prediction_frames(sources: dict[int, dict], panel: Panel) -> dict[tuple[int, str], pd.DataFrame]:
    prices = panel.frames
    calendar = panel.calendar
    positions = {date: index for index, date in enumerate(calendar)}
    frames = {}
    for seed in SEEDS:
        common = sources[seed]["common"]
        result = common[["as_of_date", "stock", *GROUPS]].copy()
        next_dates = result.as_of_date.map(lambda date: calendar[positions[date] + 1])
        result["next_day_return"] = [float(prices[stock].loc[next_date, "close"] / prices[stock].loc[date, "close"] - 1.0) for stock, date, next_date in zip(result.stock, result.as_of_date, next_dates)]
        for group in GROUPS:
            frames[(seed, group)] = result[["as_of_date", "stock", group, "next_day_return"]].rename(columns={group: "prediction_score"})
    return frames


def make_curve_data(sources: dict[int, dict], frames: dict[tuple[int, str], pd.DataFrame], panel: Panel) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    series, daily_parts, summaries = {}, [], []
    for seed in SEEDS:
        for group in GROUPS:
            result = backtest_long_only(frames[(seed, group)], score_col="prediction_score", return_col="next_day_return", config=CONFIG, dataset="CSI300", model=f"Continual {group}", seed=seed)
            daily = result["daily"].copy()
            daily.insert(0, "curve_model", f"seed{seed}-{group}")
            daily_parts.append(daily)
            summary = dict(result["summary"])
            summary["curve_model"] = f"seed{seed}-{group}"
            summaries.append(summary)
            initial = float(summary["initial_capital"])
            series[f"seed{seed}_{group}"] = {"label": f"seed{seed} · Continual {group}", "color": {"A": "#2563eb", "B": "#d97706", "C": "#059669"}[group], "dash": "solid", "seed": seed, "group": group, "dates": ["2025-06-30"] + pd.to_datetime(daily.as_of_date).dt.strftime("%Y-%m-%d").tolist(), "values": [0.0] + [finite(x / initial) - 1.0 for x in daily.portfolio_value], "test_start": START, "test_end": END, "n_returns": int(summary["n_dates"]), "final_cumulative_return": finite(summary["net_cumulative_return"]), "annualized_return_252": finite(summary["annualized_net_return"]), "max_drawdown": finite(summary["maximum_drawdown"]), "mean_turnover": finite(summary["turnover_mean"]), "total_transaction_cost": finite(summary["cumulative_transaction_cost"]), "trade_count": int(summary["trade_count"])}
    daily_all = pd.concat(daily_parts, ignore_index=True)
    for group, color in (("A", "#1d4ed8"), ("B", "#b45309"), ("C", "#047857")):
        subset = daily_all[daily_all.curve_model.str.endswith(f"-{group}")].copy()
        mean_daily = subset.groupby("as_of_date", as_index=False).agg(net_return=("net_return", "mean"), gross_return=("gross_return", "mean"), transaction_cost=("transaction_cost", "mean"), turnover=("turnover", "mean"))
        wealth = CONFIG.initial_capital * (1.0 + mean_daily.net_return).cumprod()
        peak = np.maximum.accumulate(wealth)
        values = [0.0] + (wealth / CONFIG.initial_capital - 1.0).tolist()
        dates = ["2025-06-30"] + pd.to_datetime(mean_daily.as_of_date).dt.strftime("%Y-%m-%d").tolist()
        net = mean_daily.net_return
        key = f"mean_{group}"
        series[key] = {"label": f"Continual {group} · seed均值", "color": color, "dash": "dash", "seed": "mean", "group": group, "dates": dates, "values": [finite(x) for x in values], "test_start": START, "test_end": END, "n_returns": DATES, "final_cumulative_return": finite(values[-1]), "annualized_return_252": finite((wealth.iloc[-1] / CONFIG.initial_capital) ** (252 / DATES) - 1.0), "max_drawdown": finite((wealth / peak - 1.0).min()), "mean_turnover": finite(mean_daily.turnover.mean()), "total_transaction_cost": finite(mean_daily.transaction_cost.sum()), "trade_count": None}
    data = {"dataset": "CSI300", "generated_at": datetime.now(timezone.utc).isoformat(), "config": {"top_k": CONFIG.top_k, "drop_n": CONFIG.drop_n, "min_holding_days": CONFIG.min_holding_days, "cost_rate": CONFIG.cost_rate, "initial_capital": CONFIG.initial_capital, "signal": "common_predictions.csv 的10日 Path signal", "realized_return": "信号日收盘至下一交易日收盘", "seeds": list(SEEDS), "groups": list(GROUPS)}, "rules": ["每日按10日 Path signal 降序，选择Top30等权多头。", "之后每日最多自愿调出3只，最短持有5个交易日。", "净收益 = 下一交易日 close-to-close 组合收益 − 单边权重换手率 × 0.15%。", "seed均值曲线先对三个seed的每日净收益取算术平均，再进行复利。"], "notes": ["所有曲线使用共同的66,103个股票—日期键和226个交易日。", "曲线使用面板重建的下一交易日实现收益，不把10日actual_signal当作每日资金收益。", "当前成分股回填带来幸存者偏差与当前成分股前视偏差；结果属于离线回顾性结果。"], "series": series}
    return data, daily_all, pd.DataFrame(summaries)


def curve_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSI300 Continual 多 seed 累计收益率</title><style>:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e5eaf2}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Arial,"Microsoft YaHei",sans-serif}}.wrap{{max-width:1450px;margin:auto;padding:28px}}.head{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}}h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:18px;margin:0 0 12px}}.muted{{color:var(--muted)}}.badge{{background:#e0f2fe;color:#075985;border-radius:999px;padding:6px 10px;font-weight:700;white-space:nowrap}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin:16px 0}}.controls{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}}label{{display:inline-flex;align-items:center;gap:6px;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:6px 9px;cursor:pointer}}#chart{{width:100%;height:590px;border:1px solid var(--line);border-radius:12px;background:#fff}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.table-wrap{{overflow:auto}}.notice{{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:10px;padding:9px 11px;margin:7px 0}}.rule{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:7px 0}}.tooltip{{position:fixed;display:none;background:#111827;color:#fff;border-radius:8px;padding:8px 10px;pointer-events:none;font-size:12px;z-index:3;white-space:pre-line}}</style></head><body><div class="wrap"><div class="head"><div><h1>沪深300（CSI300）Continual 多 seed 累计收益率</h1><div class="muted">9条 seed×组明细与3条跨 seed 均值曲线；累计收益使用下一交易日 close-to-close 实现收益。</div></div><span class="badge">Top30 / Drop3 / 最短持有5日 / 0.15%成本</span></div><div class="card"><h2>累计收益率曲线</h2><div id="controls" class="controls"></div><svg id="chart" role="img" aria-label="Continual 多 seed 累计收益率曲线"></svg><div id="tip" class="tooltip"></div></div><div class="card"><h2>末期结果摘要</h2><div class="table-wrap"><table><thead><tr><th>曲线</th><th>测试区间</th><th>交易日</th><th>累计收益率</th><th>252日年化</th><th>最大回撤</th><th>平均换手率</th><th>累计成本率</th><th>交易笔数</th></tr></thead><tbody id="summary"></tbody></table></div></div><div class="card"><h2>交易规则与口径</h2><div id="rules"></div><div id="notes"></div></div></div><script>const DATA={payload};const chart=document.getElementById('chart'),controls=document.getElementById('controls'),summary=document.getElementById('summary'),rules=document.getElementById('rules'),notes=document.getElementById('notes'),tip=document.getElementById('tip');const pct=v=>v==null?'无法验证':(v*100).toFixed(2)+'%';const num=v=>v==null?'无法验证':Number(v).toLocaleString('zh-CN');const all=document.createElement('label');all.innerHTML='<input type="checkbox" data-all="1" checked> 全选';controls.appendChild(all);const allBox=all.querySelector('input');Object.keys(DATA.series).forEach(k=>{{const s=DATA.series[k],l=document.createElement('label');l.innerHTML='<input type="checkbox" data-key="'+k+'" checked><span style="color:'+s.color+';font-weight:700">━</span>'+s.label;controls.appendChild(l);const tr=document.createElement('tr');tr.innerHTML='<td>'+s.label+'</td><td>'+s.test_start+' 至 '+s.test_end+'</td><td>'+s.n_returns+'</td><td>'+pct(s.final_cumulative_return)+'</td><td>'+pct(s.annualized_return_252)+'</td><td>'+pct(s.max_drawdown)+'</td><td>'+pct(s.mean_turnover)+'</td><td>'+pct(s.total_transaction_cost)+'</td><td>'+num(s.trade_count)+'</td>';summary.appendChild(tr);}});DATA.rules.forEach(x=>{{const d=document.createElement('div');d.className='rule';d.textContent=x;rules.appendChild(d);}});DATA.notes.forEach(x=>{{const d=document.createElement('div');d.className='notice';d.textContent=x;notes.appendChild(d);}});function draw(){{const chosen=[...controls.querySelectorAll('input[data-key]:checked')].map(x=>x.dataset.key);const W=chart.clientWidth||1100,H=chart.clientHeight||590;chart.setAttribute('viewBox','0 0 '+W+' '+H);chart.innerHTML='';if(!chosen.length)return;const pad={{l:80,r:30,t:24,b:50}},vals=chosen.flatMap(k=>DATA.series[k].values);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(lo===hi){{lo-=.01;hi+=.01}}const ex=(hi-lo)*.09;lo-=ex;hi+=ex;const dates=chosen.flatMap(k=>DATA.series[k].dates).sort(),t0=Date.parse(dates[0]),t1=Date.parse(dates[dates.length-1]),X=d=>pad.l+(Date.parse(d)-t0)/Math.max(1,t1-t0)*(W-pad.l-pad.r),Y=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b),ns='http://www.w3.org/2000/svg',add=(tag,a)=>{{const e=document.createElementNS(ns,tag);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));chart.appendChild(e);return e;}};for(let i=0;i<=6;i++){{const v=lo+(hi-lo)*i/6;add('line',{{x1:pad.l,x2:W-pad.r,y1:Y(v),y2:Y(v),stroke:'#e5eaf2'}});const tx=add('text',{{x:pad.l-10,y:Y(v)+4,'text-anchor':'end',fill:'#667085','font-size':12}});tx.textContent=pct(v);}}add('line',{{x1:pad.l,x2:W-pad.r,y1:Y(0),y2:Y(0),stroke:'#9ca3af','stroke-dasharray':'4 4'}});for(let i=0;i<=5;i++){{const d=new Date(t0+(t1-t0)*i/5),ds=d.toISOString().slice(0,10),tx=add('text',{{x:X(ds),y:H-17,'text-anchor':'middle',fill:'#667085','font-size':12}});tx.textContent=ds;}}chosen.forEach(k=>{{const s=DATA.series[k];let d='';s.dates.forEach((dt,i)=>d+=(i?'L':'M')+X(dt).toFixed(2)+','+Y(s.values[i]).toFixed(2));add('path',{{d,fill:'none',stroke:s.color,'stroke-width':s.dash==='dash'?2.3:2.8,'stroke-dasharray':s.dash==='dash'?'7 5':'none','stroke-linejoin':'round','stroke-linecap':'round'}});const i=s.values.length-1;add('circle',{{cx:X(s.dates[i]),cy:Y(s.values[i]),r:4,fill:s.color}});}});chart.onmousemove=e=>{{const rect=chart.getBoundingClientRect(),px=e.clientX-rect.left,tt=t0+Math.max(0,Math.min(1,(px-pad.l)/(W-pad.l-pad.r)))*(t1-t0);let date=null,bd=Infinity;DATA.series[chosen[0]].dates.forEach(dt=>{{const q=Math.abs(Date.parse(dt)-tt);if(q<bd){{bd=q;date=dt;}}}});if(date){{const lines=[date];chosen.forEach(k=>{{const s=DATA.series[k],i=s.dates.indexOf(date);if(i>=0)lines.push(s.label+'：'+pct(s.values[i]));}});tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=lines.join('\\n');}}}};chart.onmouseleave=()=>tip.style.display='none';}}controls.addEventListener('change',e=>{{if(e.target.dataset.all)controls.querySelectorAll('input[data-key]').forEach(x=>x.checked=e.target.checked);else allBox.checked=[...controls.querySelectorAll('input[data-key]')].every(x=>x.checked);draw();}});window.addEventListener('resize',draw);draw();</script></body></html>'''


def main() -> None:
    sources = audit_inputs()
    data = build_dashboard_data(sources)
    replace_template_data(data)
    panel = Panel()
    frames = build_prediction_frames(sources, panel)
    curve_data, daily, summary = make_curve_data(sources, frames, panel)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DATA.write_text(json.dumps(curve_data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    daily.to_csv(CURVE_DAILY, index=False, encoding="utf-8-sig")
    summary.to_csv(CURVE_SUMMARY, index=False, encoding="utf-8-sig")
    CURVE_HTML.write_text(curve_html(curve_data), encoding="utf-8")
    print(json.dumps({"dashboard": str(DASHBOARD), "curve": str(CURVE_HTML), "series": len(curve_data["series"]), "dashboard_bytes": DASHBOARD.stat().st_size, "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
