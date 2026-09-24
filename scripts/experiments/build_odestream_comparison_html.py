from __future__ import annotations

"""Build an independent CSI300 cumulative-return comparison including ODEStream.

The model artifacts are kept untouched. This script only reconstructs a portfolio
from stock-level prediction files using the project's existing Top30/Drop3 rule.
All models are evaluated on their common (date, stock) keys and share one realized
next-day return table.
"""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_csi300_top10pct_backtest import clean_json, simulate


DEFAULT_TEST_START = "2025-07-01"
DEFAULT_TEST_END = "2026-06-05"
DEFAULT_OUTPUT = "results/cumulative_return_curves/odestream_vs_existing"


METHODS = {
    "odestream_direct": {
        "label": "ODEStream（24→1 close）",
        "color": "#dc2626",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/odestream_direct_24x1/predictions.csv",
        "kind": "odestream",
        "signal_definition": "pred_close / current_close - 1; ODEStream direct close-only 24→1",
    },
    "adapter_continual": {
        "label": "Adapter + Continual",
        "color": "#2563eb",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/adapter_continual/predictions.csv",
        "kind": "project",
        "signal_definition": "existing frozen-hidden Adapter continual pred_signal",
    },
    "prediction_adapter": {
        "label": "静态 Prediction Adapter",
        "color": "#0891b2",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/C_prediction_adapter/predictions.csv",
        "kind": "project",
        "signal_definition": "existing static prediction adapter pred_signal",
    },
    "ranking_adapter": {
        "label": "静态 Ranking Adapter",
        "color": "#0f766e",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/D_ranking_adapter/predictions.csv",
        "kind": "project",
        "signal_definition": "existing static ranking adapter pred_signal",
    },
    "continual_A": {
        "label": "Continual A（原始信号）",
        "color": "#059669",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions_A.csv",
        "kind": "project",
        "signal_definition": "existing continual ABC group A pred_signal",
    },
    "continual_B": {
        "label": "Continual B（冻结排序头）",
        "color": "#d97706",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions_B.csv",
        "kind": "project",
        "signal_definition": "existing continual ABC group B pred_signal",
    },
    "continual_C": {
        "label": "Continual C（持续更新）",
        "color": "#7c3aed",
        "path": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions_C.csv",
        "kind": "project",
        "signal_definition": "existing continual ABC group C pred_signal",
    },
}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def max_drawdown(wealth: pd.Series) -> float:
    values = pd.to_numeric(wealth, errors="coerce").to_numpy(dtype=float)
    if not len(values):
        return float("nan")
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1.0))


def enrich_summary(summary: dict[str, Any], daily: pd.DataFrame, method: str) -> dict[str, Any]:
    returns = pd.to_numeric(daily["net_return"], errors="coerce").dropna().to_numpy(dtype=float)
    downside = returns[returns < 0]
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan")
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")
    summary.update(
        {
            "method": method,
            "metric_status": "可验证",
            "unavailable_reason": "",
            "initial_capital": 1_000_000.0,
            "sharpe": float(np.mean(returns) / std * math.sqrt(252)) if std > 0 else float("nan"),
            "sortino": float(np.mean(returns) / downside_std * math.sqrt(252)) if downside_std > 0 else float("nan"),
            "win_day_ratio": float(np.mean(returns > 0)) if len(returns) else float("nan"),
            "trade_count": int(daily["bought"].sum() + daily["sold"].sum()),
            "cumulative_transaction_cost": float(daily["transaction_cost"].sum()),
            "cumulative_transaction_cost_amount": float(daily["transaction_cost"].sum() * 1_000_000.0),
            "net_cumulative_return": float(daily["cumulative_return"].iloc[-1]),
            "final_wealth": float(daily["wealth"].iloc[-1]),
            "maximum_drawdown": max_drawdown(daily["wealth"]),
            "annualized_net_return": float(daily["wealth"].iloc[-1] ** (252.0 / len(daily)) - 1.0),
            "mean_turnover": float(daily["turnover"].mean()),
            "turnover_total": float(daily["turnover"].sum()),
        }
    )
    return summary


def load_project_frame(path: Path, test_start: str, test_end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"as_of_date", "stock", "pred_signal"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.rename(columns={"as_of_date": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock"] = frame["stock"].astype(str).str.strip()
    frame["pred_signal"] = pd.to_numeric(frame["pred_signal"], errors="coerce")
    frame["source_actual_return"] = pd.to_numeric(frame.get("actual_return_1"), errors="coerce")
    return frame[
        (frame["date"] >= pd.Timestamp(test_start))
        & (frame["date"] <= pd.Timestamp(test_end))
    ][["date", "stock", "pred_signal", "source_actual_return"]].dropna(
        subset=["date", "stock", "pred_signal"]
    )


def load_previous_trading_dates(root: Path, frame: pd.DataFrame) -> pd.Series:
    """Map each ODEStream target date to the close at which its signal exists."""
    dates_by_stock: dict[str, np.ndarray] = {}
    data_dir = root / "data" / "csi300_daily" / "csi300_daily"
    for stock in frame["stock"].dropna().astype(str).unique():
        path = data_dir / f"{stock.replace('.', '_')}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(raw["date"], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
        dates_by_stock[stock] = dates.to_numpy(dtype="datetime64[ns]")

    previous: list[pd.Timestamp | pd.NaT] = []
    for stock, target in zip(frame["stock"], frame["target_date"]):
        dates = dates_by_stock.get(str(stock))
        if dates is None:
            previous.append(pd.NaT)
            continue
        position = int(np.searchsorted(dates, np.datetime64(target), side="left"))
        previous.append(pd.Timestamp(dates[position - 1]) if position > 0 else pd.NaT)
    return pd.Series(previous, index=frame.index, dtype="datetime64[ns]")


def load_odestream_frame(root: Path, path: Path, test_start: str, test_end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "stock", "pred_close", "true_close", "current_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["target_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock"] = frame["stock"].astype(str).str.strip()
    for col in ("pred_close", "true_close", "current_close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["pred_signal"] = frame["pred_close"] / frame["current_close"] - 1.0
    frame["source_actual_return"] = frame["true_close"] / frame["current_close"] - 1.0
    frame["date"] = load_previous_trading_dates(root, frame)
    return frame[
        (frame["date"] >= pd.Timestamp(test_start))
        & (frame["date"] <= pd.Timestamp(test_end))
    ][["date", "target_date", "stock", "pred_signal", "source_actual_return"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["date", "stock", "pred_signal", "source_actual_return"])


def normalize_frames(root: Path, test_start: str, test_end: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    loaded: dict[str, pd.DataFrame] = {}
    sources: dict[str, Any] = {}
    for key, spec in METHODS.items():
        path = root / spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"missing prediction file: {path}")
        loaded[key] = load_odestream_frame(root, path, test_start, test_end) if spec["kind"] == "odestream" else load_project_frame(path, test_start, test_end)
        loaded[key] = loaded[key].sort_values(["date", "stock"]).drop_duplicates(["date", "stock"])
        sources[key] = {"path": str(path.resolve()), "rows_before_common_intersection": int(len(loaded[key]))}

    # Existing continual ABC files do not carry actual_return_1. The Adapter file
    # is used as the single realized-return reference for every model.
    reference = loaded["adapter_continual"][["date", "stock", "source_actual_return"]].rename(
        columns={"source_actual_return": "actual_return_1"}
    )
    reference = reference.dropna(subset=["actual_return_1"]).drop_duplicates(["date", "stock"])
    key_sets = [set(zip(df["date"], df["stock"])) for df in loaded.values()]
    common_keys = set.intersection(*key_sets, set(zip(reference["date"], reference["stock"])))
    if not common_keys:
        raise RuntimeError("no common (date, stock) keys across ODEStream and project models")

    common_index = pd.MultiIndex.from_tuples(sorted(common_keys), names=["date", "stock"])
    normalized: dict[str, pd.DataFrame] = {}
    mismatch: dict[str, float | None] = {}
    for key, frame in loaded.items():
        out = frame.set_index(["date", "stock"]).loc[common_index].reset_index()
        source_returns = out["source_actual_return"]
        joined = reference.set_index(["date", "stock"]).loc[common_index, "actual_return_1"].to_numpy(dtype=float)
        source_values = source_returns.to_numpy(dtype=float)
        valid = np.isfinite(source_values) & np.isfinite(joined)
        mismatch[key] = float(np.max(np.abs(source_values[valid] - joined[valid]))) if valid.any() else None
        out["actual_return_1"] = joined
        normalized[key] = out[["date", "stock", "pred_signal", "actual_return_1"]].copy()
        sources[key]["rows_after_common_intersection"] = int(len(out))
        sources[key]["date_count_after_common_intersection"] = int(out["date"].nunique())

    metadata = {
        "common_rows": int(len(common_index)),
        "common_dates": int(common_index.get_level_values("date").nunique()),
        "common_start": str(common_index.get_level_values("date").min().date()),
        "common_end": str(common_index.get_level_values("date").max().date()),
        "actual_return_reference": "adapter_continual actual_return_1, joined by (date, stock)",
        "source_actual_return_max_abs_difference_vs_reference": mismatch,
        "sources": sources,
    }
    return normalized, metadata


def build_html(data: dict[str, Any], output: Path) -> None:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    html = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSI300 ODEStream 与持续学习模型累计收益率</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e5eaf2;--shadow:0 8px 24px rgba(20,42,80,.08)}
*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Arial,"Microsoft YaHei",sans-serif}
.wrap{max-width:1480px;margin:auto;padding:28px}.head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}h1{font-size:27px;margin:0 0 6px;overflow-wrap:anywhere}h2{font-size:18px;margin:0 0 12px}.muted{color:var(--muted)}
.badge{background:#fee2e2;color:#991b1b;border-radius:8px;padding:6px 10px;font-weight:700;white-space:normal}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);padding:18px;margin:16px 0}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}label{display:inline-flex;align-items:center;gap:6px;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:6px 9px;cursor:pointer}input{accent-color:#2563eb}
#chart{width:100%;height:590px;border:1px solid var(--line);border-radius:8px;background:#fff}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto}
.notice{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:8px;padding:9px 11px;margin:7px 0}.rule{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:7px 0}
.small{font-size:12px;color:var(--muted)}.tooltip{position:fixed;display:none;background:#111827;color:#fff;border-radius:8px;padding:8px 10px;pointer-events:none;font-size:12px;z-index:3;white-space:pre-line}
@media(max-width:650px){.wrap{padding:12px}.head{display:block}h1{font-size:22px}.card{padding:12px}#chart{height:420px}}
</style></head><body><div class="wrap"><div class="head"><div><h1>沪深300（CSI300）ODEStream 与项目持续学习模型</h1><div class="muted">累计收益率 = ∏(1 + 每日组合净收益) − 1；所有曲线在共同股票—日期键上计算。</div></div><span class="badge">ODEStream 24→1 / close 单变量</span></div>
<div class="card"><h2>累计收益率曲线</h2><div id="controls" class="controls"></div><svg id="chart" role="img" aria-label="ODEStream 与项目模型累计收益率曲线"></svg><div id="tip" class="tooltip"></div></div>
<div class="card"><h2>末期结果摘要</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>测试区间</th><th>交易日</th><th>累计收益率</th><th>252日年化</th><th>最大回撤</th><th>Sharpe</th><th>胜日率</th><th>平均换手率</th><th>累计成本</th><th>交易笔数</th></tr></thead><tbody id="summary"></tbody></table></div></div>
<div class="card"><h2>交易规则与口径</h2><div id="rules"></div><div id="notes"></div><div class="small" id="sources"></div></div></div>
<script>const DATA=__DATA__;const chart=document.getElementById('chart'),controls=document.getElementById('controls'),summary=document.getElementById('summary'),rules=document.getElementById('rules'),notes=document.getElementById('notes'),sources=document.getElementById('sources'),tip=document.getElementById('tip');
const keys=Object.keys(DATA.series),pct=v=>v==null?'无法验证':(Number(v)*100).toFixed(2)+'%',num=v=>v==null?'无法验证':Number(v).toLocaleString('zh-CN'),fmt=v=>v==null?'无法验证':Number(v).toFixed(3);const all=document.createElement('label');all.innerHTML='<input type="checkbox" data-all="1" checked> 全选';controls.appendChild(all);const allBox=all.querySelector('input');
keys.forEach(k=>{const s=DATA.series[k],l=document.createElement('label');l.innerHTML='<input type="checkbox" data-key="'+k+'" checked><span style="color:'+s.color+';font-weight:700">━</span>'+s.label;controls.appendChild(l);const tr=document.createElement('tr');tr.innerHTML='<td>'+s.label+'</td><td>'+s.test_start+' 至 '+s.test_end+'</td><td>'+s.n_returns+'</td><td>'+pct(s.net_cumulative_return)+'</td><td>'+pct(s.annualized_net_return)+'</td><td>'+pct(s.maximum_drawdown)+'</td><td>'+fmt(s.sharpe)+'</td><td>'+pct(s.win_day_ratio)+'</td><td>'+pct(s.mean_turnover)+'</td><td>'+pct(s.cumulative_transaction_cost)+'</td><td>'+num(s.trade_count)+'</td>';summary.appendChild(tr);});
DATA.rules.forEach(x=>{const d=document.createElement('div');d.className='rule';d.textContent=x;rules.appendChild(d);});DATA.notes.forEach(x=>{const d=document.createElement('div');d.className='notice';d.textContent=x;notes.appendChild(d);});sources.textContent='共同样本：'+DATA.common_rows.toLocaleString('zh-CN')+' 个股票—日期键，'+DATA.common_dates+' 个交易日；真实收益参考：'+DATA.actual_return_reference+'。';
function draw(){const chosen=[...controls.querySelectorAll('input[data-key]:checked')].map(x=>x.dataset.key);const W=chart.clientWidth||1100,H=chart.clientHeight||590;chart.setAttribute('viewBox','0 0 '+W+' '+H);chart.innerHTML='';if(!chosen.length)return;const pad={l:80,r:30,t:24,b:50},vals=chosen.flatMap(k=>DATA.series[k].values);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(lo===hi){lo-=.01;hi+=.01}const ex=(hi-lo)*.09;lo-=ex;hi+=ex;const dates=chosen.flatMap(k=>DATA.series[k].dates).sort(),t0=Date.parse(dates[0]),t1=Date.parse(dates[dates.length-1]),X=d=>pad.l+(Date.parse(d)-t0)/Math.max(1,t1-t0)*(W-pad.l-pad.r),Y=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b),ns='http://www.w3.org/2000/svg',add=(tag,a)=>{const e=document.createElementNS(ns,tag);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));chart.appendChild(e);return e;};for(let i=0;i<=6;i++){const v=lo+(hi-lo)*i/6;add('line',{x1:pad.l,x2:W-pad.r,y1:Y(v),y2:Y(v),stroke:'#e5eaf2'});const tx=add('text',{x:pad.l-10,y:Y(v)+4,'text-anchor':'end',fill:'#667085','font-size':12});tx.textContent=pct(v)}add('line',{x1:pad.l,x2:W-pad.r,y1:Y(0),y2:Y(0),stroke:'#9ca3af','stroke-dasharray':'4 4'});for(let i=0;i<=5;i++){const d=new Date(t0+(t1-t0)*i/5),ds=d.toISOString().slice(0,10),tx=add('text',{x:X(ds),y:H-17,'text-anchor':'middle',fill:'#667085','font-size':12});tx.textContent=ds}chosen.forEach(k=>{const s=DATA.series[k];let d='';s.dates.forEach((dt,i)=>d+=(i?'L':'M')+X(dt).toFixed(2)+','+Y(s.values[i]).toFixed(2));add('path',{d,fill:'none',stroke:s.color,'stroke-width':'2.8','stroke-linejoin':'round','stroke-linecap':'round'});const i=s.values.length-1;add('circle',{cx:X(s.dates[i]),cy:Y(s.values[i]),r:4,fill:s.color})});chart.onmousemove=e=>{const rect=chart.getBoundingClientRect(),px=e.clientX-rect.left,tt=t0+Math.max(0,Math.min(1,(px-pad.l)/(W-pad.l-pad.r)))*(t1-t0);let date=null,bd=Infinity;DATA.series[chosen[0]].dates.forEach(dt=>{const q=Math.abs(Date.parse(dt)-tt);if(q<bd){bd=q;date=dt}});if(date){const lines=[date];chosen.forEach(k=>{const s=DATA.series[k],i=s.dates.indexOf(date);if(i>=0)lines.push(s.label+'：'+pct(s.values[i]))});tip.style.display='block';tip.style.left=Math.max(4,Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(4,Math.min(e.clientY+12,innerHeight-tip.offsetHeight-8))+'px';tip.textContent=lines.join('\n')}};chart.onmouseleave=()=>tip.style.display='none'}
controls.addEventListener('change',e=>{if(e.target.dataset.all)controls.querySelectorAll('input[data-key]').forEach(x=>x.checked=e.target.checked);else allBox.checked=[...controls.querySelectorAll('input[data-key]')].every(x=>x.checked);draw()});window.addEventListener('resize',draw);draw();</script></body></html>'''
    output.write_text(html.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--drop-n", type=int, default=3)
    parser.add_argument("--min-hold-days", type=int, default=5)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out if args.out is not None else root / DEFAULT_OUTPUT).resolve()
    out.mkdir(parents=True, exist_ok=True)

    normalized, sample_meta = normalize_frames(root, args.test_start, args.test_end)
    cfg = {
        "dataset": "CSI300",
        "test_start": args.test_start,
        "test_end": args.test_end,
        "top_k": args.top_k,
        "drop_n": args.drop_n,
        "min_holding_days": args.min_hold_days,
        "cost_rate": args.cost_rate,
        "initial_capital": 1_000_000.0,
        "annualization_days": 252,
        "portfolio_rule": "existing run_csi300_top10pct_backtest.simulate",
        "common_key_policy": "intersection of all model prediction keys and Adapter actual_return_1 keys",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "run_config.json").write_text(json.dumps(clean_json({**cfg, "sample": sample_meta}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    series: dict[str, Any] = {}
    for key, spec in METHODS.items():
        frame = normalized[key].rename(columns={"date": "as_of_date"})
        frame.to_csv(out / f"{key}_normalized_predictions.csv", index=False, encoding="utf-8-sig")
        sim_cfg = {"top_k": args.top_k, "drop_n": args.drop_n, "min_hold_days": args.min_hold_days, "cost_rate": args.cost_rate}
        daily, holdings, summary = simulate(frame, sim_cfg)
        if daily.empty:
            raise RuntimeError(f"{key}: portfolio simulation returned no daily rows")
        daily.insert(0, "model", key)
        daily.to_csv(out / f"{key}_daily_returns.csv", index=False, encoding="utf-8-sig")
        holdings.to_csv(out / f"{key}_portfolio_holdings.csv", index=False, encoding="utf-8-sig")
        enriched = enrich_summary(summary, daily, key)
        enriched.update({"label": spec["label"], "signal_definition": spec["signal_definition"], "source": spec["path"], "source_rows": sample_meta["sources"][key]["rows_before_common_intersection"], "common_rows": sample_meta["sources"][key]["rows_after_common_intersection"]})
        summary_rows.append(enriched)
        (out / f"{key}_summary.json").write_text(json.dumps(clean_json(enriched), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        values = [0.0] + [float(v) for v in daily["cumulative_return"].tolist()]
        dates = [str((pd.Timestamp(args.test_start) - pd.Timedelta(days=1)).date())] + [str(v) for v in daily["as_of_date"].tolist()]
        series[key] = {"label": spec["label"], "color": spec["color"], "dates": dates, "values": values, **{k: enriched[k] for k in ("test_start", "test_end", "n_periods", "net_cumulative_return", "annualized_net_return", "maximum_drawdown", "sharpe", "sortino", "win_day_ratio", "mean_turnover", "cumulative_transaction_cost", "trade_count")}}
        daily_frames.append(daily)

    combined_daily = pd.concat(daily_frames, ignore_index=True)
    combined_daily.to_csv(out / "odestream_vs_existing_daily_returns.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "odestream_vs_existing_summary.csv", index=False, encoding="utf-8-sig")
    data = {"dataset": "CSI300", "config": cfg, **sample_meta, "series": series, "rules": [
        "所有模型仅使用共同的股票—日期键，避免缺失样本改变组合规模。",
        "每日按预测信号降序选择 Top30；后续每日最多主动调出3只，最短持有5个交易日。",
        "组合为等权多头，净收益 = 信号日收盘至下一交易日收盘的组合毛收益 − 单边权重换手率 × 0.15%。",
        "累计收益率按每日净收益复利，初始资金100万元；ODEStream 的收益信号由 pred_close/current_close−1 重构。",
    ], "notes": [
        "ODEStream 是独立的 close 单变量 24→1 实验；Adapter/Continual 是当前项目原有的 90→10 路径预测信号，窗口和信号定义不同。",
        "这是策略层面的参考比较，不是严格同任务、同输入窗口的模型排名。",
        "时间对齐：ODEStream 的结果行记录被预测目标日，按信号日收盘到下一交易日收盘的统一口径回退一个交易日，因此严格共同区间为 2025-07-01 至 2026-06-04（225 个交易日）；不能把 ODEStream 的 2026-06-05 目标行伪装成该日收盘后的信号。",
        "ODEStream 的真实收益与项目模型统一采用 Adapter 文件中的 actual_return_1 作为共同参考；脚本同时记录各源文件与该参考的最大绝对差异。",
        "CSI300 成分股使用当前成分股回填，存在幸存者偏差和当前成分股前视偏差；结果不等同可执行实盘收益。",
    ]}
    (out / "odestream_vs_existing_data.json").write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    build_html(data, out / "odestream_vs_existing_cumulative_return_curve.html")
    (out / "summary.json").write_text(json.dumps(clean_json({"config": cfg, "sample": sample_meta, "models": summary_rows}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean_json({"output_dir": str(out), "common_rows": sample_meta["common_rows"], "common_dates": sample_meta["common_dates"], "summary": summary_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
