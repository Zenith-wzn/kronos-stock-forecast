from __future__ import annotations

"""Rebuild CSI300 cumulative-return curves for Kronos, Chronos-2, and TimesFM 3.

The holding and cost rules intentionally match the existing CSI300 HTML:
Top30 equal weight, at most three voluntary exits per day, a five-trading-day
minimum holding period, and 0.15% cost times one-way weight turnover.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only
from src.cpu_baselines.common import Panel, keyset, require_keys
OUT_DIR = ROOT / "results" / "cumulative_return_curves"
MODEL_SPECS = {
    "Kronos": {
        "label": "Kronos",
        "color": "#2563eb",
        "source": ROOT / "train3" / "csi300_csi800_aligned_seed100" / "experiments" / "A_zero_shot" / "predictions" / "by_date",
        "source_kind": "directory",
        "point_estimator": "sampled forecast path mean",
    },
    "Chronos-2": {
        "label": "Chronos-2",
        "color": "#d97706",
        "source": ROOT / "chronos2" / "results" / "predictions.csv",
        "source_kind": "file",
        "point_estimator": "q0.5 forecast path mean",
    },
    "TimesFM 3": {
        "label": "TimesFM 3",
        "color": "#059669",
        "source": ROOT / "timesfm3" / "results" / "predictions.csv",
        "source_kind": "file",
        "point_estimator": "q0.5 forecast path mean",
    },
}

CONFIG = PortfolioConfig(
    top_k=30,
    drop_n=3,
    min_holding_days=5,
    cost_rate=0.0015,
    initial_capital=1_000_000.0,
    annualization_days=252.0,
)


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("no prediction CSV files found")
    return pd.concat(frames, ignore_index=True)


def canonical_predictions(raw: pd.DataFrame, model: str, expected_keys: pd.DataFrame) -> pd.DataFrame:
    required = {"stock", "as_of_date", "pred_path_mean_return", "actual_close_1", "current_close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{model}: missing columns {sorted(missing)}")
    keyset(raw)
    wanted = expected_keys[["stock", "as_of_date"]].copy()
    membership = raw.merge(wanted, on=["stock", "as_of_date"], how="outer", indicator=True, validate="one_to_one")
    missing_keys = int(membership["_merge"].eq("right_only").sum())
    if missing_keys:
        raise ValueError(f"{model}: missing {missing_keys} fixed test keys")
    raw = raw.merge(wanted, on=["stock", "as_of_date"], how="inner", validate="one_to_one")
    require_keys(raw, expected_keys)
    score = pd.to_numeric(raw["pred_path_mean_return"], errors="coerce")
    current = pd.to_numeric(raw["current_close"], errors="coerce")
    actual1 = pd.to_numeric(raw["actual_close_1"], errors="coerce")
    frame = pd.DataFrame({
        "as_of_date": pd.to_datetime(raw["as_of_date"], errors="raise"),
        "stock": raw["stock"].astype(str),
        "prediction_score": score,
        "next_day_return": actual1 / current - 1.0,
    })
    if not np.isfinite(frame[["prediction_score", "next_day_return"]].to_numpy(float)).all():
        raise ValueError(f"{model}: non-finite score or one-step return")
    return frame.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)


def load_predictions(spec: dict, model: str, expected_keys: pd.DataFrame) -> pd.DataFrame:
    source = Path(spec["source"])
    if spec["source_kind"] == "directory":
        raw = read_csvs(sorted(source.glob("*.csv")))
    else:
        raw = pd.read_csv(source)
    return canonical_predictions(raw, model, expected_keys)


def round_or_none(value, digits=10):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if math.isfinite(value) else None


def make_series(result: dict, label: str, color: str, point_estimator: str, source: Path) -> dict:
    daily = result["daily"].copy()
    summary = dict(result["summary"])
    initial = float(summary["initial_capital"])
    start = pd.Timestamp(daily["as_of_date"].iloc[0]) - pd.Timedelta(days=1)
    dates = [start.date().isoformat()] + pd.to_datetime(daily["as_of_date"]).dt.date.astype(str).tolist()
    values = [0.0] + (pd.to_numeric(daily["portfolio_value"]) / initial - 1.0).tolist()
    return {
        "label": label,
        "color": color,
        "dash": "solid",
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "point_estimator": point_estimator,
        "dates": dates,
        "values": [round_or_none(v, 10) for v in values],
        "test_start": pd.Timestamp(daily["as_of_date"].iloc[0]).date().isoformat(),
        "test_end": pd.Timestamp(daily["as_of_date"].iloc[-1]).date().isoformat(),
        "n_returns": int(summary["n_dates"]),
        "final_cumulative_return": round_or_none(summary["net_cumulative_return"]),
        "annualized_return_252": round_or_none(summary["annualized_net_return"]),
        "max_drawdown": round_or_none(summary["maximum_drawdown"]),
        "mean_turnover": round_or_none(summary["turnover_mean"]),
        "total_transaction_cost": round_or_none(summary["cumulative_transaction_cost"]),
        "trade_count": int(summary["trade_count"]),
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSI300 三模型累计收益率</title><style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e5eaf2;--shadow:0 8px 24px rgba(20,42,80,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Arial,"Microsoft YaHei",sans-serif}}.wrap{{max-width:1450px;margin:auto;padding:28px}}.head{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}}h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:18px;margin:0 0 12px}}.muted{{color:var(--muted)}}.badge{{background:#e0f2fe;color:#075985;border-radius:999px;padding:6px 10px;font-weight:700;white-space:nowrap}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:18px;margin:16px 0}}.controls{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}}label{{display:inline-flex;align-items:center;gap:6px;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:6px 9px;cursor:pointer}}input{{accent-color:#2563eb}}#chart{{width:100%;height:590px;border:1px solid var(--line);border-radius:12px;background:#fff}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.table-wrap{{overflow:auto}}.notice{{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:10px;padding:9px 11px;margin:7px 0}}.rule{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:7px 0}}.tooltip{{position:fixed;display:none;background:#111827;color:#fff;border-radius:8px;padding:8px 10px;pointer-events:none;font-size:12px;z-index:3;white-space:pre-line}}</style></head><body><div class="wrap"><div class="head"><div><h1>沪深300（CSI300）三模型累计收益率</h1><div class="muted">累计收益率 = ∏(1 + 组合净收益) − 1；三模型使用完全相同的股票—日期键、实际收益和交易规则。</div></div><span class="badge">Top30 / Drop3 / 最短持有5日 / 0.15%成本</span></div><div class="card"><h2>累计收益率曲线</h2><div id="controls" class="controls"></div><svg id="chart" role="img" aria-label="Kronos、Chronos-2 与 TimesFM 3 累计收益率曲线"></svg><div id="tip" class="tooltip"></div></div><div class="card"><h2>末期结果摘要</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>测试区间</th><th>交易日</th><th>累计收益率</th><th>252日年化</th><th>最大回撤</th><th>平均换手率</th><th>累计成本率</th><th>交易笔数</th></tr></thead><tbody id="summary"></tbody></table></div></div><div class="card"><h2>交易规则与口径</h2><div id="rules"></div><div id="notes"></div></div></div><script>
const DATA={payload};
const chart=document.getElementById('chart'),controls=document.getElementById('controls'),summary=document.getElementById('summary'),rules=document.getElementById('rules'),notes=document.getElementById('notes'),tip=document.getElementById('tip');
const keys=Object.keys(DATA.series);const pct=v=>v==null?'无法验证':(v*100).toFixed(2)+'%';const num=v=>v==null?'无法验证':Number(v).toLocaleString('zh-CN');
const all=document.createElement('label');all.innerHTML='<input type="checkbox" data-all="1" checked> 全选';controls.appendChild(all);const allBox=all.querySelector('input');
keys.forEach(k=>{{const s=DATA.series[k],l=document.createElement('label');l.innerHTML='<input type="checkbox" data-key="'+k+'" checked><span style="color:'+s.color+';font-weight:700">━</span>'+s.label;controls.appendChild(l);const tr=document.createElement('tr');tr.innerHTML='<td>'+s.label+'</td><td>'+s.test_start+' 至 '+s.test_end+'</td><td>'+s.n_returns+'</td><td>'+pct(s.final_cumulative_return)+'</td><td>'+pct(s.annualized_return_252)+'</td><td>'+pct(s.max_drawdown)+'</td><td>'+pct(s.mean_turnover)+'</td><td>'+pct(s.total_transaction_cost)+'</td><td>'+num(s.trade_count)+'</td>';summary.appendChild(tr);}});
DATA.rules.forEach(x=>{{const d=document.createElement('div');d.className='rule';d.textContent=x;rules.appendChild(d);}});DATA.notes.forEach(x=>{{const d=document.createElement('div');d.className='notice';d.textContent=x;notes.appendChild(d);}});
function draw(){{const chosen=[...controls.querySelectorAll('input[data-key]:checked')].map(x=>x.dataset.key);const W=chart.clientWidth||1100,H=chart.clientHeight||590;chart.setAttribute('viewBox','0 0 '+W+' '+H);chart.innerHTML='';if(!chosen.length)return;const pad={{l:80,r:30,t:24,b:50}};const vals=chosen.flatMap(k=>DATA.series[k].values);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(lo===hi){{lo-=.01;hi+=.01}}const ex=(hi-lo)*.09;lo-=ex;hi+=ex;const dates=chosen.flatMap(k=>DATA.series[k].dates).sort(),t0=Date.parse(dates[0]),t1=Date.parse(dates[dates.length-1]);const X=d=>pad.l+(Date.parse(d)-t0)/Math.max(1,t1-t0)*(W-pad.l-pad.r),Y=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b);const ns='http://www.w3.org/2000/svg';const add=(tag,a)=>{{const e=document.createElementNS(ns,tag);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));chart.appendChild(e);return e;}};for(let i=0;i<=6;i++){{const v=lo+(hi-lo)*i/6;add('line',{{x1:pad.l,x2:W-pad.r,y1:Y(v),y2:Y(v),stroke:'#e5eaf2'}});const tx=add('text',{{x:pad.l-10,y:Y(v)+4,'text-anchor':'end',fill:'#667085','font-size':12}});tx.textContent=pct(v);}}add('line',{{x1:pad.l,x2:W-pad.r,y1:Y(0),y2:Y(0),stroke:'#9ca3af','stroke-dasharray':'4 4'}});for(let i=0;i<=5;i++){{const d=new Date(t0+(t1-t0)*i/5),ds=d.toISOString().slice(0,10),tx=add('text',{{x:X(ds),y:H-17,'text-anchor':'middle',fill:'#667085','font-size':12}});tx.textContent=ds;}}chosen.forEach(k=>{{const s=DATA.series[k];let d='';s.dates.forEach((dt,i)=>d+=(i?'L':'M')+X(dt).toFixed(2)+','+Y(s.values[i]).toFixed(2));add('path',{{d,fill:'none',stroke:s.color,'stroke-width':'2.8','stroke-linejoin':'round','stroke-linecap':'round'}});const i=s.values.length-1;add('circle',{{cx:X(s.dates[i]),cy:Y(s.values[i]),r:4,fill:s.color}});}});chart.onmousemove=e=>{{const rect=chart.getBoundingClientRect(),px=e.clientX-rect.left,tt=t0+Math.max(0,Math.min(1,(px-pad.l)/(W-pad.l-pad.r)))*(t1-t0);let date=null,bd=Infinity;DATA.series[chosen[0]].dates.forEach(dt=>{{const q=Math.abs(Date.parse(dt)-tt);if(q<bd){{bd=q;date=dt;}}}});if(date){{const lines=[date];chosen.forEach(k=>{{const s=DATA.series[k],i=s.dates.indexOf(date);if(i>=0)lines.push(s.label+'：'+pct(s.values[i]));}});tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=lines.join('\\n');}}}};chart.onmouseleave=()=>tip.style.display='none';}}
controls.addEventListener('change',e=>{{if(e.target.dataset.all)controls.querySelectorAll('input[data-key]').forEach(x=>x.checked=e.target.checked);else allBox.checked=[...controls.querySelectorAll('input[data-key]')].every(x=>x.checked);draw();}});window.addEventListener('resize',draw);draw();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = Panel()
    expected = panel.test[["stock", "as_of_date"]].copy()
    series, daily_parts, summaries = {}, [], []
    for model, spec in MODEL_SPECS.items():
        predictions = load_predictions(spec, model, expected)
        result = backtest_long_only(
            predictions,
            score_col="prediction_score",
            return_col="next_day_return",
            config=CONFIG,
            dataset="CSI300",
            model=model,
            seed=100,
        )
        daily = result["daily"].copy()
        daily.insert(0, "curve_model", model)
        daily_parts.append(daily)
        summary = dict(result["summary"])
        summary["point_estimator"] = spec["point_estimator"]
        summaries.append(summary)
        series[model] = make_series(result, spec["label"], spec["color"], spec["point_estimator"], Path(spec["source"]))

    # Cross-model realized labels must be identical before plotting comparable curves.
    frames = [load_predictions(spec, model, expected)[["as_of_date", "stock", "next_day_return"]].rename(columns={"next_day_return": model}) for model, spec in MODEL_SPECS.items()]
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.merge(frame, on=["as_of_date", "stock"], validate="one_to_one")
    base = joined[list(MODEL_SPECS)[0]].to_numpy(float)
    for model in list(MODEL_SPECS)[1:]:
        if not np.allclose(base, joined[model].to_numpy(float), rtol=1e-10, atol=1e-12):
            raise ValueError(f"realized one-step returns disagree for {model}")

    data = {
        "dataset": "CSI300",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": CONFIG.top_k,
            "drop_n": CONFIG.drop_n,
            "min_holding_days": CONFIG.min_holding_days,
            "cost_rate": CONFIG.cost_rate,
            "annualization_days": CONFIG.annualization_days,
            "initial_capital": CONFIG.initial_capital,
            "signal": "predicted 10-day path mean return",
            "realized_return": "signal-date close to next-trading-day close",
            "turnover": "half L1 change in equal portfolio weights including cash; initial build = 1.0",
        },
        "rules": [
            "每日按预测的未来10日路径平均收益降序排列，选择Top30（CSI300约Top10%）等权多头。",
            "初始日建满30只；之后每日最多自愿调出3只，优先调出当前预测排名最低且已满足持有期的股票，再按预测排名补入。",
            "最短持有5个交易日；股票若退出当日可观测股票池，则强制退出，不受最短持有期和Drop3限制。",
            "每日组合毛收益使用信号日收盘至下一交易日收盘收益；净收益 = 毛收益 − 单边权重换手率 × 0.15%。",
            "累计收益率按每日净收益复利，初始资金100万元；三模型使用相同66,103个股票—日期键和226个交易日。",
        ],
        "notes": [
            "该交易规则与既有 CSI300 HTML 的 Top30/Drop3/最短持有5日/0.15%换手成本口径一致。",
            "当前CSI300成分股被回填到历史区间，存在幸存者偏差和当前成分股前视偏差；曲线属于离线回顾性结果。",
            "预测信号不同：Kronos使用既有采样预测路径，Chronos-2与TimesFM 3使用q0.5预测路径。",
        ],
        "series": series,
    }

    data_path = args.out_dir / "csi300_three_model_cumulative_return_data.json"
    summary_path = args.out_dir / "csi300_three_model_cumulative_return_summary.csv"
    daily_path = args.out_dir / "csi300_three_model_daily_returns.csv"
    html_path = args.out_dir / "csi300_three_model_cumulative_return_curve.html"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(daily_path, index=False, encoding="utf-8-sig")
    html_path.write_text(render_html(data), encoding="utf-8")
    print(json.dumps({"html": str(html_path), "data": str(data_path), "summary": str(summary_path), "daily": str(daily_path), "models": {k: {x: v[x] for x in ("final_cumulative_return", "annualized_return_252", "max_drawdown")} for k, v in series.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



