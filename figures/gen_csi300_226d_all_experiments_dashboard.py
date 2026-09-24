from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "train3" / "csi300_csi800_aligned_seed100" / "experiments"
OUT = ROOT / "results" / "dashboards"
OUT_HTML = OUT / "all_experiments_dashboard_226d.html"
OUT_JSON = OUT / "all_experiments_dashboard_226d_data.json"
START, END = "2025-07-01", "2026-06-05"


def finite(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def load_sources():
    b = pd.read_csv(RUN / "B_linear_head" / "predictions.csv")
    keys = b[["as_of_date", "stock"]].drop_duplicates()
    if len(keys) != 66103:
        raise ValueError(f"Expected 66,103 fixed keys, got {len(keys):,}")

    raw_a = pd.concat(
        [pd.read_csv(p, usecols=["as_of_date", "stock", "pred_path_mean_return", "actual_path_mean_return"])
         for p in sorted((RUN / "A_zero_shot" / "predictions" / "by_date").glob("*.csv"))],
        ignore_index=True,
    )
    a = raw_a.merge(keys, on=["as_of_date", "stock"], how="inner", validate="one_to_one")
    a = a.rename(columns={"pred_path_mean_return": "pred_signal", "actual_path_mean_return": "actual_signal"})

    frames = {"A": a, "B": b}
    for key, folder in [("C", "C_prediction_adapter"), ("D", "D_ranking_adapter")]:
        frames[key] = pd.read_csv(RUN / folder / "predictions.csv")

    cont = pd.read_csv(RUN / "continual_ABC" / "predictions.csv")
    for g in "ABC":
        frames[f"Continual {g}"] = cont.loc[cont["group"].eq(g)].copy()
    frames["Adapter + Continual"] = pd.read_csv(RUN / "adapter_continual" / "predictions.csv")

    for name, frame in frames.items():
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")
        frame["stock"] = frame["stock"].astype(str)
        frame = frame.merge(keys, on=["as_of_date", "stock"], how="inner", validate="one_to_one")
        if len(frame) != 66103 or frame["as_of_date"].nunique() != 226:
            raise ValueError(f"{name}: expected 66,103 rows / 226 dates, got {len(frame):,} / {frame['as_of_date'].nunique()}")
        if frame["as_of_date"].min() != START or frame["as_of_date"].max() != END:
            raise ValueError(f"{name}: wrong date boundary")
        frames[name] = frame.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    return frames, len(raw_a)


def nw_t(values, lag=9):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(lrv, 0.0) / n)
    return finite(x.mean() / se) if se > 0 else None


def model_metrics(df):
    p = pd.to_numeric(df["pred_signal"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df["actual_signal"], errors="coerce").to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    err = p - a
    pred_pos, actual_pos = p > 0, a > 0
    tp = int(np.sum(pred_pos & actual_pos)); tn = int(np.sum(~pred_pos & ~actual_pos))
    fp = int(np.sum(pred_pos & ~actual_pos)); fn = int(np.sum(~pred_pos & actual_pos))
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    daily = []
    for date, g in df.groupby("as_of_date", sort=True):
        x = pd.to_numeric(g["pred_signal"], errors="coerce")
        y = pd.to_numeric(g["actual_signal"], errors="coerce")
        m = x.notna() & y.notna()
        x, y = x[m], y[m]
        order = np.argsort(x.to_numpy(), kind="mergesort")
        n = len(x); q = max(1, n // 5); k = min(50, n // 2)
        yy = y.to_numpy()[order]
        ic = x.corr(y, method="pearson")
        ric = x.corr(y, method="spearman")
        top = float(np.mean(yy[-k:])); bottom = float(np.mean(yy[:k])); universe = float(y.mean())
        daily.append({
            "date": date, "n": n, "ic": finite(ic), "rankic": finite(ric),
            "q5_q1": finite(np.mean(yy[-q:]) - np.mean(yy[:q])),
            "top50_raw": top, "top50_excess": top - universe,
            "top50_bottom50": top - bottom,
        })
    dd = pd.DataFrame(daily)
    result = {
        "n": int(len(p)), "dates": int(len(dd)),
        "mae": finite(np.mean(np.abs(err))),
        "rmse": finite(np.sqrt(np.mean(err ** 2))),
        "mase": finite(np.sum(np.abs(err)) / np.sum(np.abs(a))) if np.sum(np.abs(a)) else None,
        "oos_r2": finite(1.0 - np.sum(err ** 2) / np.sum(a ** 2)) if np.sum(a ** 2) else None,
        "accuracy": finite((tp + tn) / len(p)),
        "balanced_accuracy": finite((recall + specificity) / 2),
        "mcc": finite((tp * tn - fp * fn) / den) if den else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "path_ic": finite(dd["ic"].mean()),
        "path_rankic": finite(dd["rankic"].mean()),
        "ic_positive_ratio": finite((dd["ic"] > 0).mean()),
        "rankic_positive_ratio": finite((dd["rankic"] > 0).mean()),
        "ic_nw_t": nw_t(dd["ic"], 9), "rankic_nw_t": nw_t(dd["rankic"], 9),
        "q5_q1": finite(dd["q5_q1"].mean()),
        "top50_raw": finite(dd["top50_raw"].mean()),
        "top50_excess": finite(dd["top50_excess"].mean()),
        "top50_bottom50": finite(dd["top50_bottom50"].mean()),
    }
    return result


def build_data():
    frames, raw_a_rows = load_sources()
    source_map = {
        "A": "train3/csi300_csi800_aligned_seed100/experiments/A_zero_shot/predictions/by_date/*.csv",
        "B": "train3/csi300_csi800_aligned_seed100/experiments/B_linear_head/predictions.csv",
        "C": "train3/csi300_csi800_aligned_seed100/experiments/C_prediction_adapter/predictions.csv",
        "D": "train3/csi300_csi800_aligned_seed100/experiments/D_ranking_adapter/predictions.csv",
        "Continual A": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions.csv[group=A]",
        "Continual B": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions.csv[group=B]",
        "Continual C": "train3/csi300_csi800_aligned_seed100/experiments/continual_ABC/predictions.csv[group=C]",
        "Adapter + Continual": "train3/csi300_csi800_aligned_seed100/experiments/adapter_continual/predictions.csv",
    }
    labels = {
        "A": "Zero-shot 原始信号", "B": "Linear Head", "C": "Prediction Adapter", "D": "Ranking Adapter",
        "Continual A": "固定初始模型", "Continual B": "每月更新，不设验证门", "Continual C": "每月更新 + Validation RankIC Gate",
        "Adapter + Continual": "Prediction Adapter + Continual",
    }
    families = {k: ("ABCD 静态对照" if k in "ABCD" and len(k) == 1 else ("Continual A/B/C" if k.startswith("Continual") else "Adapter + Continual")) for k in frames}
    models = []
    for name, frame in frames.items():
        models.append({
            "id": name, "label": labels[name], "family": families[name], "source": source_map[name],
            "window": {"start": frame.as_of_date.min(), "end": frame.as_of_date.max(), "dates": int(frame.as_of_date.nunique()), "rows": int(len(frame))},
            "metrics": model_metrics(frame),
        })
    return {
        "schema_version": 1,
        "title": "CSI300 226日全量实验看板",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "仅保留具有完整 2025-07-01 至 2026-06-05、226 个交易日预测的 CSI300 正式实验；所有模型显式对齐同一 66,103 个股票—日期键。",
        "common_parameters": {"dataset": "CSI300", "train_end": "2024-12-15", "validation": ["2025-01-02", "2025-06-14"], "test": [START, END], "lookback": 90, "pred_len": 10, "seed": 100, "dates": 226, "rows": 66103, "alignment_key": ["as_of_date", "stock"], "signal": "10日预测路径均值收益", "label": "10日真实路径均值收益"},
        "old_dashboard_audit": {
            "old_file": r"C:\Users\huawei_wzn\Desktop\Desktop\时间序列预测\股票预测\results\dashboards\all_experiments_dashboard.html",
            "recent50_source_window": ["2026-05-27", "2026-08-05"], "recent50_dates": 50, "recent50_rows": 14842,
            "affected_sections": [
                "CSI300 ABCD A/B/C/D：点预测、方向、IC/RankIC、分位数组合及旧组合指标",
                "CSI300 Continual A/B/C：方向、IC/RankIC、组合指标",
                "CSI300 Adapter + Continual：点预测、方向、IC/RankIC、组合指标",
                "S&P500/Yahoo S&P500 的对应 ABCD、Continual、Adapter + Continual 实际为 49 个有效日期，不属于226日CSI300面板",
                "历史 recent50/recent10、SWVMD、LSTM、Adapter/RL/LoRA 页面不是统一226日实验",
            ],
            "replacement": "本看板从本地完整预测产物重新计算，不沿用旧看板 recent50 汇总。",
            "raw_zero_shot_rows_before_alignment": raw_a_rows,
            "zero_shot_rows_after_fixed_key_alignment": 66103,
        },
        "limitations": [
            "使用当前 CSI300 成分股回填历史区间，存在幸存者偏差与当前成分股前视偏差。",
            "横截面组合列使用同一日的10日Path真实标签，是预测排序代理，不是逐日可执行资金曲线；不与AER混称。",
            "Continual A/B/C 源文件只保存Path聚合信号，因此本看板不伪造其Endpoint或逐horizon指标。",
        ],
        "models": models,
    }


CSS = r'''
:root{--ink:#172033;--muted:#657087;--line:#dce3ee;--bg:#f4f7fb;--card:#fff;--blue:#3157d5;--blue2:#eaf0ff;--green:#067a58;--greenbg:#eafaf4;--red:#b42318;--redbg:#fff0ee;--amber:#9a5b00;--amberbg:#fff8e6}*{box-sizing:border-box}body{margin:0;font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:linear-gradient(150deg,#eef3ff 0,#f8fafc 38%,#f4f7fb 100%);color:var(--ink)}.wrap{max-width:1500px;margin:auto;padding:28px}.hero{background:linear-gradient(125deg,#172554,#2947a9 58%,#4d6be3);color:#fff;border-radius:20px;padding:28px 32px;box-shadow:0 18px 50px #1f3b8d2b}.hero h1{margin:0 0 8px;font-size:29px}.hero p{margin:4px 0;color:#dfe7ff;line-height:1.65}.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:15px}.chip{background:#ffffff1c;border:1px solid #ffffff35;border-radius:999px;padding:7px 11px;font-size:12px}.notice{margin:18px 0;padding:15px 17px;border:1px solid #f1cf78;background:var(--amberbg);color:#754700;border-radius:13px;line-height:1.65}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tab{border:1px solid var(--line);background:#fff;padding:10px 16px;border-radius:10px;font-weight:700;color:#526078;cursor:pointer}.tab.active{background:var(--blue);border-color:var(--blue);color:#fff}.page{display:none}.page.active{display:block}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 8px 30px #25345d0b}.card h2,.card h3{margin:0 0 10px}.sub{color:var(--muted);font-size:13px;line-height:1.6}.params{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:9px}.param{padding:10px 12px;background:#f8faff;border:1px solid #e3e9f4;border-radius:10px}.param b{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}.param span{font-weight:700}.tablebox{overflow:auto;border:1px solid var(--line);border-radius:12px}table{border-collapse:separate;border-spacing:0;width:100%;min-width:1160px;font-size:12px}th,td{padding:10px 9px;border-bottom:1px solid #e8edf4;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#eef3ff;color:#38445c;z-index:1}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:inherit}tbody tr:nth-child(even){background:#fafcff}tbody tr:hover{background:#f0f5ff}td:first-child{font-weight:750}.best{background:var(--greenbg)!important;color:var(--green);font-weight:800}.worst{background:var(--redbg)!important;color:var(--red)}.ok{color:var(--green);font-weight:750}.audit-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}.audit-item{border:1px solid #f0d58f;background:#fffbef;border-radius:11px;padding:12px;line-height:1.55}.metric-help{border:0;background:#dce6ff;color:#2947a9;border-radius:50%;width:18px;height:18px;font-weight:800;cursor:pointer}.foot{color:var(--muted);font-size:12px;line-height:1.7;margin-top:18px}.modal{display:none;position:fixed;inset:0;background:#10182899;z-index:20;padding:28px}.modal.open{display:grid;place-items:center}.modalbox{background:#fff;border-radius:16px;max-width:650px;width:100%;padding:22px;box-shadow:0 24px 80px #0005}.modalbox button{float:right;border:0;background:#eef2f8;border-radius:8px;padding:6px 9px;cursor:pointer}.source{font-family:Consolas,monospace;font-size:11px;color:#566178;word-break:break-all}@media(max-width:720px){.wrap{padding:14px}.hero{padding:20px}.hero h1{font-size:23px}}
'''

JS = r'''
const DATA=__DATA__;
const groups={abcd:['A','B','C','D'],continual:['Continual A','Continual B','Continual C'],adapter:['Adapter + Continual']};
const defs={mae:['MAE','预测Path信号与真实Path标签的平均绝对误差；越低越好。'],rmse:['RMSE','均方根误差，对大误差更敏感；越低越好。'],mase:['MASE','MAE除以零收益持久性基线的平均绝对误差；小于1优于该基线。'],oos_r2:['OOS R²','1−SSE/Σy²，以零收益为基准；越高越好。'],accuracy:['方向准确率','sign(pred)==sign(actual) 的样本比例，零收益视为负类。'],balanced_accuracy:['平衡准确率','正类召回率与负类召回率的平均，缓解类别不均衡。'],mcc:['MCC','Matthews相关系数，范围[-1,1]；越高越好。'],path_ic:['Path IC','每日横截面Pearson相关系数的226日均值；越高越好。'],path_rankic:['Path RankIC','每日横截面Spearman相关系数的226日均值；越高越好。'],q5_q1:['Q5−Q1','每日预测最高五分位与最低五分位的10日Path标签收益差。'],top50_excess:['Top50超额','每日预测Top50的10日Path标签均值减去当日股票池均值。'],top50_bottom50:['Top50−Bottom50','每日预测最高50与最低50股票的10日Path标签收益差。']};
const cols=[['mae','MAE','low','num'],['rmse','RMSE','low','num'],['mase','MASE','low','num'],['oos_r2','OOS R²','high','num'],['accuracy','方向Acc','high','pct'],['balanced_accuracy','Balanced Acc','high','pct'],['mcc','MCC','high','num'],['path_ic','IC','high','num'],['path_rankic','RankIC','high','num'],['q5_q1','Q5−Q1','high','pct'],['top50_excess','Top50超额','high','pct'],['top50_bottom50','Top50−Bottom50','high','pct']];
function fmt(v,t){if(v===null||v===undefined)return '—';if(t==='pct')return (v*100).toFixed(2)+'%';return Number(v).toFixed(4)}
function render(which){const ids=groups[which], rows=DATA.models.filter(x=>ids.includes(x.id));let h='<table><thead><tr><th>实验</th><th>样本/日期</th>'+cols.map(c=>`<th>${c[1]} <button class="metric-help" data-k="${c[0]}">i</button></th>`).join('')+'</tr></thead><tbody>';for(const r of rows){h+=`<tr><td>${r.id}<div class="sub">${r.label}</div></td><td><span class="ok">${r.window.rows.toLocaleString()} / ${r.window.dates}</span></td>`+cols.map(c=>`<td data-col="${c[0]}" data-val="${r.metrics[c[0]]}">${fmt(r.metrics[c[0]],c[3])}</td>`).join('')+'</tr>'}h+='</tbody></table>';const box=document.querySelector(`#${which} .tablebox`);box.innerHTML=h;for(const c of cols){const cells=[...box.querySelectorAll(`[data-col="${c[0]}"]`)];const vals=cells.map(x=>+x.dataset.val).filter(Number.isFinite);if(vals.length>1){const best=c[2]==='low'?Math.min(...vals):Math.max(...vals),worst=c[2]==='low'?Math.max(...vals):Math.min(...vals);cells.forEach(x=>{const v=+x.dataset.val;if(v===best)x.classList.add('best');if(v===worst)x.classList.add('worst')})}}box.querySelectorAll('.metric-help').forEach(b=>b.onclick=()=>openHelp(b.dataset.k));document.querySelector(`#${which} .sources`).innerHTML=rows.map(r=>`<div><b>${r.id}</b>：<span class="source">${r.source}</span></div>`).join('')}
function openHelp(k){const d=defs[k]||[k,''];document.querySelector('#modalTitle').textContent=d[0];document.querySelector('#modalText').textContent=d[1];document.querySelector('#modal').classList.add('open')}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.page').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelector('#'+b.dataset.page).classList.add('active')});document.querySelector('#closeModal').onclick=()=>document.querySelector('#modal').classList.remove('open');document.querySelector('#modal').onclick=e=>{if(e.target.id==='modal')e.currentTarget.classList.remove('open')};Object.keys(groups).forEach(render);
'''


def html(data):
    p = data["common_parameters"]
    params = [("数据集", p["dataset"]), ("训练截止", p["train_end"]), ("验证区间", " → ".join(p["validation"])), ("测试区间", " → ".join(p["test"])), ("Lookback / Horizon", f"{p['lookback']} / {p['pred_len']}"), ("Seed", p["seed"]), ("统一样本", f"{p['rows']:,}"), ("交易日", p["dates"]), ("对齐键", " + ".join(p["alignment_key"])), ("预测信号", p["signal"])]
    param_html = ''.join(f'<div class="param"><b>{k}</b><span>{v}</span></div>' for k,v in params)
    audit_html = ''.join(f'<div class="audit-item">{x}</div>' for x in data['old_dashboard_audit']['affected_sections'])
    pages = [('abcd','页面一：ABCD 静态对照','A=Zero-shot，B=Linear Head，C=Prediction Adapter，D=Ranking Adapter。'),('continual','页面二：Continual A/B/C','三种持续学习策略；仅使用源文件保存的Path聚合信号。'),('adapter','页面三：Adapter + Continual','Prediction Adapter 与持续学习结合的正式完整窗口结果。')]
    page_html=''.join(f'<section id="{i}" class="page {"active" if i=="abcd" else ""}"><div class="card"><h2>{t}</h2><p class="sub">{s}</p><div class="tablebox"></div></div><div class="card"><h3>逐模型数据来源</h3><div class="sources sub"></div></div></section>' for i,t,s in pages)
    js=JS.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{data['title']}</title><style>{CSS}</style></head><body><div class="wrap"><header class="hero"><h1>{data['title']}</h1><p>{data['scope']}</p><div class="chips"><span class="chip">226 个交易日</span><span class="chip">66,103 统一样本</span><span class="chip">8 个正式实验</span><span class="chip">离线单文件</span></div></header><div class="notice"><b>旧看板审计结论：</b>旧 CSI300 ABCD、Continual A/B/C、Adapter + Continual 的相关指标来自 50 日 / 14,842 样本的 recent50 cohort，而页面公共标题误写成226日。本页已全部替换为完整226日预测，并严格对齐同一组股票—日期键。</div><div class="card"><h3>统一实验参数</h3><div class="params">{param_html}</div></div><div class="tabs"><button class="tab active" data-page="abcd">ABCD 静态对照</button><button class="tab" data-page="continual">Continual A/B/C</button><button class="tab" data-page="adapter">Adapter + Continual</button></div>{page_html}<div class="card"><h3>旧看板中不是226日的数据</h3><div class="audit-grid">{audit_html}</div><p class="sub">旧 recent50 实际窗口：2026-05-27 → 2026-08-05，共50日、14,842样本；其与目标测试窗口仅重叠8日。旧数据没有被混入本看板。</p></div><div class="card"><h3>口径与限制</h3><ul class="sub">{''.join(f'<li>{x}</li>' for x in data['limitations'])}</ul></div><div class="foot">生成时间：{data['generated_at']}<br>外部 JSON 与 HTML 内嵌数据内容一致，可离线 file:// 打开。</div></div><div id="modal" class="modal"><div class="modalbox"><button id="closeModal">关闭</button><h3 id="modalTitle"></h3><p id="modalText" class="sub"></p></div></div><script>{js}</script></body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    OUT_JSON.write_text(text, encoding="utf-8")
    OUT_HTML.write_text(html(data), encoding="utf-8")
    print(json.dumps({"html": str(OUT_HTML), "json": str(OUT_JSON), "models": len(data["models"]), "rows": data["common_parameters"]["rows"], "dates": data["common_parameters"]["dates"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
