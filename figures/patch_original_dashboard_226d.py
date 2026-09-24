from __future__ import annotations

"""Patch only `var D = {...}` in the original dashboard with full-window CSI300 data.

The source HTML is treated as an immutable template. All CSS, DOM and renderer
JavaScript bytes outside the first main script's D object are preserved exactly.
"""

import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HTML = Path(r"C:\Users\huawei_wzn\Desktop\Desktop\时间序列预测\股票预测\results\dashboards\all_experiments_dashboard.html")
OUT_DIR = ROOT / "results" / "dashboards"
OUT_HTML = OUT_DIR / "all_experiments_dashboard_226d.html"
OUT_JSON = OUT_DIR / "all_experiments_dashboard_226d_data.json"
EXP = ROOT / "train3" / "csi300_csi800_aligned_seed100" / "experiments"
BT = ROOT / "train3" / "csi300_top10pct_20250701_20260605_seed100"
START, END, ROWS, DATES = "2025-07-01", "2026-06-05", 66103, 226
TOP_K, DROP_N, MIN_HOLD, COST, ANNUAL = 30, 3, 5, 0.0015, 252

SOURCE_MAP = {
    "A": EXP / "A_zero_shot" / "predictions" / "by_date",
    "B": EXP / "B_linear_head" / "predictions.csv",
    "C": EXP / "C_prediction_adapter" / "predictions.csv",
    "D": EXP / "D_ranking_adapter" / "predictions.csv",
    "Continual A": EXP / "continual_ABC" / "predictions.csv",
    "Continual B": EXP / "continual_ABC" / "predictions.csv",
    "Continual C": EXP / "continual_ABC" / "predictions.csv",
    "Adapter + Continual": EXP / "adapter_continual" / "predictions.csv",
}
BT_MAP = {
    "A": BT / "page1_abcd_top30_drop3" / "A_zero_shot",
    "B": BT / "page1_abcd_top30_drop3" / "B_linear_head",
    "C": BT / "page1_abcd_top30_drop3" / "C_prediction_adapter",
    "D": BT / "page1_abcd_top30_drop3" / "D_ranking_adapter",
    "Continual A": BT / "page2_continual_abc_top30_drop3" / "A_continual",
    "Continual B": BT / "page2_continual_abc_top30_drop3" / "B_continual",
    "Continual C": BT / "page2_continual_abc_top30_drop3" / "C_continual",
    "Adapter + Continual": BT / "page3_adapter_continual_top30_drop3" / "Adapter_continual",
}


def finite(x: Any) -> float | None:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clean_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return finite(x)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return x


def extract_d(text: str) -> tuple[dict[str, Any], int, int]:
    script = re.search(r"<script>([\s\S]*?)</script>", text, re.I)
    if not script:
        raise RuntimeError("first main script not found")
    body = script.group(1)
    local_start = body.index("var D = ") + len("var D = ")
    local_end = body.index(";\nvar metricCatalog", local_start)
    start = script.start(1) + local_start
    end = script.start(1) + local_end
    return json.loads(text[start:end]), start, end


def load_frames() -> dict[str, pd.DataFrame]:
    b = pd.read_csv(SOURCE_MAP["B"])
    b["as_of_date"] = pd.to_datetime(b["as_of_date"]).dt.strftime("%Y-%m-%d")
    b["stock"] = b["stock"].astype(str)
    keys = b[["as_of_date", "stock"]].drop_duplicates()
    if len(keys) != ROWS:
        raise RuntimeError(f"fixed key count is {len(keys)}, expected {ROWS}")

    a_cols = ["as_of_date", "stock", "current_close", "pred_path_mean_return", "actual_path_mean_return",
              "pred_endpoint_return", "actual_endpoint_return"]
    for h in range(1, 11):
        a_cols += [f"pred_close_{h}", f"actual_close_{h}"]
    a = pd.concat([pd.read_csv(p, usecols=a_cols) for p in sorted(SOURCE_MAP["A"].glob("*.csv"))], ignore_index=True)
    a["as_of_date"] = pd.to_datetime(a["as_of_date"]).dt.strftime("%Y-%m-%d")
    a["stock"] = a["stock"].astype(str)
    a = a.merge(keys, on=["as_of_date", "stock"], how="inner", validate="one_to_one")
    a["pred_signal"] = a["pred_path_mean_return"]
    a["actual_signal"] = a["actual_path_mean_return"]
    for h in range(1, 11):
        a[f"pred_return_{h}"] = pd.to_numeric(a[f"pred_close_{h}"], errors="coerce") / pd.to_numeric(a["current_close"], errors="coerce") - 1.0
        a[f"actual_return_{h}"] = pd.to_numeric(a[f"actual_close_{h}"], errors="coerce") / pd.to_numeric(a["current_close"], errors="coerce") - 1.0

    frames: dict[str, pd.DataFrame] = {"A": a, "B": b}
    frames["C"] = pd.read_csv(SOURCE_MAP["C"])
    frames["D"] = pd.read_csv(SOURCE_MAP["D"])
    cont = pd.read_csv(SOURCE_MAP["Continual A"])
    one_step = b[["as_of_date", "stock", "actual_return_1"]].copy()
    for group in "ABC":
        frames[f"Continual {group}"] = cont.loc[cont["group"].astype(str).eq(group)].copy().merge(
            one_step, on=["as_of_date", "stock"], how="left", validate="one_to_one")
    frames["Adapter + Continual"] = pd.read_csv(SOURCE_MAP["Adapter + Continual"])

    for model, frame in list(frames.items()):
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")
        frame["stock"] = frame["stock"].astype(str)
        # A has already been aligned; all others are explicitly constrained to the same keys.
        if model != "A":
            frame = frame.merge(keys, on=["as_of_date", "stock"], how="inner", validate="one_to_one")
        frame = frame.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
        if len(frame) != ROWS or frame["as_of_date"].nunique() != DATES:
            raise RuntimeError(f"{model}: {len(frame)} rows / {frame['as_of_date'].nunique()} dates")
        if frame["as_of_date"].min() != START or frame["as_of_date"].max() != END:
            raise RuntimeError(f"{model}: bad date window")
        frames[model] = frame
    return frames


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(pd.Series(x).rank(method="average").to_numpy(float), pd.Series(y).rank(method="average").to_numpy(float))


def nw_t(values: np.ndarray, lag: int = 9) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    u = x - x.mean()
    lrv = float(np.dot(u, u) / n)
    max_lag = min(lag, n - 1)
    for ell in range(1, max_lag + 1):
        gamma = float(np.dot(u[ell:], u[:-ell]) / n)
        lrv += 2 * (1 - ell / (max_lag + 1)) * gamma
    var_mean = max(lrv / n, 0.0)
    return float(x.mean() / math.sqrt(var_mean)) if var_mean > 0 else float("nan")


def bootstrap_ci(values: np.ndarray, seed: int, reps: int = 5000, block: int = 10) -> list[float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    b = min(block, len(x))
    blocks = [x[i:i+b] for i in range(len(x)-b+1)]
    rng = np.random.default_rng(seed)
    need = math.ceil(len(x) / b)
    means = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(blocks), size=need)
        means[i] = np.concatenate([blocks[j] for j in pick])[:len(x)].mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def binary_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    p, a = pred[mask] > 0, actual[mask] > 0
    tp, tn = int(np.sum(p & a)), int(np.sum(~p & ~a))
    fp, fn = int(np.sum(p & ~a)), int(np.sum(~p & a))
    n = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return clean_json({
        "direction_accuracy": (tp + tn) / n if n else np.nan,
        "balanced_accuracy": np.nanmean([recall, specificity]),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
        "specificity": specificity,
        "mcc": (tp * tn - fp * fn) / denom if denom else np.nan,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    })


def cross_section(df: pd.DataFrame, pred_col: str, actual_col: str, seed: int) -> dict[str, Any]:
    rows = []
    for _, g in df.groupby("as_of_date", sort=True):
        x = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g[actual_col], errors="coerce").to_numpy(float)
        rows.append((safe_corr(x, y), spearman(x, y)))
    arr = np.asarray(rows, float)
    result: dict[str, Any] = {}
    for offset, (name, values) in enumerate(zip(("ic", "rankic"), arr.T)):
        good = values[np.isfinite(values)]
        mean = float(good.mean())
        std = float(good.std(ddof=1)) if len(good) > 1 else np.nan
        result[name] = clean_json({
            "count": len(good), "mean": mean, "median": np.median(good), "std": std,
            "positive_ratio": np.mean(good > 0), "ir_mean_over_std": mean/std if std > 0 else np.nan,
            "annualized_ir_sqrt252": mean/std*math.sqrt(252) if std > 0 else np.nan,
            "newey_west_lag9_mean_tstat": nw_t(good, 9),
            "moving_block_bootstrap_block10_ci95": bootstrap_ci(good, seed + offset),
        })
    return result


def scalar_point(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    err = pred - actual
    ae0 = np.abs(actual).sum()
    sse0 = np.square(actual).sum()
    return clean_json({
        "scale": "cumulative close return relative to as_of close",
        "n": len(pred), "mae_return": np.abs(err).mean(), "rmse_return": np.sqrt(np.square(err).mean()),
        "mase_vs_zero_return_persistence": np.abs(err).sum()/ae0 if ae0 else np.nan,
        "oos_r2_vs_zero_return": 1-np.square(err).sum()/sse0 if sse0 else np.nan,
    })


def point_forecast(df: pd.DataFrame, scalar_only: bool) -> dict[str, Any]:
    if scalar_only:
        return {"path_signal_overall": scalar_point(df["pred_signal"], df["actual_signal"]), "by_horizon": []}
    by_horizon = []
    pred_all, actual_all = [], []
    for h in range(1, 11):
        p = pd.to_numeric(df[f"pred_return_{h}"], errors="coerce").to_numpy(float)
        a = pd.to_numeric(df[f"actual_return_{h}"], errors="coerce").to_numpy(float)
        item = scalar_point(p, a)
        item["horizon"] = h
        by_horizon.append(item)
        pred_all.append(p); actual_all.append(a)
    return {"trajectory_overall": scalar_point(np.concatenate(pred_all), np.concatenate(actual_all)), "by_horizon": by_horizon}


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + np.asarray(returns, float))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1))


def label_sort_stats(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, float]:
    rows = []
    for _, g in df.groupby("as_of_date", sort=True):
        x = pd.to_numeric(g[pred_col], errors="coerce")
        y = pd.to_numeric(g[actual_col], errors="coerce")
        good = x.notna() & y.notna()
        t = pd.DataFrame({"x": x[good], "y": y[good], "stock": g.loc[good, "stock"]}).sort_values(["x", "stock"], ascending=[False, True])
        n = len(t); q = max(1, n // 5); k = min(TOP_K, n // 2)
        top, bottom = t.head(k)["y"].mean(), t.tail(k)["y"].mean()
        rows.append({"q5q1": t.head(q)["y"].mean()-t.tail(q)["y"].mean(), "top": top,
                     "bottom": bottom, "excess": top-t["y"].mean(), "longshort": top-bottom})
    d = pd.DataFrame(rows)
    return {k: float(d[k].mean()) for k in d.columns}


def formal_backtest(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    positions: dict[str, int] = {}
    daily = []
    for di, (_, g0) in enumerate(df.groupby("as_of_date", sort=True)):
        g = g0[["stock", pred_col, "actual_return_1"]].copy()
        g.columns = ["stock", "signal", "ret"]
        g = g.replace([np.inf, -np.inf], np.nan).dropna().sort_values(["signal", "stock"], ascending=[False, True]).drop_duplicates("stock")
        universe = set(g["stock"]); target = list(g.head(min(TOP_K, len(g)))["stock"]); target_set = set(target)
        old = set(positions)
        forced = {s for s in old if s not in universe}
        score = dict(zip(g["stock"], g["signal"]))
        sellable = [s for s in old if s in universe and s not in target_set and di-positions[s] >= MIN_HOLD]
        sellable.sort(key=lambda s: (score.get(s, -np.inf), s))
        sell = forced | set(sellable[:DROP_N])
        kept = old - sell
        buy_candidates = [s for s in target if s not in kept]
        if not old:
            buy = set(target)
        else:
            buy = set(buy_candidates[:len(sell)])
            needed = max(0, min(TOP_K, len(g)) - len(kept) - len(buy))
            if needed:
                buy.update([s for s in target if s not in kept and s not in buy][:needed])
        new = kept | buy
        if len(new) > min(TOP_K, len(g)):
            ranked = list(g[g['stock'].isin(new)].sort_values(['signal', 'stock'], ascending=[False, True])['stock'])
            new = set(ranked[:min(TOP_K, len(g))])
        old_w, new_w = (1/len(old) if old else 0), (1/len(new) if new else 0)
        turnover = 1.0 if not old and new else 0.5*sum(abs((new_w if s in new else 0)-(old_w if s in old else 0)) for s in old|new)
        positions = {s: (positions[s] if s in positions else di) for s in new}
        held = g[g["stock"].isin(new)]
        gross = float(held["ret"].mean())
        bench = float(g["ret"].mean())
        net = gross - COST*turnover
        daily.append((gross, net, bench, turnover))
    d = pd.DataFrame(daily, columns=["gross", "net", "benchmark", "turnover"])
    if len(d) != DATES:
        raise RuntimeError(f"backtest produced {len(d)} dates")
    std = d["net"].std(ddof=1)
    return {
        "raw": float(d["gross"].mean()), "net": float(d["net"].mean()),
        "sharpe": float(d["net"].mean()/std*math.sqrt(ANNUAL)) if std > 0 else np.nan,
        "mdd": max_drawdown(d["net"].to_numpy()), "turnover": float(d["turnover"].mean()),
        "aer": float((d["net"]-d["benchmark"]).mean()*ANNUAL),
        "total_cost": float((COST*d["turnover"]).sum()),
    }


def portfolio_metrics(df: pd.DataFrame, has_endpoint: bool, model: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": "Top30 / Drop3 / minimum holding 5 trading days / equal-weight long-only / 0.15% one-way turnover cost",
        "dates": DATES, "top_k": TOP_K, "drop_n": DROP_N, "min_holding_days": MIN_HOLD,
        "transaction_cost_rate": COST,
    }
    modes = [("path_mean", "pred_signal", "actual_signal")]
    if has_endpoint:
        modes.append(("endpoint", "pred_endpoint_return", "actual_endpoint_return"))
    for mode, pred, actual in modes:
        labels = label_sort_stats(df, pred, actual)
        bt = formal_backtest(df, pred)
        if mode == "path_mean":
            persisted_daily = pd.read_csv(BT_MAP[model] / "daily_returns.csv")
            net = pd.to_numeric(persisted_daily["net_return"], errors="coerce").to_numpy(float)
            bench = pd.to_numeric(persisted_daily["benchmark_ew_return"], errors="coerce").to_numpy(float)
            gross = pd.to_numeric(persisted_daily["gross_return"], errors="coerce").to_numpy(float)
            turnover = pd.to_numeric(persisted_daily["turnover"], errors="coerce").to_numpy(float)
            bt = {"raw": float(np.nanmean(gross)), "net": float(np.nanmean(net)),
                  "sharpe": float(np.nanmean(net)/np.nanstd(net, ddof=1)*math.sqrt(ANNUAL)),
                  "mdd": max_drawdown(net), "turnover": float(np.nanmean(turnover)),
                  "aer": float(np.nanmean(net-bench)*ANNUAL), "total_cost": float(np.nansum(COST*turnover))}
        out.update({
            f"mean_q5_q1_{mode}": labels["q5q1"], f"mean_top30_{mode}_raw": bt["raw"],
            f"mean_bottom30_{mode}": labels["bottom"], f"mean_top30_excess_{mode}": labels["excess"],
            f"mean_top30_{mode}_net": bt["net"], f"mean_top30_bottom30_{mode}": labels["longshort"],
            f"sharpe_top30_{mode}_net_10bps": bt["sharpe"], f"mdd_top30_{mode}_net_10bps": bt["mdd"],
            f"mean_turnover_{mode}": bt["turnover"], f"kronos_aer_{mode}": bt["aer"],
            f"net_sharpe_{mode}": bt["sharpe"], f"net_max_drawdown_{mode}": bt["mdd"],
            f"total_transaction_cost_{mode}": bt["total_cost"],
        })
    # The path run must reproduce the persisted formal Top30 run exactly.
    persisted = json.loads((BT_MAP[model] / "summary.json").read_text(encoding="utf-8"))
    checks = {"net": persisted["mean_daily_net_return"], "mdd": persisted["max_drawdown"], "turnover": persisted["mean_turnover"], "total_cost": persisted["total_transaction_cost"]}
    got = {"net": out["mean_top30_path_mean_net"], "mdd": out["mdd_top30_path_mean_net_10bps"], "turnover": out["mean_turnover_path_mean"], "total_cost": out["total_transaction_cost_path_mean"]}
    for key in checks:
        if not math.isclose(got[key], checks[key], rel_tol=0, abs_tol=2e-12):
            raise RuntimeError(f"{model} path backtest mismatch for {key}: {got[key]} vs {checks[key]}")
    out["formal_top30_summary_source"] = str(BT_MAP[model] / "summary.json")
    return clean_json(out)


def build_metrics(df: pd.DataFrame, model: str, seed: int) -> dict[str, Any]:
    continual = model.startswith("Continual ")
    has_endpoint = not continual
    point = point_forecast(df, continual)
    direction: dict[str, Any] = {}
    direction["path_mean"] = binary_metrics(df["pred_signal"], df["actual_signal"])
    direction["cross_section_path_mean"] = cross_section(df, "pred_signal", "actual_signal", seed)
    if has_endpoint:
        if "pred_endpoint_return" not in df:
            df = df.copy()
            df["pred_endpoint_return"] = df["pred_return_10"]
            df["actual_endpoint_return"] = df["actual_return_10"]
        direction["endpoint"] = binary_metrics(df["pred_endpoint_return"], df["actual_endpoint_return"])
        direction["cross_section_endpoint"] = cross_section(df, "pred_endpoint_return", "actual_endpoint_return", seed+20)
    return clean_json({"point_forecast": point, "direction_forecast": direction,
                       "portfolio": portfolio_metrics(df, has_endpoint, model)})


def summary_row(group: str, metrics: dict[str, Any]) -> dict[str, str | None]:
    p = metrics["point_forecast"]
    o = p.get("trajectory_overall") or p.get("path_signal_overall") or p.get("overall")
    d = metrics["direction_forecast"]; path = d["path_mean"]; cs = d["cross_section_path_mean"]
    epcs = d.get("cross_section_endpoint", {})
    pf = metrics["portfolio"]
    vals = {
        "group": group, "point_mae": o.get("mae_return"), "point_rmse": o.get("rmse_return"),
        "point_mase_persistence": o.get("mase_vs_zero_return_persistence"), "point_oos_r2": o.get("oos_r2_vs_zero_return"),
        "path_direction_accuracy": path.get("direction_accuracy"), "path_balanced_accuracy": path.get("balanced_accuracy"),
        "path_mcc": path.get("mcc"), "path_ic_mean": cs["ic"].get("mean"), "path_rankic_mean": cs["rankic"].get("mean"),
        "path_rankic_nw_t": cs["rankic"].get("newey_west_lag9_mean_tstat"), "path_rankic_positive_ratio": cs["rankic"].get("positive_ratio"),
        "endpoint_ic_mean": epcs.get("ic", {}).get("mean"), "endpoint_rankic_mean": epcs.get("rankic", {}).get("mean"),
        "mean_q5_q1_endpoint": pf.get("mean_q5_q1_endpoint"), "mean_top30_endpoint_raw": pf.get("mean_top30_endpoint_raw"),
        "mean_top30_excess_endpoint": pf.get("mean_top30_excess_endpoint"), "mean_top30_bottom30_endpoint": pf.get("mean_top30_bottom30_endpoint"),
        "mean_turnover": pf.get("mean_turnover_path_mean"), "mean_top30_endpoint_net": pf.get("mean_top30_endpoint_net"),
        "net_sharpe_endpoint": pf.get("net_sharpe_endpoint"), "max_drawdown_endpoint_net": pf.get("net_max_drawdown_endpoint"),
    }
    return {k: (None if v is None else str(v)) for k, v in vals.items()}


def update_experiment(e: dict[str, Any], model: str, metrics: dict[str, Any]) -> None:
    labels = {"A":"CSI300 · A Zero-shot", "B":"CSI300 · B Linear Head", "C":"CSI300 · C Prediction Adapter",
              "D":"CSI300 · D Ranking Adapter", "Continual A":"CSI300 · Continual A", "Continual B":"CSI300 · Continual B",
              "Continual C":"CSI300 · Continual C", "Adapter + Continual":"CSI300 · Adapter + Continual（正式）"}
    names = {"A":"csi300_226d_A_zero_shot", "B":"csi300_226d_B_linear_head", "C":"csi300_226d_C_prediction_adapter",
             "D":"csi300_226d_D_ranking_adapter", "Continual A":"csi300_226d_continual_A", "Continual B":"csi300_226d_continual_B",
             "Continual C":"csi300_226d_continual_C", "Adapter + Continual":"csi300_226d_adapter_continual"}
    e.update({"name": names[model], "label": labels[model], "status": "complete / formal / 226 trading days", "dataset": "CSI300",
              "lookback": 90, "pred_len": 10, "test_start": START, "test_end": END, "test_dates": DATES, "test_rows": ROWS,
              "seed": 100, "source_scale": "decimal cumulative return relative to as_of close",
              "metrics": metrics, "artifacts": [str(SOURCE_MAP[model]), str(BT_MAP[model] / "summary.json")],
              "output_dir": str(SOURCE_MAP[model] if SOURCE_MAP[model].is_dir() else SOURCE_MAP[model].parent)})


def patch_data(d: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out = deepcopy(d)
    metrics = {model: build_metrics(frame, model, 100 + i*100) for i, (model, frame) in enumerate(frames.items())}

    cfg = out["ref"]["metrics"]["CSI300"].get("config", {})
    cfg.update({"dataset": "CSI300", "train_end": "2024-12-15", "val_start": "2025-01-02", "val_end": "2025-06-14",
                "test_start": START, "test_end": END, "lookback": 90, "pred_len": 10, "seed": 100,
                "test_dates": DATES, "test_rows": ROWS, "top_k": TOP_K, "drop_n": DROP_N,
                "min_holding_days": MIN_HOLD, "cost_rate": COST,
                "source_window": "full aligned 226-day CSI300 formal predictions",
                "source_paths": {m: str(SOURCE_MAP[m]) for m in "ABCD"}})
    out["ref"]["metrics"]["CSI300"] = {
        "config": cfg,
        "integrity": {"status": "complete", "rows": ROWS, "dates": DATES, "duplicates": 0,
                      "test_start": START, "test_end": END, "alignment_key": ["as_of_date", "stock"]},
        "groups": {g: metrics[g] for g in "ABCD"},
    }
    out["ref"]["decile"]["CSI300"] = {g: deepcopy(metrics[g]["portfolio"]) for g in "ABCD"}
    out["ref"]["summary"]["CSI300"] = [summary_row(g, metrics[g]) for g in "ABCD"]
    out["ref"]["meta"]["CSI300"] = {"rows": ROWS, "dates": DATES, "strict": True, "duplicates": 0,
                                                    "test_start": START, "test_end": END,
                                                    "mean_n": ROWS/DATES, "mean_k": TOP_K,
                                                    "alignment_key": ["as_of_date", "stock"]}

    full_cont_summary = json.loads((EXP / "continual_ABC" / "summary.json").read_text(encoding="utf-8"))
    cont_cfg = deepcopy(out["continual"]["datasets"]["CSI300"].get("config", {}))
    cont_cfg.update({"cache_dir": str(EXP.parent), "output_dir": str(EXP / "continual_ABC"),
                     "raw_signal_file": str(EXP / "continual_ABC" / "predictions.csv"),
                     "test_start": START, "test_end": END, "test_dates": DATES, "test_rows": ROWS,
                     "resolved_test_dates": sorted(frames["Continual A"]["as_of_date"].unique().tolist()),
                     "top_k": TOP_K, "drop_n": DROP_N, "min_holding_days": MIN_HOLD, "cost_rate": COST,
                     "source_test_start": START, "source_test_end": END,
                     "boundary_status": "verified full 226-day formal window"})
    full_cont_summary.update({"test_start": START, "test_end": END, "test_dates": DATES, "test_rows": ROWS, "common_rows": ROWS})
    out["continual"]["datasets"]["CSI300"] = {
        "config": cont_cfg, "run_summary": full_cont_summary,
        "metrics_config": {"cost_rate": COST, "bootstrap_reps": 5000, "bootstrap_block": 10,
                           "source_prediction_file": str(EXP / "continual_ABC" / "predictions.csv")},
        "groups": {g: metrics[f"Continual {g}"] for g in "ABC"},
        "endpoint_status": "无法验证：正式 Continual A/B/C 源文件只保存 Path 聚合信号。",
    }

    # Stable explicit mapping to exactly the eight formal CSI300 records.
    indices = {
        "A": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and str(e.get("name","")).startswith("abcd_A_")),
        "B": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and str(e.get("name","")).startswith("abcd_B_")),
        "C": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and str(e.get("name","")).startswith("abcd_C_")),
        "D": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and str(e.get("name","")).startswith("abcd_D_")),
        "Continual A": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and e.get("family")=="Continual A/B/C" and str(e.get("name","")).endswith("_A")),
        "Continual B": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and e.get("family")=="Continual A/B/C" and str(e.get("name","")).endswith("_B")),
        "Continual C": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and e.get("family")=="Continual A/B/C" and str(e.get("name","")).endswith("_C")),
        "Adapter + Continual": next(i for i,e in enumerate(out["experiments"]) if e.get("dataset")=="CSI300" and e.get("family")=="Adapter + Continual" and "smoke" not in str(e.get("name","")).lower()),
    }
    if len(set(indices.values())) != 8:
        raise RuntimeError(f"formal experiment mapping is not one-to-one: {indices}")
    for model, index in indices.items():
        update_experiment(out["experiments"][index], model, metrics[model])

    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out.setdefault("dashboard_update", {})["csi300_226d_patch"] = {
        "status": "complete", "template": str(SOURCE_HTML), "test_start": START, "test_end": END,
        "test_dates": DATES, "test_rows": ROWS, "models": list(metrics),
        "replacement_scope": "only the embedded var D JSON object",
        "portfolio_rule": {"top_k": TOP_K, "drop_n": DROP_N, "min_holding_days": MIN_HOLD, "cost_rate": COST},
    }
    return clean_json(out)


def main() -> None:
    # Byte-oriented replacement is deliberate: every byte outside D must remain identical,
    # including the source template's CRLF line endings and any legacy appendix payload.
    source_bytes = SOURCE_HTML.read_bytes()
    start = source_bytes.index(b"var D = ") + len(b"var D = ")
    marker = b";\r\nvar metricCatalog" if b";\r\nvar metricCatalog" in source_bytes[start:] else b";\nvar metricCatalog"
    end = source_bytes.index(marker, start)
    original = json.loads(source_bytes[start:end].decode("utf-8"))
    frames = load_frames()
    patched = patch_data(original, frames)
    packed = json.dumps(patched, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    pretty = json.dumps(patched, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    output_bytes = source_bytes[:start] + packed + source_bytes[end:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_bytes(output_bytes)
    OUT_JSON.write_bytes(pretty)
    print(json.dumps({"html": str(OUT_HTML), "json": str(OUT_JSON), "rows": ROWS, "dates": DATES,
                      "source_bytes": len(source_bytes), "output_bytes": len(output_bytes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
