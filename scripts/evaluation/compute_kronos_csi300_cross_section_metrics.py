#!/usr/bin/env python
"""Compute detailed cross-sectional ranking metrics from per-date Kronos predictions.

Reads only finalized predictions/by_date/*.csv files, so it can safely produce an
interim snapshot while inference continues. All time-series summaries are date-equal
weighted. Q1 is the lowest predicted-signal group and Q5/Q10 is the highest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--min-cross-section", type=int, default=30)
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--nw-lag", type=int, default=9)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--bootstrap-block", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def corr(x, y, method="pearson"):
    z = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(z) < 3 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return math.nan
    return float(z.x.corr(z.y, method=method))


def nw_tstat(x: Iterable[float], lag: int):
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return math.nan
    u = x - x.mean()
    lrv = float(u @ u / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(u[k:] @ u[:-k] / n)
        lrv += 2 * (1 - k / (lag + 1)) * gamma
    se = math.sqrt(max(lrv, 0) / n)
    return float(x.mean() / se) if se > 0 else math.nan


def moving_block_mean_ci(x: Iterable[float], reps: int, block: int, seed: int):
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or reps <= 0:
        return [math.nan, math.nan]
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    means = np.empty(reps, dtype=float)
    blocks_needed = math.ceil(n / block)
    for r in range(reps):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[(s + np.arange(block)) % n] for s in chosen])[:n]
        means[r] = sample.mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def describe(x: Iterable[float], nw_lag: int, reps: int, block: int, seed: int):
    x = pd.Series(list(x), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(x)
    if n == 0:
        return {"count": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 and np.isfinite(sd) else math.nan
    return {
        "count": int(n), "mean": mean, "median": float(x.median()), "std": sd,
        "min": float(x.min()), "max": float(x.max()), "positive_ratio": float((x > 0).mean()),
        "mean_over_std": mean / sd if sd > 0 else math.nan,
        "annualized_mean_over_std_sqrt252": mean / sd * math.sqrt(252) if sd > 0 else math.nan,
        "iid_tstat": mean / se if se > 0 else math.nan,
        "newey_west_tstat": nw_tstat(x, nw_lag),
        "moving_block_bootstrap_mean_ci95": moving_block_mean_ci(x, reps, block, seed),
    }


def analyze_date(df: pd.DataFrame, q: int, top_n: int):
    required = {"as_of_date", "stock", "pred_path_mean_return", "actual_path_mean_return",
                "pred_endpoint_return", "actual_endpoint_return"}
    if not required <= set(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    df = df.drop_duplicates("stock", keep="last").copy()
    for col in required - {"as_of_date", "stock"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required - {"as_of_date", "stock"}))
    n = len(df)
    date = str(df.as_of_date.iloc[0])
    row = {"as_of_date": date, "n_stocks": n}
    pairs = [
        ("path_mean", "pred_path_mean_return", "actual_path_mean_return"),
        ("endpoint", "pred_endpoint_return", "actual_endpoint_return"),
    ]
    q_rows = []
    for tag, pred, actual in pairs:
        row[f"pearson_ic_{tag}"] = corr(df[pred], df[actual], "pearson")
        row[f"spearman_rankic_{tag}"] = corr(df[pred], df[actual], "spearman")
        ordered = df.sort_values([pred, "stock"], kind="mergesort").reset_index(drop=True)
        q_eff = min(q, n)
        ordered["quantile"] = np.floor(np.arange(n) * q_eff / n).astype(int) + 1
        qmeans = ordered.groupby("quantile", sort=True)[actual].mean()
        for quantile, val in qmeans.items():
            q_rows.append({"as_of_date": date, "target": tag, "quantile": int(quantile),
                           "mean_actual_return": float(val), "n_stocks": int((ordered["quantile"] == quantile).sum())})
        row[f"quantile_monotonic_rankcorr_{tag}"] = corr(pd.Series(qmeans.index, dtype=float), qmeans.values, "spearman")
        row[f"q_high_minus_q_low_{tag}"] = float(qmeans.iloc[-1] - qmeans.iloc[0])
        k = min(top_n, n // 2)
        predicted_top = ordered.nlargest(k, pred, keep="first")
        predicted_bottom = ordered.nsmallest(k, pred, keep="first")
        actual_top_stocks = set(ordered.nlargest(k, actual, keep="first").stock.astype(str))
        row[f"top{k}_mean_{tag}"] = float(predicted_top[actual].mean())
        row[f"bottom{k}_mean_{tag}"] = float(predicted_bottom[actual].mean())
        row[f"top{k}_minus_bottom{k}_{tag}"] = float(predicted_top[actual].mean() - predicted_bottom[actual].mean())
        row[f"universe_mean_{tag}"] = float(ordered[actual].mean())
        row[f"top{k}_excess_universe_{tag}"] = float(predicted_top[actual].mean() - ordered[actual].mean())
        row[f"top{k}_oracle_hit_rate_{tag}"] = float(len(set(predicted_top.stock.astype(str)) & actual_top_stocks) / k)
        row[f"random_hit_rate_baseline_{tag}"] = float(k / n)
    return row, q_rows


def main():
    a = parse_args()
    run = Path(a.run_dir).resolve()
    out = Path(a.output_dir).resolve() if a.output_dir else run / "ranking_metrics_detailed"
    files = sorted((run / "predictions" / "by_date").glob("*.csv"))
    daily_rows, quantile_rows, errors = [], [], []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if len(df) < a.min_cross_section:
                raise ValueError(f"only {len(df)} rows")
            row, qrows = analyze_date(df, a.quantiles, a.top_n)
            daily_rows.append(row); quantile_rows.extend(qrows)
        except Exception as e:
            errors.append({"file": fp.name, "error": repr(e)})
    daily = pd.DataFrame(daily_rows).sort_values("as_of_date") if daily_rows else pd.DataFrame()
    quantile_daily = pd.DataFrame(quantile_rows)
    atomic_csv(out / "daily_metrics.csv", daily)
    atomic_csv(out / "quantile_daily.csv", quantile_daily)
    atomic_csv(out / "errors.csv", pd.DataFrame(errors))
    if daily.empty:
        atomic_json(out / "summary.json", {"completed_dates": 0, "errors": errors})
        print("No valid prediction dates.")
        return
    metric_cols = [c for c in daily.columns if c not in {"as_of_date", "n_stocks"}]
    summaries = {}
    flat_rows = []
    for i, col in enumerate(metric_cols):
        stats = describe(daily[col], a.nw_lag, a.bootstrap_reps, a.bootstrap_block, a.seed + i)
        summaries[col] = stats
        flat_rows.append({"metric": col, **stats,
                          "ci95_low": stats.get("moving_block_bootstrap_mean_ci95", [math.nan, math.nan])[0],
                          "ci95_high": stats.get("moving_block_bootstrap_mean_ci95", [math.nan, math.nan])[1]})
    atomic_csv(out / "metric_summary.csv", pd.DataFrame(flat_rows).drop(columns=["moving_block_bootstrap_mean_ci95"], errors="ignore"))
    q_summary = (quantile_daily.groupby(["target", "quantile"], as_index=False)
                 .agg(date_count=("mean_actual_return", "count"), mean_return=("mean_actual_return", "mean"),
                      median_return=("mean_actual_return", "median"), std_return=("mean_actual_return", "std")))
    atomic_csv(out / "quantile_summary.csv", q_summary)
    key = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run), "completed_dates": int(len(daily)),
        "date_start": str(daily.as_of_date.min()), "date_end": str(daily.as_of_date.max()),
        "mean_stocks_per_date": float(daily.n_stocks.mean()),
        "primary_paper_eq13_signal": "mean(pred_close_1..10)/close_t-1",
        "primary_symmetric_label": "mean(actual_close_1..10)/close_t-1",
        "primary": {
            "pearson_ic": summaries["pearson_ic_path_mean"],
            "spearman_rankic": summaries["spearman_rankic_path_mean"],
            "quintile_high_minus_low": summaries["q_high_minus_q_low_path_mean"],
            f"top{a.top_n}_excess_universe": summaries[f"top{a.top_n}_excess_universe_path_mean"],
            f"top{a.top_n}_minus_bottom{a.top_n}": summaries[f"top{a.top_n}_minus_bottom{a.top_n}_path_mean"],
            f"top{a.top_n}_oracle_hit_rate": summaries[f"top{a.top_n}_oracle_hit_rate_path_mean"],
        },
        "auxiliary_endpoint": {
            "pearson_ic": summaries["pearson_ic_endpoint"],
            "spearman_rankic": summaries["spearman_rankic_endpoint"],
            "quintile_high_minus_low": summaries["q_high_minus_q_low_endpoint"],
            f"top{a.top_n}_excess_universe": summaries[f"top{a.top_n}_excess_universe_endpoint"],
            f"top{a.top_n}_minus_bottom{a.top_n}": summaries[f"top{a.top_n}_minus_bottom{a.top_n}_endpoint"],
        },
        "quantile_summary": q_summary.to_dict("records"),
        "inference_status": "interim snapshot if fewer than all configured dates are finalized",
        "interpretation": "paper-parameter-aligned, current-constituent approximation; not strict Kronos Table 10 reproduction",
        "method_notes": {
            "date_weighting": "equal weight per signal date", "quantile_orientation": "Q1 lowest predicted signal, highest Q is highest",
            "ICIR_ambiguity": "mean_over_std is unannualized; annualized_mean_over_std_sqrt252 is also reported",
            "overlap_horizon": "10-day labels overlap; Newey-West lag 9 and moving-block bootstrap block 10 are reported",
        },
        "errors": errors,
    }
    atomic_json(out / "summary.json", key)
    print(json.dumps(key, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
