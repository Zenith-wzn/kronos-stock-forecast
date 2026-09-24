"""Common-sample evaluation for the CSI300 A/B/C experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED = {"as_of_date", "stock", "pred_signal", "actual_signal"}


def _load_one(source: str | Path | pd.DataFrame, group: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        out = source.copy()
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "predictions.csv"
        out = pd.read_csv(path)
    missing = REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"{group} predictions missing columns: {sorted(missing)}")
    out["group"] = group
    out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.normalize()
    out["stock"] = out["stock"].astype(str)
    return out


def load_common_predictions(run_dirs: Mapping[str, str | Path | pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return each group restricted to the exact common (date, stock) key set."""
    if set(run_dirs) != {"A", "B", "C"}:
        raise ValueError("run_dirs must contain exactly A, B and C")
    groups = {g: _load_one(src, g) for g, src in run_dirs.items()}
    keys = None
    for df in groups.values():
        current = set(zip(df.as_of_date, df.stock))
        keys = current if keys is None else keys & current
    if not keys:
        raise ValueError("no common (as_of_date, stock) samples")
    key_df = pd.DataFrame(sorted(keys), columns=["as_of_date", "stock"])
    return {
        g: df.merge(key_df, on=["as_of_date", "stock"], how="inner").sort_values(["as_of_date", "stock"]).reset_index(drop=True)
        for g, df in groups.items()
    }


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y, method=method))


def compute_daily_metrics(predictions: pd.DataFrame, *, top_n: int = 50) -> pd.DataFrame:
    missing = REQUIRED - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    rows = []
    for date, g in predictions.groupby("as_of_date", sort=True):
        g = g.dropna(subset=["pred_signal", "actual_signal"]).copy()
        if len(g) < 3:
            continue
        q = max(1, len(g) // 5)
        ordered = g.sort_values("pred_signal")
        top = ordered.tail(min(top_n, len(ordered)))
        bottom = ordered.head(q)
        rows.append({
            "as_of_date": pd.Timestamp(date), "n": len(g),
            "rankic": _corr(g.pred_signal, g.actual_signal, "spearman"),
            "pearson_ic": _corr(g.pred_signal, g.actual_signal, "pearson"),
            "q5_q1": float(ordered.tail(q).actual_signal.mean() - bottom.actual_signal.mean()),
            "top_n_return": float(top.actual_signal.mean()),
            "top_n_excess": float(top.actual_signal.mean() - g.actual_signal.mean()),
        })
    return pd.DataFrame(rows)


def newey_west_mean_t(values: Sequence[float], lag: int = 9) -> float:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3: return float("nan")
    mu = float(x.mean()); u = x - mu
    var = float((u @ u) / n)
    for k in range(1, min(lag, n - 1) + 1):
        cov = float((u[k:] @ u[:-k]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return float(mu / np.sqrt(max(var, 1.0e-15) / n))


def _summary(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty: return {"n_dates": 0}
    rank = metrics.rankic.to_numpy(dtype=float)
    return {"n_dates": int(len(metrics)), "rankic_mean": float(np.nanmean(rank)),
            "rankic_median": float(np.nanmedian(rank)), "rankic_nw_t_lag9": newey_west_mean_t(rank),
            "rankic_positive_ratio": float(np.nanmean(rank > 0)), "q5_q1_mean": float(metrics.q5_q1.mean()),
            "top_n_excess_mean": float(metrics.top_n_excess.mean())}


def compute_paired_deltas(common: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    result = {}
    daily = {g: compute_daily_metrics(df) for g, df in common.items()}
    result["absolute"] = {g: _summary(m) for g, m in daily.items()}
    for left, right, name in (("B", "A", "B-A"), ("C", "B", "C-B"), ("C", "A", "C-A")):
        l = daily[left].set_index("as_of_date"); r = daily[right].set_index("as_of_date")
        joined = l.join(r, lsuffix="_left", rsuffix="_right", how="inner")
        result[name] = {metric: float((joined[f"{metric}_left"] - joined[f"{metric}_right"]).mean())
                        for metric in ("rankic", "pearson_ic", "q5_q1", "top_n_excess")}
    return result


def write_evaluation_report(run_dirs: Mapping[str, str | Path | pd.DataFrame], output_dir: str | Path) -> dict[str, object]:
    common = load_common_predictions(run_dirs)
    report = compute_paired_deltas(common)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for g, df in common.items():
        df.to_csv(out / f"{g}_common_predictions.csv", index=False)
        compute_daily_metrics(df).to_csv(out / f"{g}_daily_metrics.csv", index=False)
    (out / "paired_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
