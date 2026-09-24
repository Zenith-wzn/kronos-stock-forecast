from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

H = 10
COST_RATE = 0.001
BOOT_REPS = 5000
BOOT_BLOCK = 10
RNG_SEED = 20260827


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(
        pd.Series(x).rank(method="average").to_numpy(dtype=float),
        pd.Series(y).rank(method="average").to_numpy(dtype=float),
    )


def newey_west_tstat_mean(values: np.ndarray, lag: int = 9) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    residual = x - x.mean()
    long_run_variance = float(np.dot(residual, residual) / n)
    max_lag = min(lag, n - 1)
    for ell in range(1, max_lag + 1):
        gamma = float(np.dot(residual[ell:], residual[:-ell]) / n)
        long_run_variance += 2.0 * (1.0 - ell / (max_lag + 1.0)) * gamma
    variance_of_mean = max(long_run_variance / n, 0.0)
    return float(x.mean() / math.sqrt(variance_of_mean)) if variance_of_mean > 0 else float("nan")


def moving_block_bootstrap_mean_ci(values: np.ndarray, seed: int) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return [float("nan"), float("nan")]
    block = min(BOOT_BLOCK, n)
    blocks = [x[i : i + block] for i in range(n - block + 1)]
    draws_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOT_REPS, dtype=float)
    for rep in range(BOOT_REPS):
        chosen = rng.integers(0, len(blocks), size=draws_needed)
        means[rep] = np.concatenate([blocks[i] for i in chosen])[:n].mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def scalar_point_metrics(pred: np.ndarray, actual: np.ndarray, scale: str) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    error = pred - actual
    baseline_mae = float(np.mean(np.abs(actual)))
    baseline_sse = float(np.sum(actual**2))
    return {
        "available": True,
        "scale": scale,
        "n": int(len(pred)),
        "mae_return": float(np.mean(np.abs(error))),
        "rmse_return": float(np.sqrt(np.mean(error**2))),
        "mase_vs_zero_return_persistence": float(np.mean(np.abs(error)) / baseline_mae) if baseline_mae > 0 else float("nan"),
        "oos_r2_vs_zero_return": float(1.0 - np.sum(error**2) / baseline_sse) if baseline_sse > 0 else float("nan"),
    }


def binary_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    p = pred[mask] > 0
    a = actual[mask] > 0
    tp = int(np.sum(p & a)); tn = int(np.sum(~p & ~a))
    fp = int(np.sum(p & ~a)); fn = int(np.sum(~p & a))
    n = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n,
        "direction_accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "specificity": specificity,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def cross_section_metrics(df: pd.DataFrame, pred_col: str, actual_col: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for date, group in df.groupby("as_of_date", sort=True):
        rows.append({
            "as_of_date": str(date),
            "n": int(len(group)),
            "ic": safe_corr(group[pred_col].to_numpy(), group[actual_col].to_numpy()),
            "rankic": spearman_corr(group[pred_col].to_numpy(), group[actual_col].to_numpy()),
        })
    daily = pd.DataFrame(rows)
    result: dict[str, Any] = {}
    for offset, metric in enumerate(("ic", "rankic")):
        values = daily[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        result[metric] = {
            "count": int(len(values)),
            "mean": mean,
            "median": float(np.median(values)),
            "std": std,
            "positive_ratio": float(np.mean(values > 0)),
            "ir_mean_over_std": float(mean / std) if std > 0 else float("nan"),
            "annualized_ir_sqrt252": float(mean / std * math.sqrt(252)) if std > 0 else float("nan"),
            "newey_west_lag9_mean_tstat": newey_west_tstat_mean(values),
            "moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(values, seed + offset),
        }
    return result, daily


def max_drawdown(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + returns)
    if not wealth.size:
        return float("nan")
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def sharpe_10day(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    std = np.std(returns, ddof=1)
    return float(np.mean(returns) / std * math.sqrt(252 / 10)) if std > 0 else float("nan")


def portfolio_metrics(df: pd.DataFrame, mode: str, pred_col: str, actual_col: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    previous_weights: dict[str, float] = {}
    rows = []
    for date, source in df.groupby("as_of_date", sort=True):
        group = source[["stock", pred_col, actual_col]].dropna().copy()
        group = group.sort_values([pred_col, "stock"], ascending=[False, True]).reset_index(drop=True)
        n = len(group)
        top_n = max(1, int(math.ceil(n * 0.10)))
        quintile_n = max(1, n // 5)
        top = group.iloc[:top_n]
        bottom = group.iloc[-top_n:]
        top_quintile = group.iloc[:quintile_n]
        bottom_quintile = group.iloc[-quintile_n:]
        weights = {stock: 1.0 / top_n for stock in top["stock"].tolist()}
        names = set(previous_weights) | set(weights)
        turnover = 0.5 * sum(abs(weights.get(stock, 0.0) - previous_weights.get(stock, 0.0)) for stock in names) if previous_weights else 1.0
        previous_weights = weights
        top_return = float(top[actual_col].mean())
        bottom_return = float(bottom[actual_col].mean())
        universe_return = float(group[actual_col].mean())
        longshort = top_return - bottom_return
        cost = COST_RATE * turnover
        rows.append({
            "as_of_date": str(date), "mode": mode, "n": n, "top_n": top_n, "quintile_n": quintile_n,
            "turnover": turnover, "q5_q1": float(top_quintile[actual_col].mean() - bottom_quintile[actual_col].mean()),
            "top10": top_return, "bottom10": bottom_return, "top10_excess": top_return - universe_return,
            "top10_net": top_return - cost, "longshort": longshort, "longshort_net": longshort - cost,
        })
    daily = pd.DataFrame(rows)
    top_net = daily["top10_net"].to_numpy(dtype=float)
    longshort_net = daily["longshort_net"].to_numpy(dtype=float)
    result = {
        "dates": int(len(daily)),
        "definition": "rank separately by matching Path/Endpoint signal; exact top/bottom 10% using ceil and Q5/Q1 using floor(n/5); 10bps times one-way turnover",
        "transaction_cost_rate": COST_RATE,
        "mean_n": float(daily["n"].mean()),
        "mean_k": float(daily["top_n"].mean()),
        f"mean_turnover_{mode}": float(daily["turnover"].mean()),
        f"mean_q5_q1_{mode}": float(daily["q5_q1"].mean()),
        f"mean_top10_{mode}": float(daily["top10"].mean()),
        f"mean_bottom10_{mode}": float(daily["bottom10"].mean()),
        f"mean_top10_excess_{mode}": float(daily["top10_excess"].mean()),
        f"mean_top10_{mode}_net_10bps": float(daily["top10_net"].mean()),
        f"mean_longshort_{mode}": float(daily["longshort"].mean()),
        f"mean_longshort_{mode}_net_10bps": float(daily["longshort_net"].mean()),
        f"sharpe_top10_{mode}_net_10bps": sharpe_10day(top_net),
        f"sharpe_longshort_{mode}_net_10bps": sharpe_10day(longshort_net),
        f"mdd_top10_{mode}_net_10bps": max_drawdown(top_net),
        f"mdd_longshort_{mode}_net_10bps": max_drawdown(longshort_net),
        f"positive_ratio_top10_{mode}": float((daily["top10"] > 0).mean()),
        f"positive_ratio_longshort_{mode}": float((daily["longshort"] > 0).mean()),
        f"top10_{mode}_net_moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(top_net, seed),
        f"longshort_{mode}_moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(longshort_net, seed + 1),
        f"q5_q1_{mode}_moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(daily["q5_q1"].to_numpy(dtype=float), seed + 2),
    }
    return result, daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute all reproducible CSI300 Adapter + Continual formal-test metrics.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions_path)
    df["as_of_date"] = df["as_of_date"].astype(str)
    df["stock"] = df["stock"].astype(str)
    df = df.sort_values(["as_of_date", "stock"]).reset_index(drop=True)

    required = ["as_of_date", "stock", "pred_signal", "actual_signal", "pred_endpoint_return", "actual_endpoint_return"]
    required += [f"pred_return_{h}" for h in range(1, H + 1)] + [f"actual_return_{h}" for h in range(1, H + 1)]
    missing = [col for col in required if col not in df]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    trajectory_pred = np.concatenate([df[f"pred_return_{h}"].to_numpy(dtype=float) for h in range(1, H + 1)])
    trajectory_actual = np.concatenate([df[f"actual_return_{h}"].to_numpy(dtype=float) for h in range(1, H + 1)])
    point_overall = scalar_point_metrics(trajectory_pred, trajectory_actual, "10-step cumulative return trajectory pooled across horizons")
    point_horizons = []
    for h in range(1, H + 1):
        metrics = scalar_point_metrics(df[f"pred_return_{h}"].to_numpy(), df[f"actual_return_{h}"].to_numpy(), f"cumulative return horizon {h}")
        point_horizons.append({"horizon": h, **{k: v for k, v in metrics.items() if k not in ("available", "scale")}})

    path_direction = binary_metrics(df["pred_signal"].to_numpy(), df["actual_signal"].to_numpy())
    endpoint_direction = binary_metrics(df["pred_endpoint_return"].to_numpy(), df["actual_endpoint_return"].to_numpy())
    path_cross, path_cross_daily = cross_section_metrics(df, "pred_signal", "actual_signal", RNG_SEED)
    endpoint_cross, endpoint_cross_daily = cross_section_metrics(df, "pred_endpoint_return", "actual_endpoint_return", RNG_SEED + 20)
    path_portfolio, path_portfolio_daily = portfolio_metrics(df, "path", "pred_signal", "actual_signal", RNG_SEED + 40)
    endpoint_portfolio, endpoint_portfolio_daily = portfolio_metrics(df, "endpoint", "pred_endpoint_return", "actual_endpoint_return", RNG_SEED + 60)
    portfolio = {**path_portfolio, **endpoint_portfolio}
    # Compatibility aliases used by the dashboard's final two columns.
    for mode in ("path", "endpoint"):
        portfolio[f"net_sharpe_{mode}"] = portfolio[f"sharpe_top10_{mode}_net_10bps"]
        portfolio[f"net_max_drawdown_{mode}"] = portfolio[f"mdd_top10_{mode}_net_10bps"]

    finite_cols = [col for col in required if col not in ("as_of_date", "stock")]
    payload = {
        "source": str(predictions_path.resolve()),
        "status": "complete",
        "evaluation": "formal_test_predictions_only",
        "dataset": "CSI300",
        "method": str(df["method"].iloc[0]) if "method" in df else "adapter_continual",
        "test_start": str(df["as_of_date"].min()),
        "test_end": str(df["as_of_date"].max()),
        "test_dates": int(df["as_of_date"].nunique()),
        "test_rows": int(len(df)),
        "strict_alignment_key": ["as_of_date", "stock"],
        "integrity": {
            "duplicate_keys": int(df.duplicated(["as_of_date", "stock"]).sum()),
            "finite_all_required_predictions_and_actuals": bool(np.isfinite(df[finite_cols].to_numpy(dtype=float)).all()),
            "path_mean_pred_signal_max_abs_diff": float(np.max(np.abs(df[[f"pred_return_{h}" for h in range(1, H + 1)]].mean(axis=1) - df["pred_signal"]))),
            "path_mean_actual_signal_max_abs_diff": float(np.max(np.abs(df[[f"actual_return_{h}" for h in range(1, H + 1)]].mean(axis=1) - df["actual_signal"]))),
            "endpoint_pred_max_abs_diff": float(np.max(np.abs(df["pred_return_10"] - df["pred_endpoint_return"]))),
            "endpoint_actual_max_abs_diff": float(np.max(np.abs(df["actual_return_10"] - df["actual_endpoint_return"]))),
        },
        "point_forecast": {"trajectory_overall": point_overall, "by_horizon": point_horizons},
        "direction_forecast": {
            "path_mean": path_direction,
            "endpoint": endpoint_direction,
            "cross_section_path_mean": path_cross,
            "cross_section_endpoint": endpoint_cross,
        },
        "portfolio": portfolio,
        "leakage_control": "Metrics use only stored formal-test predictions and matured actual_return_* labels; test labels are not used for training, replay, hyperparameter selection, or version selection.",
    }

    path_cross_daily.rename(columns={"ic": "path_ic", "rankic": "path_rankic"}).merge(
        endpoint_cross_daily.rename(columns={"ic": "endpoint_ic", "rankic": "endpoint_rankic"}),
        on=["as_of_date", "n"], how="outer",
    ).to_csv(output_dir / "adapter_cross_section_metrics_by_date.csv", index=False)
    pd.concat([path_portfolio_daily, endpoint_portfolio_daily], ignore_index=True).to_csv(output_dir / "adapter_portfolio_metrics_by_date.csv", index=False)
    pd.DataFrame(point_horizons).to_csv(output_dir / "adapter_point_metrics_by_horizon.csv", index=False)
    summary = {
        "dataset": "CSI300", "test_dates": payload["test_dates"], "test_rows": payload["test_rows"],
        "trajectory_mae": point_overall["mae_return"], "trajectory_rmse": point_overall["rmse_return"],
        "trajectory_mase": point_overall["mase_vs_zero_return_persistence"], "trajectory_oos_r2": point_overall["oos_r2_vs_zero_return"],
        "path_direction_accuracy": path_direction["direction_accuracy"], "endpoint_direction_accuracy": endpoint_direction["direction_accuracy"],
        "path_ic_mean": path_cross["ic"]["mean"], "path_rankic_mean": path_cross["rankic"]["mean"],
        "endpoint_ic_mean": endpoint_cross["ic"]["mean"], "endpoint_rankic_mean": endpoint_cross["rankic"]["mean"],
        "path_q5_q1": portfolio["mean_q5_q1_path"], "endpoint_q5_q1": portfolio["mean_q5_q1_endpoint"],
        "path_top10_net": portfolio["mean_top10_path_net_10bps"], "endpoint_top10_net": portfolio["mean_top10_endpoint_net_10bps"],
        "path_top10_net_sharpe": portfolio["net_sharpe_path"], "endpoint_top10_net_sharpe": portfolio["net_sharpe_endpoint"],
        "path_top10_net_mdd": portfolio["net_max_drawdown_path"], "endpoint_top10_net_mdd": portfolio["net_max_drawdown_endpoint"],
    }
    pd.DataFrame([summary]).to_csv(output_dir / "adapter_metrics_summary.csv", index=False)
    (output_dir / "adapter_metrics_full.json").write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_json(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

