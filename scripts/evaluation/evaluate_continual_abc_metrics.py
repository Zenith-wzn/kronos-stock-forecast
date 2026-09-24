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
RNG_SEED = 20260825
GROUPS = ("A", "B", "C")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (pd.Timestamp,)):
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
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    max_lag = min(lag, n - 1)
    for ell in range(1, max_lag + 1):
        gamma = float(np.dot(u[ell:], u[:-ell]) / n)
        weight = 1.0 - ell / (max_lag + 1.0)
        lrv += 2.0 * weight * gamma
    var_mean = max(lrv / n, 0.0)
    return float(x.mean() / math.sqrt(var_mean)) if var_mean > 0 else float("nan")


def moving_block_bootstrap_mean_ci(values: np.ndarray, block: int, reps: int, seed: int) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return [float("nan"), float("nan")]
    b = min(block, n)
    blocks = [x[i : i + b] for i in range(n - b + 1)]
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    need = math.ceil(n / b)
    for r in range(reps):
        chosen = rng.integers(0, len(blocks), size=need)
        means[r] = np.concatenate([blocks[i] for i in chosen])[:n].mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def binary_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    p = np.asarray(pred, dtype=float)[mask] > 0
    a = np.asarray(actual, dtype=float)[mask] > 0
    tp = int(np.sum(p & a)); tn = int(np.sum(~p & ~a))
    fp = int(np.sum(p & ~a)); fn = int(np.sum(~p & a))
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    balanced = float(np.nanmean([recall, specificity]))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "direction_accuracy": float(accuracy), "balanced_accuracy": float(balanced),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "specificity": float(specificity), "mcc": float(mcc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def cross_section_metrics(df: pd.DataFrame, pred_col: str, actual_col: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for date, g in df.groupby("as_of_date", sort=True):
        rows.append({
            "as_of_date": date,
            "n": int(len(g)),
            "ic": safe_corr(g[pred_col].to_numpy(), g[actual_col].to_numpy()),
            "rankic": spearman_corr(g[pred_col].to_numpy(), g[actual_col].to_numpy()),
        })
    daily = pd.DataFrame(rows)
    result: dict[str, Any] = {}
    for offset, col in enumerate(("ic", "rankic")):
        x = daily[col].to_numpy(dtype=float)
        good = x[np.isfinite(x)]
        mean = float(np.mean(good))
        std = float(np.std(good, ddof=1)) if good.size > 1 else float("nan")
        result[col] = {
            "count": int(good.size), "mean": mean, "median": float(np.median(good)), "std": std,
            "positive_ratio": float(np.mean(good > 0)),
            "ir_mean_over_std": float(mean / std) if std > 0 else float("nan"),
            "annualized_ir_sqrt252": float(mean / std * math.sqrt(252)) if std > 0 else float("nan"),
            "newey_west_lag9_mean_tstat": newey_west_tstat_mean(good, 9),
            "moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(good, BOOT_BLOCK, BOOT_REPS, seed + offset),
        }
    return result, daily


def scalar_point_metrics(pred: np.ndarray, actual: np.ndarray, scale: str) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    p = np.asarray(pred, dtype=float)[mask]
    y = np.asarray(actual, dtype=float)[mask]
    err = p - y
    denom_abs = np.mean(np.abs(y))
    denom_sq = np.sum(y ** 2)
    return {
        "available": True,
        "scale": scale,
        "mae_return": float(np.mean(np.abs(err))),
        "rmse_return": float(np.sqrt(np.mean(err ** 2))),
        "mase_vs_zero_return_persistence": float(np.mean(np.abs(err)) / denom_abs) if denom_abs > 0 else float("nan"),
        "oos_r2_vs_zero_return": float(1.0 - np.sum(err ** 2) / denom_sq) if denom_sq > 0 else float("nan"),
    }


def trajectory_point_metrics(a_full: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_pred: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []
    horizons: list[dict[str, Any]] = []
    current = a_full["current_close"].to_numpy(dtype=float)
    for h in range(1, H + 1):
        pred = a_full[f"pred_close_{h}"].to_numpy(dtype=float) / current - 1.0
        actual = a_full[f"actual_close_{h}"].to_numpy(dtype=float) / current - 1.0
        m = scalar_point_metrics(pred, actual, f"cumulative return horizon {h}")
        horizons.append({"horizon": h, **{k: v for k, v in m.items() if k not in ("available", "scale")}})
        all_pred.append(pred); all_actual.append(actual)
    p = np.concatenate(all_pred); y = np.concatenate(all_actual)
    overall = scalar_point_metrics(p, y, "10-step cumulative return trajectory pooled across horizons")
    return overall, horizons


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0)) if wealth.size else float("nan")


def sharpe_10day(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * math.sqrt(252 / 10)) if sd > 0 else float("nan")


def portfolio_metrics(df: pd.DataFrame, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    prev_weights: dict[str, float] = {}
    for date, g0 in df.groupby("as_of_date", sort=True):
        g = g0.sort_values(["pred_score", "stock"], ascending=[False, True]).reset_index(drop=True)
        n = len(g)
        k = max(1, int(math.ceil(n * 0.10)))
        top = g.iloc[:k]
        bottom = g.iloc[-k:]
        weights = {s: 1.0 / k for s in top["stock"].tolist()}
        names = set(prev_weights) | set(weights)
        turnover = 0.5 * sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in names) if prev_weights else 1.0
        prev_weights = weights
        row = {"as_of_date": date, "n": n, "k": k, "turnover": turnover}
        for label, actual_col in (("path", "actual_path"), ("endpoint", "actual_endpoint")):
            top_ret = float(top[actual_col].mean())
            bottom_ret = float(bottom[actual_col].mean())
            universe = float(g[actual_col].mean())
            longshort = top_ret - bottom_ret
            cost = COST_RATE * turnover
            row.update({
                f"top10_{label}": top_ret, f"bottom10_{label}": bottom_ret,
                f"longshort_{label}": longshort, f"top10_excess_{label}": top_ret - universe,
                f"top10_{label}_net": top_ret - cost, f"longshort_{label}_net": longshort - cost,
            })
        rows.append(row)
    daily = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "definition": "rank by model pred_score; exact top/bottom 10% using ceil(0.10*n); 10bps times one-way turnover",
        "dates": int(len(daily)), "transaction_cost_rate": COST_RATE,
        "mean_n": float(daily["n"].mean()), "mean_k": float(daily["k"].mean()),
        "mean_turnover": float(daily["turnover"].mean()),
    }
    for label in ("path", "endpoint"):
        out.update({
            f"mean_top10_{label}": float(daily[f"top10_{label}"].mean()),
            f"mean_bottom10_{label}": float(daily[f"bottom10_{label}"].mean()),
            f"mean_longshort_{label}": float(daily[f"longshort_{label}"].mean()),
            f"mean_top10_excess_{label}": float(daily[f"top10_excess_{label}"].mean()),
            f"mean_top10_{label}_net_10bps": float(daily[f"top10_{label}_net"].mean()),
            f"mean_longshort_{label}_net_10bps": float(daily[f"longshort_{label}_net"].mean()),
            f"sharpe_top10_{label}_net_10bps": sharpe_10day(daily[f"top10_{label}_net"].to_numpy()),
            f"sharpe_longshort_{label}_net_10bps": sharpe_10day(daily[f"longshort_{label}_net"].to_numpy()),
            f"mdd_top10_{label}_net_10bps": max_drawdown(daily[f"top10_{label}_net"].to_numpy()),
            f"mdd_longshort_{label}_net_10bps": max_drawdown(daily[f"longshort_{label}_net"].to_numpy()),
            f"positive_ratio_top10_{label}": float((daily[f"top10_{label}"] > 0).mean()),
            f"positive_ratio_longshort_{label}": float((daily[f"longshort_{label}"] > 0).mean()),
            f"top10_{label}_net_moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(daily[f"top10_{label}_net"].to_numpy(), BOOT_BLOCK, BOOT_REPS, seed + (0 if label == "path" else 10)),
            f"longshort_{label}_moving_block_bootstrap_block10_ci95": moving_block_bootstrap_mean_ci(daily[f"longshort_{label}"].to_numpy(), BOOT_BLOCK, BOOT_REPS, seed + (1 if label == "path" else 11)),
        })
    return out, daily


def load_a_full(a_dir: Path) -> pd.DataFrame:
    files = sorted((a_dir / "predictions" / "by_date").glob("*.csv"))
    if len(files) != 50:
        raise RuntimeError(f"expected 50 A prediction files, got {len(files)}")
    a = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)
    a["as_of_date"] = a["as_of_date"].astype(str)
    a["stock"] = a["stock"].astype(str)
    return a.sort_values(["as_of_date", "stock"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_dir, cache_dir, a_dir, out_dir = map(Path, (args.run_dir, args.cache_dir, args.a_dir, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    update_log = pd.read_csv(run_dir / "update_log.csv")
    meta = pd.read_csv(cache_dir / "meta_test.csv")
    labels = np.load(cache_dir / "labels_test.npy")
    if labels.shape != (len(meta), H):
        raise RuntimeError(f"unexpected labels shape {labels.shape}, meta rows {len(meta)}")
    meta["as_of_date"] = meta["as_of_date"].astype(str)
    meta["stock"] = meta["stock"].astype(str)

    # The strict-49 continual run intentionally starts on 2026-05-27, while
    # the shared static A cache contains the preceding 2026-05-26 date.
    # Select labels by the completed run's prediction keys after the run has
    # finished; this is evaluation-only and does not feed labels back into
    # training, replay, tuning, or model selection.
    run_keys = pd.read_csv(run_dir / "predictions_A.csv", usecols=["as_of_date", "stock"])
    run_keys["as_of_date"] = run_keys["as_of_date"].astype(str)
    run_keys["stock"] = run_keys["stock"].astype(str)
    run_key_set = set(map(tuple, run_keys[["as_of_date", "stock"]].drop_duplicates().to_numpy()))
    meta_mask = [tuple(x) in run_key_set for x in meta[["as_of_date", "stock"]].to_numpy()]
    meta = meta.loc[meta_mask].reset_index(drop=True)
    labels = labels[np.asarray(meta_mask, dtype=bool)]
    if len(meta) != len(run_keys) or len(meta) != len(run_keys.drop_duplicates()):
        raise RuntimeError("run prediction keys do not map one-to-one onto cache labels")
    actual = meta[["as_of_date", "stock"]].copy()
    actual["actual_path"] = labels.mean(axis=1)
    actual["actual_endpoint"] = labels[:, H - 1]
    actual = actual.sort_values(["as_of_date", "stock"]).reset_index(drop=True)

    a_full = load_a_full(a_dir)
    a_full = a_full.merge(run_keys.drop_duplicates(), on=["as_of_date", "stock"], how="inner")
    a_full = a_full.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    a_keys = a_full[["as_of_date", "stock"]]
    if not a_keys.equals(actual[["as_of_date", "stock"]]):
        raise RuntimeError("A full prediction keys do not equal cache keys")
    max_path_label_diff = float(np.max(np.abs(a_full["actual_path_mean_return"].to_numpy() - actual["actual_path"].to_numpy())))
    max_endpoint_label_diff = float(np.max(np.abs(a_full["actual_endpoint_return"].to_numpy() - actual["actual_endpoint"].to_numpy())))
    a_trajectory, a_horizons = trajectory_point_metrics(a_full)

    payload: dict[str, Any] = {
        "config": {**run_config, "cost_rate": COST_RATE, "bootstrap_reps": BOOT_REPS, "bootstrap_block": BOOT_BLOCK},
        "run_summary": run_summary,
        "update_log": update_log.to_dict("records"),
        "integrity": {
            "expected_rows": int(len(actual)), "expected_dates": int(actual["as_of_date"].nunique()),
            "first_date": str(actual["as_of_date"].min()), "last_date": str(actual["as_of_date"].max()),
            "labels_shape": list(labels.shape), "max_abs_a_path_label_diff_vs_cache": max_path_label_diff,
            "max_abs_a_endpoint_label_diff_vs_cache": max_endpoint_label_diff, "groups": {},
        },
        "groups": {},
    }
    summary_rows = []
    trajectory_rows = []
    for gi, group in enumerate(GROUPS):
        pred = pd.read_csv(run_dir / f"predictions_{group}.csv")
        pred["as_of_date"] = pred["as_of_date"].astype(str)
        pred["stock"] = pred["stock"].astype(str)
        pred = pred.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
        keys_equal = pred[["as_of_date", "stock"]].equals(actual[["as_of_date", "stock"]])
        if not keys_equal:
            raise RuntimeError(f"{group} keys do not equal cache keys")
        df = actual.copy()
        df["pred_score"] = pred["pred_signal"].to_numpy(dtype=float)
        path_label_diff = float(np.max(np.abs(pred["actual_signal"].to_numpy(dtype=float) - df["actual_path"].to_numpy(dtype=float))))
        integrity = {
            "rows": int(len(df)), "dates": int(df["as_of_date"].nunique()),
            "duplicates": int(pred.duplicated(["as_of_date", "stock"]).sum()),
            "keys_equal_strict_cache": bool(keys_equal),
            "finite_predictions": bool(np.isfinite(df["pred_score"]).all()),
            "finite_actuals": bool(np.isfinite(df[["actual_path", "actual_endpoint"]].to_numpy()).all()),
            "max_abs_actual_path_diff_vs_cache": path_label_diff,
        }
        payload["integrity"]["groups"][group] = integrity
        path_point = scalar_point_metrics(df["pred_score"].to_numpy(), df["actual_path"].to_numpy(), "scalar ranking score versus future 10-step mean return")
        df["pred_endpoint_score"] = a_full["pred_endpoint_return"].to_numpy(dtype=float) if group == "A" else df["pred_score"]
        path_direction = binary_metrics(df["pred_score"].to_numpy(), df["actual_path"].to_numpy())
        endpoint_direction = binary_metrics(df["pred_endpoint_score"].to_numpy(), df["actual_endpoint"].to_numpy())
        x_path, daily_path = cross_section_metrics(df, "pred_score", "actual_path", RNG_SEED + gi * 100)
        x_endpoint, daily_endpoint = cross_section_metrics(df, "pred_endpoint_score", "actual_endpoint", RNG_SEED + gi * 100 + 20)
        portfolio, portfolio_daily = portfolio_metrics(df, RNG_SEED + gi * 100 + 40)
        trajectory = a_trajectory if group == "A" else {
            "available": False, "scale": "10-step cumulative return trajectory pooled across horizons",
            "reason": "Frozen/continual ranking heads output one scalar ranking score, not a 10-step return trajectory.",
            "mae_return": None, "rmse_return": None, "mase_vs_zero_return_persistence": None, "oos_r2_vs_zero_return": None,
        }
        payload["groups"][group] = {
            "point_forecast": {"trajectory_overall": trajectory, "path_signal_overall": path_point},
            "direction_forecast": {"path_mean": path_direction, "endpoint": endpoint_direction,
                "cross_section_path_mean": x_path, "cross_section_endpoint": x_endpoint},
            "portfolio": portfolio,
        }
        if group == "A":
            trajectory_rows = [{"group": group, **r} for r in a_horizons]
        daily_path.rename(columns={"ic": "path_ic", "rankic": "path_rankic"}).merge(
            daily_endpoint.rename(columns={"ic": "endpoint_ic", "rankic": "endpoint_rankic"}),
            on=["as_of_date", "n"], how="outer",
        ).to_csv(out_dir / f"{group}_cross_section_metrics_by_date.csv", index=False)
        portfolio_daily.to_csv(out_dir / f"{group}_portfolio_metrics_by_date.csv", index=False)
        summary_rows.append({
            "group": group,
            "trajectory_mae": trajectory.get("mae_return"), "trajectory_rmse": trajectory.get("rmse_return"),
            "trajectory_mase": trajectory.get("mase_vs_zero_return_persistence"), "trajectory_oos_r2": trajectory.get("oos_r2_vs_zero_return"),
            "path_signal_mae": path_point["mae_return"], "path_signal_rmse": path_point["rmse_return"],
            "path_direction_accuracy": path_direction["direction_accuracy"], "path_balanced_accuracy": path_direction["balanced_accuracy"], "path_mcc": path_direction["mcc"],
            "path_ic_mean": x_path["ic"]["mean"], "path_rankic_mean": x_path["rankic"]["mean"],
            "path_rankic_positive_ratio": x_path["rankic"]["positive_ratio"], "path_rankic_nw_t": x_path["rankic"]["newey_west_lag9_mean_tstat"],
            "endpoint_ic_mean": x_endpoint["ic"]["mean"], "endpoint_rankic_mean": x_endpoint["rankic"]["mean"],
            "mean_top10_endpoint": portfolio["mean_top10_endpoint"], "mean_bottom10_endpoint": portfolio["mean_bottom10_endpoint"],
            "mean_longshort_endpoint": portfolio["mean_longshort_endpoint"], "mean_top10_excess_endpoint": portfolio["mean_top10_excess_endpoint"],
            "mean_turnover": portfolio["mean_turnover"], "mean_top10_endpoint_net_10bps": portfolio["mean_top10_endpoint_net_10bps"],
            "sharpe_top10_endpoint_net_10bps": portfolio["sharpe_top10_endpoint_net_10bps"], "mdd_top10_endpoint_net_10bps": portfolio["mdd_top10_endpoint_net_10bps"],
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "continual_abc_comparison_summary.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(out_dir / "A_point_metrics_by_horizon.csv", index=False)
    (out_dir / "continual_abc_metrics_full.json").write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_json({"status": "complete", "output_dir": str(out_dir), "summary": summary_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



