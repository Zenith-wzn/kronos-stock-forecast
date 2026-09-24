from __future__ import annotations

"""Audit and compare the seed=100 RankIC- and Path-MAE-gated runs.

The audit uses prediction artifacts for signal metrics and the existing
``backtest_long_only`` implementation for next-trading-day portfolio returns.
It never uses test outcomes to select a checkpoint or gate an update.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only


GROUPS = ("A", "B", "C")
RUN_NAMES = {
    "rankic_gate": "continual_ABC",
    "mae_gate_tol001": "continual_ABC_mae_gate_tol001",
}


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    return value


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(x[mask].corr(y[mask], method=method))


def point_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    pred = pd.to_numeric(frame["pred_signal"], errors="coerce").to_numpy(float)
    actual = pd.to_numeric(frame["actual_signal"], errors="coerce").to_numpy(float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    error = pred - actual
    abs_actual = float(np.mean(np.abs(actual))) if len(actual) else float("nan")
    sq_actual = float(np.sum(actual**2)) if len(actual) else float("nan")
    p = pred > 0
    y = actual > 0
    tp = int(np.sum(p & y)); tn = int(np.sum(~p & ~y))
    fp = int(np.sum(p & ~y)); fn = int(np.sum(~p & y))
    n = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "rows": int(len(pred)),
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else float("nan"),
        "mase_vs_zero": float(np.mean(np.abs(error)) / abs_actual) if abs_actual > 0 else float("nan"),
        "oos_r2_vs_zero": float(1.0 - np.sum(error**2) / sq_actual) if sq_actual > 0 else float("nan"),
        "direction_accuracy": float(np.mean(p == y)) if len(p) else float("nan"),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "mcc": float((tp * tn - fp * fn) / denom) if denom else float("nan"),
    }


def daily_signal_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("as_of_date", sort=True):
        g = group.dropna(subset=["pred_signal", "actual_signal"]).copy()
        if len(g) < 3:
            continue
        q = max(1, len(g) // 5)
        ordered = g.sort_values(["pred_signal", "stock"], kind="mergesort")
        rows.append({
            "as_of_date": date,
            "n_stocks": int(len(g)),
            "rankic": corr(g["pred_signal"], g["actual_signal"], "spearman"),
            "pearson_ic": corr(g["pred_signal"], g["actual_signal"], "pearson"),
            "q5_q1": float(ordered.tail(q)["actual_signal"].mean() - ordered.head(q)["actual_signal"].mean()),
        })
    return pd.DataFrame(rows)


def read_predictions(run_dir: Path, group: str) -> pd.DataFrame:
    path = run_dir / f"predictions_{group}.csv"
    frame = pd.read_csv(path)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.normalize()
    frame["stock"] = frame["stock"].astype(str)
    frame = frame.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["as_of_date", "stock"]).any():
        raise ValueError(f"duplicate keys in {path}")
    return frame


def backtest(frame: pd.DataFrame, next_day: pd.DataFrame, *, model: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    data = frame[["as_of_date", "stock", "pred_signal"]].merge(
        next_day, on=["as_of_date", "stock"], how="left", validate="one_to_one"
    )
    data = data.rename(columns={"pred_signal": "prediction_score"})
    data = data.dropna(subset=["prediction_score", "next_day_return"])
    result = backtest_long_only(
        data,
        score_col="prediction_score",
        return_col="next_day_return",
        config=PortfolioConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=0.0015),
        dataset="CSI300",
        model=model,
        seed=seed,
    )
    return dict(result["summary"]), result["daily"].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(experiment_root / "B_linear_head" / "predictions.csv")
    reference["as_of_date"] = pd.to_datetime(reference["as_of_date"], errors="coerce").dt.normalize()
    reference["stock"] = reference["stock"].astype(str)
    reference = reference[["as_of_date", "stock", "actual_return_1"]].rename(
        columns={"actual_return_1": "next_day_return"}
    )
    if reference.duplicated(["as_of_date", "stock"]).any():
        raise ValueError("duplicate keys in next-day return reference")
    reference["next_day_return"] = pd.to_numeric(reference["next_day_return"], errors="coerce")

    report: dict[str, Any] = {
        "config": {
            "dataset": "CSI300",
            "seed": 100,
            "test_start": "2025-07-01",
            "test_end": "2026-06-05",
            "portfolio": {
                "top_k": 30,
                "drop_n": 3,
                "min_holding_days": 5,
                "cost_rate": 0.0015,
                "initial_capital": 1_000_000.0,
            },
            "signal": "pred_signal from each run's common stock-date predictions",
            "realized_return": "next_day_return from static B cache artifact (signal-date close to next-trading-day close)",
        },
        "runs": {},
        "comparisons": {},
    }
    daily_by_run_group: dict[tuple[str, str], pd.DataFrame] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    for run_name, directory_name in RUN_NAMES.items():
        run_dir = experiment_root / directory_name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        update_log = pd.read_csv(run_dir / "update_log.csv")
        report["runs"][run_name] = {
            "directory": str(run_dir),
            "summary": summary,
            "accepted_updates": int(
                update_log["accepted"].astype(str).str.strip().str.lower().eq("true").sum()
            ),
            "update_log": update_log.to_dict("records"),
            "groups": {},
        }
        for group in GROUPS:
            frame = read_predictions(run_dir, group)
            frames[(run_name, group)] = frame
            metrics = point_metrics(frame)
            daily = daily_signal_metrics(frame)
            daily.to_csv(output_dir / f"{run_name}_{group}_daily_signal_metrics.csv", index=False)
            portfolio_summary, portfolio_daily = backtest(
                frame, reference, model=f"{run_name}-{group}", seed=100
            )
            portfolio_daily.to_csv(output_dir / f"{run_name}_{group}_daily_returns.csv", index=False)
            signal_summary = {
                **metrics,
                "dates": int(daily["as_of_date"].nunique()),
                "mean_rankic": float(daily["rankic"].mean()),
                "mean_pearson_ic": float(daily["pearson_ic"].mean()),
                "mean_q5_q1": float(daily["q5_q1"].mean()),
            }
            report["runs"][run_name]["groups"][group] = {
                "signal": signal_summary,
                "portfolio": portfolio_summary,
            }
            daily_by_run_group[(run_name, group)] = daily

    for group in GROUPS:
        old = report["runs"]["rankic_gate"]["groups"][group]
        new = report["runs"]["mae_gate_tol001"]["groups"][group]
        report["comparisons"][f"mae_gate_minus_rankic_gate_{group}"] = {
            "signal": {
                key: finite(new["signal"].get(key)) - finite(old["signal"].get(key))
                for key in ("mae", "rmse", "mase_vs_zero", "oos_r2_vs_zero", "direction_accuracy", "balanced_accuracy", "mcc", "mean_rankic", "mean_pearson_ic", "mean_q5_q1")
                if finite(new["signal"].get(key)) is not None and finite(old["signal"].get(key)) is not None
            },
            "portfolio": {
                key: finite(new["portfolio"].get(key)) - finite(old["portfolio"].get(key))
                for key in ("net_cumulative_return", "annualized_net_return", "maximum_drawdown", "turnover_mean", "cumulative_transaction_cost_amount", "trade_count")
                if finite(new["portfolio"].get(key)) is not None and finite(old["portfolio"].get(key)) is not None
            },
        }

    for left, right, label in (
        (("rankic_gate", "C"), ("rankic_gate", "B"), "rankic_gate_C_minus_B"),
        (("mae_gate_tol001", "C"), ("mae_gate_tol001", "B"), "mae_gate_tol001_C_minus_B"),
    ):
        left_frame = frames[left].set_index(["as_of_date", "stock"])
        right_frame = frames[right].set_index(["as_of_date", "stock"])
        joined = left_frame[["pred_signal"]].join(right_frame[["pred_signal"]], lsuffix="_left", rsuffix="_right", how="inner")
        report["comparisons"][label] = {
            "rows": int(len(joined)),
            "mean_abs_signal_difference": float(np.mean(np.abs(joined.pred_signal_left - joined.pred_signal_right))),
            "max_abs_signal_difference": float(np.max(np.abs(joined.pred_signal_left - joined.pred_signal_right))),
            "nonzero_signal_rows": int(np.sum(np.abs(joined.pred_signal_left - joined.pred_signal_right) > 1e-12)),
        }

    report["integrity"] = {
        "reference_rows": int(len(reference)),
        "reference_dates": int(reference["as_of_date"].nunique()),
        "reference_finite_next_day_returns": bool(reference["next_day_return"].notna().all()),
        "groups": {
            f"{run_name}_{group}": {
                "rows": int(len(frame)),
                "dates": int(frame["as_of_date"].nunique()),
                "start": frame["as_of_date"].min(),
                "end": frame["as_of_date"].max(),
                "duplicate_keys": int(frame.duplicated(["as_of_date", "stock"]).sum()),
                "finite_predictions": bool(pd.to_numeric(frame["pred_signal"], errors="coerce").notna().all()),
            }
            for (run_name, group), frame in frames.items()
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
