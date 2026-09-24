from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    pred = pd.to_numeric(frame["pred_signal"], errors="coerce")
    actual = pd.to_numeric(frame["actual_signal"], errors="coerce")
    mask = pred.notna() & actual.notna()
    pred, actual = pred[mask], actual[mask]
    daily = []
    for _, group in frame.loc[mask].groupby("as_of_date", sort=True):
        p = pd.to_numeric(group["pred_signal"], errors="coerce")
        a = pd.to_numeric(group["actual_signal"], errors="coerce")
        daily.append((p.corr(a, method="spearman"), p.corr(a, method="pearson"),
                      a.loc[p.nlargest(max(1, len(p) // 5)).index].mean() -
                      a.loc[p.nsmallest(max(1, len(p) // 5)).index].mean()))
    arr = np.asarray(daily, dtype=float)
    return {
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "rankic": float(np.nanmean(arr[:, 0])),
        "pearson_ic": float(np.nanmean(arr[:, 1])),
        "q5_q1": float(np.nanmean(arr[:, 2])),
    }


def main() -> None:
    root = ROOT / "train3" / "csi300_csi800_aligned_seed100" / "experiments"
    strict = ROOT / "continual_ABC_mae_strict_gate_retry01"
    reference = pd.read_csv(root / "B_linear_head" / "predictions.csv")
    reference["as_of_date"] = pd.to_datetime(reference["as_of_date"]).dt.normalize()
    reference["stock"] = reference["stock"].astype(str)
    reference = reference[["as_of_date", "stock", "actual_return_1"]].rename(columns={"actual_return_1": "next_day_return"})
    runs = {
        "rankic_gate": root / "continual_ABC",
        "mae_gate_tol001": root / "continual_ABC_mae_gate_tol001",
        "mae_gate_strict": strict,
    }
    report = {}
    for name, directory in runs.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        update = pd.read_csv(directory / "update_log.csv")
        report[name] = {"accepted_updates": int(update["accepted"].astype(str).str.lower().eq("true").sum()), "summary": summary, "groups": {}}
        for group in ("A", "B", "C"):
            frame = pd.read_csv(directory / f"predictions_{group}.csv")
            frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
            frame["stock"] = frame["stock"].astype(str)
            data = frame[["as_of_date", "stock", "pred_signal"]].merge(reference, on=["as_of_date", "stock"], validate="one_to_one")
            data = data.rename(columns={"pred_signal": "prediction_score"}).dropna(subset=["prediction_score", "next_day_return"])
            result = backtest_long_only(data, score_col="prediction_score", return_col="next_day_return",
                                        config=PortfolioConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=0.0015),
                                        dataset="CSI300", model=f"{name}-{group}", seed=100)
            daily = result["daily"].copy()
            daily.to_csv(ROOT / "results" / "continual_seed100_mae_gate_audit" / f"{name}_{group}_daily_returns.csv", index=False)
            report[name]["groups"][group] = {"signal": metrics(frame), "portfolio": dict(result["summary"])}
    for name in runs:
        c = report[name]["groups"]["C"]["portfolio"]
        b = report[name]["groups"]["B"]["portfolio"]
        report[name]["C_minus_B"] = {k: c[k] - b[k] for k in ("net_cumulative_return", "annualized_net_return", "maximum_drawdown", "sharpe", "turnover_mean", "cumulative_transaction_cost_amount", "trade_count")}
    out = ROOT / "results" / "continual_seed100_mae_gate_audit" / "strict_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name, run in report.items():
        print(name, "accepted_updates=", run["accepted_updates"])
        for group, item in run["groups"].items():
            s, p = item["signal"], item["portfolio"]
            print(group, f"MAE={s['mae']:.8f}", f"RankIC={s['rankic']:.8f}", f"Net={p['net_cumulative_return']:.8f}", f"Ann={p['annualized_net_return']:.8f}", f"MDD={p['maximum_drawdown']:.8f}", f"Sharpe={p['sharpe']:.6f}", f"Cost={p['cumulative_transaction_cost_amount']:.2f}", f"Trades={p['trade_count']}")
        print("C_minus_B", run["C_minus_B"])


if __name__ == "__main__":
    main()
