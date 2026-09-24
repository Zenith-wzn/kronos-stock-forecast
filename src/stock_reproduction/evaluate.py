from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .common import EXPECTED_TEST_DATES, EXPECTED_TEST_ROWS, REFERENCE, atomic_csv, atomic_json

def _newey_west_t(values: np.ndarray, lag: int = 9) -> float:
    centered, n = values-values.mean(), len(values); variance = float(centered@centered/n)
    for offset in range(1, min(lag, n-1)+1): variance += 2*(1-offset/(lag+1))*float(centered[offset:]@centered[:-offset]/n)
    standard_error = np.sqrt(max(variance, 0)/n)
    return float(values.mean()/standard_error) if standard_error else float("nan")

def _block_ci(values: np.ndarray, seed: int, repetitions: int = 5000, block: int = 10) -> list[float]:
    generator, n = np.random.default_rng(seed), len(values); means = np.empty(repetitions); starts = np.arange(max(1, n-block+1))
    for index in range(repetitions):
        sampled = []
        while len(sampled) < n:
            start = int(generator.choice(starts)); sampled.extend(values[start:start+block])
        means[index] = np.mean(sampled[:n])
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]

def evaluate(home: Path, model: str, seed: int) -> dict:
    predictions = pd.read_csv(home/"results/predictions.csv", dtype={"stock": str, "as_of_date": str})
    expected = pd.read_csv(REFERENCE/"eligible_stock_date_pairs.csv", dtype=str)
    expected = expected[expected.split.eq("test")][["stock", "as_of_date"]]
    if predictions.duplicated(["stock", "as_of_date"]).any(): raise ValueError("Duplicate prediction keys")
    actual_keys = set(map(tuple, predictions[["stock", "as_of_date"]].itertuples(index=False, name=None)))
    expected_keys = set(map(tuple, expected.itertuples(index=False, name=None)))
    if actual_keys != expected_keys or len(predictions) != EXPECTED_TEST_ROWS: raise ValueError(f"Prediction key mismatch: missing={len(expected_keys-actual_keys)}, extra={len(actual_keys-expected_keys)}")
    if not np.isfinite(predictions[["pred_signal", "actual_signal"]]).all().all(): raise ValueError("Non-finite predictions")
    daily = []
    for date, group in predictions.groupby("as_of_date", sort=True):
        ordered = group.sort_values("pred_signal"); top, bottom = ordered.tail(min(50, len(group))), ordered.head(min(50, len(group)))
        daily.append({"as_of_date": date, "n_stocks": len(group), "pearson_ic": float(pearsonr(group.pred_signal, group.actual_signal).statistic), "spearman_rankic": float(spearmanr(group.pred_signal, group.actual_signal).statistic), "top50_excess_universe": float(top.actual_signal.mean()-group.actual_signal.mean()), "top50_minus_bottom50": float(top.actual_signal.mean()-bottom.actual_signal.mean())})
    daily_frame = pd.DataFrame(daily)
    if len(daily_frame) != EXPECTED_TEST_DATES: raise ValueError("Expected 226 evaluated dates")
    atomic_csv(home/"results/evaluation/daily_metrics.csv", daily_frame)
    summary = {"model": model, "seed": seed, "rows": len(predictions), "dates": len(daily_frame), "metrics": {}, "portfolio_note": "Cross-sectional forward-return proxy; not an executable trading simulator."}
    for column in ("pearson_ic", "spearman_rankic", "top50_excess_universe", "top50_minus_bottom50"):
        values = daily_frame[column].dropna().to_numpy(); std = values.std(ddof=1)
        summary["metrics"][column] = {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(std), "mean_over_std": float(values.mean()/std), "newey_west_t_lag9": _newey_west_t(values), "moving_block_bootstrap_ci95_block10": _block_ci(values, seed)}
    atomic_json(home/"results/evaluation/summary.json", summary); return summary

