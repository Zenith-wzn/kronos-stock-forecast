#!/usr/bin/env python
"""Kronos zero-shot cross-sectional ranking on the current-constituent CSI800 panel.

The inference configuration follows the Kronos investment experiment: 90-day context,
10-day horizon, T=0.6, top-p=0.9, and Monte Carlo sample_count=10. Predictions are
checkpointed atomically by signal date so long GPU runs can resume safely.

Important: the supplied panel contains constituents downloaded on 2026-08-20 rather
than point-in-time CSI800 membership. Therefore this is a paper-parameter-aligned,
current-constituent approximation, not a strict reproduction of paper Table 10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED = ["date", "code", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
VALUE_COLS = PRICE_COLS + ["volume"]
PRED_STEPS = tuple(range(1, 11))


@dataclass(frozen=True)
class RunConfig:
    stocks_dir: str
    kronos_repo: str
    output_dir: str
    model_name: str
    tokenizer_name: str
    device: str
    lookback: int
    pred_len: int
    start_date: str
    end_date: str
    temperature: float
    top_p: float
    top_k: int
    sample_count: int
    seed: int
    batch_size: int
    max_stocks: int | None
    max_dates: int | None
    date_offset: int
    min_cross_section: int
    calendar_min_fraction: float
    quantiles: int
    top_n: int
    data_vintage: str
    signal_definition: str
    label_definition: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stocks-dir", default=str(here / "data" / "data" / "csi800_daily"))
    p.add_argument("--kronos-repo", default=str(here / "third_party" / "Kronos_official_src" / "Kronos-master"))
    p.add_argument("--output-dir", default=str(here / "train3" / "kronos_csi800_paper_zero_shot_small_20240701_20250605"))
    p.add_argument("--model-name", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer-name", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--start-date", default="2025-07-01")
    p.add_argument("--end-date", default="2026-06-05")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--sample-count", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32, help="Number of stocks per call; each is expanded sample_count times internally.")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--date-offset", type=int, default=0)
    p.add_argument("--min-cross-section", type=int, default=30)
    p.add_argument("--calendar-min-fraction", type=float, default=0.5)
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(base: int, *parts: str) -> int:
    payload = "|".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def config_from_args(a: argparse.Namespace) -> RunConfig:
    return RunConfig(
        stocks_dir=str(Path(a.stocks_dir).resolve()), kronos_repo=str(Path(a.kronos_repo).resolve()),
        output_dir=str(Path(a.output_dir).resolve()), model_name=a.model_name,
        tokenizer_name=a.tokenizer_name, device=a.device, lookback=a.lookback,
        pred_len=a.pred_len, start_date=a.start_date, end_date=a.end_date,
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
        sample_count=a.sample_count, seed=a.seed, batch_size=a.batch_size,
        max_stocks=a.max_stocks, max_dates=a.max_dates, date_offset=a.date_offset,
        min_cross_section=a.min_cross_section, calendar_min_fraction=a.calendar_min_fraction,
        quantiles=a.quantiles, top_n=a.top_n, data_vintage="current CSI800 constituents downloaded 2026-08-20",
        signal_definition="mean(predicted close steps 1..10) / close_t - 1 (paper Eq.13 investment signal)",
        label_definition="mean(actual close steps 1..10) / close_t - 1 (symmetric ranking label)",
    )


def validate_config(c: RunConfig) -> None:
    if c.lookback != 90 or c.pred_len != 10:
        raise ValueError("Paper investment-task alignment requires --lookback 90 and --pred-len 10.")
    if c.sample_count < 1 or c.batch_size < 1:
        raise ValueError("sample-count and batch-size must be positive.")
    if not (0 < c.calendar_min_fraction <= 1):
        raise ValueError("calendar-min-fraction must be in (0, 1].")
    if c.min_cross_section < 3:
        raise ValueError("min-cross-section must be >= 3 for correlations.")


def load_panel(c: RunConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    root = Path(c.stocks_dir)
    files = sorted(root.glob("*.csv"))
    if c.max_stocks is not None:
        files = files[: c.max_stocks]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    panel: dict[str, pd.DataFrame] = {}
    inventory = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            missing = sorted(set(REQUIRED) - set(df.columns))
            if missing:
                raise ValueError(f"missing columns {missing}")
            df = df[REQUIRED].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            for col in VALUE_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
            code_values = df["code"].dropna().astype(str)
            code = code_values.iloc[-1] if len(code_values) else fp.stem
            df["code"] = code
            df = df.set_index("date", drop=False)
            panel[code] = df
            inventory.append({"stock": code, "file": fp.name, "rows": len(df),
                              "first_date": df.index.min().date().isoformat() if len(df) else None,
                              "last_date": df.index.max().date().isoformat() if len(df) else None,
                              "file_sha256": sha256_file(fp)})
        except Exception as e:
            inventory.append({"stock": fp.stem, "file": fp.name, "rows": 0,
                              "first_date": None, "last_date": None, "load_error": repr(e)})
    if not panel:
        raise RuntimeError("All stock CSV files failed to load.")
    return panel, pd.DataFrame(inventory)


def build_master_calendar(panel: dict[str, pd.DataFrame], c: RunConfig) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    counts: dict[pd.Timestamp, int] = {}
    start, end = pd.Timestamp(c.start_date), pd.Timestamp(c.end_date)
    for df in panel.values():
        dates = df.index[(df.index >= start) & (df.index <= end)]
        for d in dates:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        raise RuntimeError("No observations in requested test range.")
    max_count = max(counts.values())
    threshold = max(c.min_cross_section, math.ceil(len(panel) * c.calendar_min_fraction))
    dates = sorted(d for d, n in counts.items() if n >= threshold)
    dates = dates[c.date_offset:]
    if c.max_dates is not None:
        dates = dates[: c.max_dates]
    cal = pd.DataFrame([{"as_of_date": d.date().isoformat(), "observed_stocks": counts[d],
                         "calendar_threshold": threshold, "max_observed_stocks": max_count,
                         "selected": d in set(dates)} for d in sorted(counts)])
    if not dates:
        raise RuntimeError(f"No master dates passed threshold={threshold}.")
    return dates, cal


def valid_numeric_window(df: pd.DataFrame) -> bool:
    vals = df[VALUE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        return False
    if (df[PRICE_COLS].to_numpy(dtype=np.float64) <= 0).any():
        return False
    if (df["volume"].to_numpy(dtype=np.float64) < 0).any():
        return False
    return True


def prepare_date(panel: dict[str, pd.DataFrame], date: pd.Timestamp, c: RunConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
    for code in sorted(panel):
        df = panel[code]
        try:
            loc = df.index.get_loc(date)
            if not isinstance(loc, (int, np.integer)):
                reject("duplicate_or_ambiguous_date"); continue
            if loc < c.lookback - 1:
                reject("insufficient_history"); continue
            if loc + c.pred_len >= len(df):
                reject("insufficient_future"); continue
            context = df.iloc[loc - c.lookback + 1: loc + 1]
            future = df.iloc[loc + 1: loc + 1 + c.pred_len]
            if len(context) != c.lookback or len(future) != c.pred_len:
                reject("window_length_mismatch"); continue
            if context.index.max() != date or not (context.index <= date).all():
                reject("context_date_leakage"); continue
            if not (future.index > date).all():
                reject("future_date_not_after_signal"); continue
            if not valid_numeric_window(context) or not valid_numeric_window(future):
                reject("invalid_ohlcv"); continue
            item = {"stock": code, "context": context, "future": future,
                    "current_close": float(context["close"].iloc[-1]),
                    "current_open": float(context["open"].iloc[-1])}
            items.append(item)
        except KeyError:
            reject("missing_as_of_date")
        except Exception:
            reject("window_exception")
    return items, reasons


def run_splitter_self_test() -> dict[str, Any]:
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    base = pd.DataFrame({"date": dates, "code": "SYN", "open": np.arange(120)+10.0,
                         "high": np.arange(120)+11.0, "low": np.arange(120)+9.0,
                         "close": np.arange(120)+10.5, "volume": np.arange(120)+100.0}).set_index("date", drop=False)
    c = RunConfig(".", ".", ".", "m", "t", "cpu", 90, 10, str(dates[89].date()), str(dates[89].date()),
                  .6, .9, 0, 10, 100, 2, None, None, 0, 3, .5, 5, 50, "synthetic", "s", "l")
    original, _ = prepare_date({"SYN": base}, dates[89], c)
    altered = base.copy()
    altered.loc[altered.index > dates[89], "close"] = 1e9
    mutated, _ = prepare_date({"SYN": altered}, dates[89], c)
    a = original[0]["context"][VALUE_COLS].to_numpy()
    b = mutated[0]["context"][VALUE_COLS].to_numpy()
    if not np.array_equal(a, b):
        raise AssertionError("Future-label perturbation changed model context: leakage detected.")
    if original[0]["future"]["close"].equals(mutated[0]["future"]["close"]):
        raise AssertionError("Self-test failed to perturb labels.")
    return {"name": "future_perturbation_splitter_test", "passed": True,
            "assertion": "changing all post-signal closes leaves the 90-row model context byte-identical"}


def load_predictor(c: RunConfig):
    repo = str(Path(c.kronos_repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained(c.tokenizer_name)
    model = Kronos.from_pretrained(c.model_name)
    predictor = KronosPredictor(model, tokenizer, device=c.device, max_context=512)
    model.eval(); tokenizer.eval()
    return predictor, torch


def make_row(item: dict[str, Any], pred: pd.DataFrame, date: pd.Timestamp, c: RunConfig, batch_index: int) -> dict[str, Any]:
    pc = pred["close"].to_numpy(dtype=np.float64)
    ac = item["future"]["close"].to_numpy(dtype=np.float64)
    ao = item["future"]["open"].to_numpy(dtype=np.float64)
    if len(pc) != c.pred_len or not np.isfinite(pc).all():
        raise ValueError("invalid predicted close path")
    close_t = item["current_close"]
    row: dict[str, Any] = {
        "model": c.model_name, "seed": c.seed, "as_of_date": date.date().isoformat(),
        "stock": item["stock"], "batch_index": batch_index,
        "current_open": item["current_open"], "current_close": close_t,
        "next_open": float(ao[0]),
        "pred_path_mean_return": float(pc.mean() / close_t - 1.0),
        "actual_path_mean_return": float(ac.mean() / close_t - 1.0),
        "pred_endpoint_return": float(pc[-1] / close_t - 1.0),
        "actual_endpoint_return": float(ac[-1] / close_t - 1.0),
    }
    for i in range(c.pred_len):
        row[f"pred_close_{i+1}"] = float(pc[i])
        row[f"actual_close_{i+1}"] = float(ac[i])
        row[f"actual_open_{i+1}"] = float(ao[i])
        row[f"future_date_{i+1}"] = item["future"].index[i].date().isoformat()
    return row


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    z = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return math.nan, math.nan, len(z)
    pearson = float(z["x"].corr(z["y"], method="pearson"))
    rankic = float(z["x"].rank(method="average").corr(z["y"].rank(method="average"), method="pearson"))
    return pearson, rankic, len(z)


def daily_metrics(df: pd.DataFrame, c: RunConfig) -> dict[str, Any]:
    d = str(df["as_of_date"].iloc[0])
    ic, ric, n = corr_pair(df["pred_path_mean_return"], df["actual_path_mean_return"])
    eic, eric, en = corr_pair(df["pred_endpoint_return"], df["actual_endpoint_return"])
    ordered = df.sort_values(["pred_path_mean_return", "stock"], kind="mergesort").reset_index(drop=True)
    qn = min(c.quantiles, len(ordered))
    ordered["quantile"] = np.floor(np.arange(len(ordered)) * qn / len(ordered)).astype(int) + 1
    qmeans = ordered.groupby("quantile", sort=True)["actual_path_mean_return"].mean()
    q_ic, q_rankic, _ = corr_pair(pd.Series(qmeans.index, dtype=float), qmeans.reset_index(drop=True))
    topn = min(c.top_n, len(ordered))
    top = ordered.nlargest(topn, "pred_path_mean_return", keep="first")
    result: dict[str, Any] = {
        "as_of_date": d, "n_stocks": n, "pearson_ic_path_mean": ic, "spearman_rankic_path_mean": ric,
        "pearson_ic_endpoint": eic, "spearman_rankic_endpoint": eric,
        "quantile_monotonic_pearson": q_ic, "quantile_monotonic_spearman": q_rankic,
        "top_n_effective": topn,
        "topn_actual_path_mean_return": float(top["actual_path_mean_return"].mean()),
        "universe_actual_path_mean_return": float(ordered["actual_path_mean_return"].mean()),
        "topn_excess_path_mean_return": float(top["actual_path_mean_return"].mean() - ordered["actual_path_mean_return"].mean()),
        "topn_actual_endpoint_return": float(top["actual_endpoint_return"].mean()),
        "universe_actual_endpoint_return": float(ordered["actual_endpoint_return"].mean()),
        "topn_excess_endpoint_return": float(top["actual_endpoint_return"].mean() - ordered["actual_endpoint_return"].mean()),
    }
    for q, value in qmeans.items():
        result[f"q{int(q)}_actual_path_mean_return"] = float(value)
    return result


def nw_mean_tstat(values: Iterable[float], lag: int = 9) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 3:
        return math.nan
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(lrv, 0.0) / n)
    return float(x.mean() / se) if se > 0 else math.nan


def describe_metric(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {"count": 0}
    sd = float(x.std(ddof=1)) if len(x) > 1 else math.nan
    mean = float(x.mean())
    return {"count": int(len(x)), "mean": mean, "median": float(x.median()), "std": sd,
            "positive_ratio": float((x > 0).mean()), "ir_mean_over_std": mean / sd if sd and np.isfinite(sd) else math.nan,
            "annualized_ir_sqrt252": mean / sd * math.sqrt(252) if sd and np.isfinite(sd) else math.nan,
            "newey_west_lag9_mean_tstat": nw_mean_tstat(x, 9)}


def summarize_outputs(out: Path, c: RunConfig) -> dict[str, Any]:
    files = sorted((out / "predictions" / "by_date").glob("*.csv"))
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if len(df) >= c.min_cross_section and df["stock"].is_unique:
                rows.append(daily_metrics(df, c))
        except Exception as e:
            print(f"[WARN] cannot summarize {fp.name}: {e}", flush=True)
    daily = pd.DataFrame(rows).sort_values("as_of_date") if rows else pd.DataFrame()
    atomic_write_csv(out / "metrics" / "daily_cross_section_metrics.csv", daily)
    if daily.empty:
        summary = {"completed_dates": 0, "status": "no valid completed dates"}
    else:
        summary = {
            "completed_dates": int(len(daily)),
            "date_start": str(daily["as_of_date"].min()), "date_end": str(daily["as_of_date"].max()),
            "stocks_per_date": describe_metric(daily["n_stocks"]),
            "path_mean_signal_vs_symmetric_label": {
                "pearson_ic": describe_metric(daily["pearson_ic_path_mean"]),
                "spearman_rankic": describe_metric(daily["spearman_rankic_path_mean"]),
            },
            "endpoint_signal_vs_endpoint_label": {
                "pearson_ic": describe_metric(daily["pearson_ic_endpoint"]),
                "spearman_rankic": describe_metric(daily["spearman_rankic_endpoint"]),
            },
            "top200_current_pool_approximation": {
                "mean_topn_excess_path_mean_return": float(daily["topn_excess_path_mean_return"].mean()),
                "mean_topn_excess_endpoint_return": float(daily["topn_excess_endpoint_return"].mean()),
            },
            "interpretation": "paper-parameter-aligned, current-constituent approximation; not strict Table 10 reproduction",
        }
    atomic_write_json(out / "metrics" / "summary.json", summary)
    return summary


def record_error(errors_path: Path, row: dict[str, Any]) -> None:
    old = pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    atomic_write_csv(errors_path, new)


def valid_completed_file(path: Path, date: pd.Timestamp, c: RunConfig) -> bool:
    try:
        df = pd.read_csv(path)
        needed = {"model", "seed", "as_of_date", "stock", "pred_path_mean_return", "actual_path_mean_return"}
        return (needed <= set(df.columns) and len(df) >= c.min_cross_section and df["stock"].is_unique
                and set(df["as_of_date"].astype(str)) == {date.date().isoformat()}
                and set(df["model"].astype(str)) == {c.model_name} and set(df["seed"].astype(int)) == {c.seed})
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    c = config_from_args(args)
    validate_config(c)
    out = Path(c.output_dir)
    by_date = out / "predictions" / "by_date"
    partial_dir = out / "predictions" / "partial"
    by_date.mkdir(parents=True, exist_ok=True); partial_dir.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    errors_path = out / "errors.csv"
    config_path = out / "run_config.json"
    current_cfg = asdict(c)
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != current_cfg:
            raise RuntimeError(f"Existing output directory has a different run_config.json: {config_path}")
    else:
        atomic_write_json(config_path, current_cfg)

    self_test = run_splitter_self_test()
    panel, inventory = load_panel(c)
    atomic_write_csv(out / "data_inventory.csv", inventory)
    dates, calendar_df = build_master_calendar(panel, c)
    atomic_write_csv(out / "master_calendar.csv", calendar_df)
    script_path = Path(__file__).resolve()
    manifest = {
        "created_utc": utc_now(), "script": str(script_path), "script_sha256": sha256_file(script_path),
        "official_investment_parameters": {"lookback": 90, "forecast_horizon": 10, "temperature": 0.6,
            "top_p": 0.9, "paper_table6_sample_count": 10, "official_finetune_demo_sample_count": 5,
            "main_run_choice": "N=10 follows paper Table 6; demo discrepancy is disclosed"},
        "prediction_path_storage": "official Kronos predict_batch averages sample_count decoded paths internally; only the averaged path is available without modifying official model code",
        "cross_section": "master market dates; stocks require exact signal-date observation, 90 prior/including observations, and 10 subsequent observed bars",
        "data_limitations": ["static constituents downloaded 2026-08-20; survivorship bias",
            "raw/unadjusted vendor OHLCV rather than confirmed Qlib paper data", "no point-in-time CSI800 membership",
            "ranking results are not strict paper Table 10 AER/IR reproduction"],
        "leakage_test": self_test, "selected_date_count": len(dates), "loaded_stock_count": len(panel),
    }
    atomic_write_json(out / "manifest.json", manifest)
    if args.summarize_only:
        print(json.dumps(summarize_outputs(out, c), ensure_ascii=False, indent=2), flush=True)
        return

    pending = [d for d in dates if args.force_recompute or not valid_completed_file(by_date / f"{d.date().isoformat()}.csv", d, c)]
    print(f"[SETUP] stocks={len(panel)} dates={len(dates)} completed={len(dates)-len(pending)} pending={len(pending)}", flush=True)
    if not pending:
        print(json.dumps(summarize_outputs(out, c), ensure_ascii=False, indent=2), flush=True)
        return
    seed_everything(c.seed)
    predictor, torch = load_predictor(c)
    started = time.time()
    for date_no, date in enumerate(dates, 1):
        final_path = by_date / f"{date.date().isoformat()}.csv"
        if not args.force_recompute and valid_completed_file(final_path, date, c):
            print(f"[SKIP] {date.date()} valid completed file", flush=True)
            continue
        if args.force_recompute:
            final_path.unlink(missing_ok=True)
            (partial_dir / f"{date.date().isoformat()}.csv").unlink(missing_ok=True)
        items, reasons = prepare_date(panel, date, c)
        status = {"updated_utc": utc_now(), "date_index": date_no, "total_dates": len(dates),
                  "as_of_date": date.date().isoformat(), "candidate_stocks": len(items),
                  "rejections": reasons, "elapsed_seconds": time.time() - started}
        atomic_write_json(out / "progress.json", status)
        if len(items) < c.min_cross_section:
            record_error(errors_path, {"time_utc": utc_now(), "as_of_date": date.date().isoformat(), "stock": "*",
                                      "stage": "prepare_date", "error": f"only {len(items)} valid candidates", "traceback": ""})
            print(f"[DATE-SKIP] {date.date()} candidates={len(items)} reasons={reasons}", flush=True)
            continue
        partial_path = partial_dir / f"{date.date().isoformat()}.csv"
        rows: list[dict[str, Any]] = []
        if partial_path.exists() and not args.force_recompute:
            try:
                partial_df = pd.read_csv(partial_path)
                if partial_df["stock"].is_unique:
                    rows = partial_df.to_dict("records")
            except Exception:
                rows = []
        done = {str(r["stock"]) for r in rows}
        item_by_stock = {x["stock"]: x for x in items}
        chunks = [items[i:i+c.batch_size] for i in range(0, len(items), c.batch_size)]
        for batch_index, chunk in enumerate(chunks):
            todo = [x for x in chunk if x["stock"] not in done]
            if not todo:
                continue
            bseed = stable_seed(c.seed, date.date().isoformat(), str(batch_index), ",".join(x["stock"] for x in chunk))
            seed_everything(bseed)
            dfs = [x["context"][PRICE_COLS + ["volume"]].reset_index(drop=True) for x in todo]
            xts = [pd.Series(x["context"].index).reset_index(drop=True) for x in todo]
            yts = [pd.Series(x["future"].index).reset_index(drop=True) for x in todo]
            try:
                preds = predictor.predict_batch(dfs, xts, yts, pred_len=c.pred_len, T=c.temperature,
                                                top_k=c.top_k, top_p=c.top_p, sample_count=c.sample_count, verbose=False)
                for item, pred in zip(todo, preds):
                    rows.append(make_row(item, pred, date, c, batch_index)); done.add(item["stock"])
                atomic_write_csv(partial_path, pd.DataFrame(rows).sort_values("stock"))
                status.update({"updated_utc": utc_now(), "completed_in_date": len(rows), "batch_index": batch_index,
                               "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0})
                atomic_write_json(out / "progress.json", status)
                print(f"[BATCH] {date.date()} {len(rows)}/{len(items)}", flush=True)
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for item in todo:
                    try:
                        sseed = stable_seed(c.seed, date.date().isoformat(), item["stock"])
                        seed_everything(sseed)
                        pred = predictor.predict(item["context"][PRICE_COLS + ["volume"]].reset_index(drop=True),
                                                 pd.Series(item["context"].index).reset_index(drop=True),
                                                 pd.Series(item["future"].index).reset_index(drop=True),
                                                 pred_len=c.pred_len, T=c.temperature, top_k=c.top_k,
                                                 top_p=c.top_p, sample_count=c.sample_count, verbose=False)
                        rows.append(make_row(item, pred, date, c, batch_index)); done.add(item["stock"])
                        atomic_write_csv(partial_path, pd.DataFrame(rows).sort_values("stock"))
                    except Exception as single_e:
                        record_error(errors_path, {"time_utc": utc_now(), "as_of_date": date.date().isoformat(),
                            "stock": item["stock"], "stage": "predict", "error": repr(single_e),
                            "batch_error": repr(e), "traceback": traceback.format_exc(limit=8)})
                        print(f"[ERROR] {date.date()} {item['stock']}: {single_e}", flush=True)
        result = pd.DataFrame(rows).drop_duplicates("stock", keep="last").sort_values("stock") if rows else pd.DataFrame()
        if len(result) >= c.min_cross_section:
            atomic_write_csv(final_path, result)
            partial_path.unlink(missing_ok=True)
            metric = daily_metrics(result, c)
            print(f"[DATE-DONE] {date.date()} n={len(result)} RankIC={metric['spearman_rankic_path_mean']:.6f}", flush=True)
        else:
            print(f"[DATE-INCOMPLETE] {date.date()} successful={len(result)} required={c.min_cross_section}", flush=True)
        summarize_outputs(out, c)
    summary = summarize_outputs(out, c)
    status = {"updated_utc": utc_now(), "status": "finished_selected_dates", "elapsed_seconds": time.time()-started,
              "completed_dates": summary.get("completed_dates", 0), "selected_dates": len(dates)}
    atomic_write_json(out / "progress.json", status)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()



