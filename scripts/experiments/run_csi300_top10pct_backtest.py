from __future__ import annotations

"""Rebuild CSI300 page-1..3 portfolio results with the Top10% (Top30/Drop3) rule.

This script deliberately reads only stock-level prediction artifacts and reconstructs
portfolio returns; it never infers daily returns from aggregate metrics.
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    "dataset": "CSI300",
    "train_end": "2024-12-15",
    "val_start": "2025-01-02",
    "val_end": "2025-06-14",
    "test_start": "2025-07-01",
    "test_end": "2026-06-05",
    "lookback": 90,
    "pred_len": 10,
    "seed": 100,
    "top_k": 30,
    "top_k_fraction": 0.10,
    "drop_n": 3,
    "min_hold_days": 5,
    "cost_rate": 0.0015,
    "annualization_days": 252,
    "portfolio_cost_definition": "cost_rate times one-way weight turnover",
    "constituent_pool": "current CSI300 constituents backfilled over available history",
    "bias_warning": "survivorship bias and current-constituent look-ahead bias are possible; not point-in-time membership",
}

PAGE_METHODS = {
    "page1_abcd_top30_drop3": {
        "A_zero_shot": {"page": "页面一：ABCD 静态对照", "label": "页面一 A：Kronos zero-shot", "kind": "static_a"},
        "B_linear_head": {"page": "页面一：ABCD 静态对照", "label": "页面一 B：静态训练排序头", "kind": "static"},
        "C_prediction_adapter": {"page": "页面一：ABCD 静态对照", "label": "页面一 C：静态 Adapter/预测头", "kind": "static"},
        "D_ranking_adapter": {"page": "页面一：ABCD 静态对照", "label": "页面一 D：静态 Adapter+排序头", "kind": "static"},
        "Kronos_LoRA": {"page": "页面一：ABCD 静态对照", "label": "页面一：Kronos LoRA（partial cache pilot）", "kind": "static"},
    },
    "page2_continual_abc_top30_drop3": {
        "A_continual": {"page": "页面二：Continual A/B/C", "label": "页面二 A：Continual zero-shot", "kind": "continual"},
        "B_continual": {"page": "页面二：Continual A/B/C", "label": "页面二 B：Continual 冻结排序头", "kind": "continual"},
        "C_continual": {"page": "页面二：Continual A/B/C", "label": "页面二 C：Continual 持续更新", "kind": "continual"},
    },
    "page3_adapter_continual_top30_drop3": {
        "Adapter_continual": {"page": "页面三：Adapter + Continual", "label": "页面三：Adapter + Continual", "kind": "adapter"},
    },
}


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_json(v):
    if isinstance(v, dict):
        return {str(k): clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"failed reading {path}: {exc}") from exc
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_frame(df: pd.DataFrame, *, method: str, source: str, kind: str, test_start: str, test_end: str, return_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = next((c for c in ("as_of_date", "date", "signal_date") if c in df.columns), None)
    stock_col = next((c for c in ("stock", "code", "symbol") if c in df.columns), None)
    if not date_col or not stock_col:
        raise ValueError(f"{method}: missing date/stock columns in {source}")
    pred_col = next((c for c in ("pred_path_mean_return", "pred_signal", "raw_signal", "pred_endpoint_return") if c in df.columns), None)
    if not pred_col:
        raise ValueError(f"{method}: no prediction signal column in {source}")
    ret_col = next((c for c in ("daily_return_1", "actual_return_1") if c in df.columns), None)
    if ret_col is None and {"actual_close_1", "current_close"}.issubset(df.columns):
        ret = pd.to_numeric(df["actual_close_1"], errors="coerce") / pd.to_numeric(df["current_close"], errors="coerce") - 1.0
    elif ret_col is not None:
        ret = pd.to_numeric(df[ret_col], errors="coerce")
    else:
        # Continual A/B/C stores actual_signal as a 10-day path target, not a
        # one-day tradable return. Join the shared one-step realized return by
        # (date, stock) instead of silently misusing actual_signal.
        ret = pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.DataFrame({
        "as_of_date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "stock": df[stock_col].astype(str).str.strip(),
        "pred_signal": pd.to_numeric(df[pred_col], errors="coerce"),
        "actual_return_1": ret,
        "source_file": source,
    })
    if out["actual_return_1"].isna().any() and return_reference is not None:
        ref = return_reference[["as_of_date", "stock", "actual_return_1"]].drop_duplicates(["as_of_date", "stock"])
        out = out.drop(columns=["actual_return_1"]).merge(ref, on=["as_of_date", "stock"], how="left", validate="many_to_one")
    if out["actual_return_1"].notna().sum() == 0:
        raise ValueError(f"{method}: no strictly reconstructable one-step actual return in {source}")
    out = out[(out["as_of_date"] >= pd.Timestamp(test_start)) & (out["as_of_date"] <= pd.Timestamp(test_end))]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["as_of_date", "stock", "pred_signal", "actual_return_1"])
    out = out[out["stock"].ne("")].copy()
    # Multiple copies can exist in a source directory. Keep the first valid row per key.
    out = out.sort_values(["as_of_date", "stock", "source_file"]).drop_duplicates(["as_of_date", "stock"], keep="first")
    out["method"] = method
    out["kind"] = kind
    return out.sort_values(["as_of_date", "stock"]).reset_index(drop=True)


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    a = pd.Series(x).rank(method="average").to_numpy(float)
    b = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def max_drawdown(wealth):
    w = np.asarray(wealth, dtype=float)
    if len(w) == 0:
        return np.nan
    peak = np.maximum.accumulate(w)
    return float(np.min(w / peak - 1.0))


@dataclass
class PortfolioState:
    positions: dict[str, int]


def simulate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "reason": "no valid prediction rows"}
    dates = sorted(pd.to_datetime(df["as_of_date"]).dt.normalize().unique())
    k = int(cfg["top_k"])
    drop_n = int(cfg["drop_n"])
    min_hold = int(cfg["min_hold_days"])
    cost_rate = float(cfg["cost_rate"])
    state = PortfolioState(positions={})
    daily_rows, holding_rows = [], []
    for di, date_value in enumerate(dates):
        date = pd.Timestamp(date_value).normalize()
        g = df[df["as_of_date"].eq(date)].copy()
        g = g.sort_values(["pred_signal", "stock"], ascending=[False, True]).drop_duplicates("stock")
        if g.empty:
            continue
        universe = set(g["stock"])
        target = list(g.head(min(k, len(g)))["stock"])
        target_set = set(target)
        old_set = set(state.positions)
        forced_sell = {s for s in old_set if s not in universe}
        ordinary_sellable = [s for s in old_set if s in universe and s not in target_set and di - state.positions[s] >= min_hold]
        # Lowest current scores leave first. Missing names are always forced out.
        score = dict(zip(g["stock"], g["pred_signal"]))
        ordinary_sellable.sort(key=lambda s: (score.get(s, -np.inf), s))
        sell = forced_sell | set(ordinary_sellable[:drop_n])
        kept = old_set - sell
        buy_candidates = [s for s in target if s not in kept]
        # Initial portfolio is filled to Top30; later ordinary rotation is capped by the number sold.
        if not old_set:
            buy = set(target)
        else:
            buy = set(buy_candidates[:len(sell)])
            # If a previous forced/missing holding reduced the book, fill the remaining slots.
            needed = max(0, min(k, len(g)) - len(kept) - len(buy))
            if needed:
                buy.update([s for s in target if s not in kept and s not in buy][:needed])
        new_set = kept | buy
        # Ensure a deterministic, at-most-k portfolio if data anomalies create too many positions.
        if len(new_set) > min(k, len(g)):
            new_set = set(g[g["stock"].isin(new_set)].head(min(k, len(g)))["stock"])
        old_n = len(old_set)
        new_n = len(new_set)
        old_w = 1.0 / old_n if old_n else 0.0
        new_w = 1.0 / new_n if new_n else 0.0
        turnover = 1.0 if not old_set and new_set else 0.5 * sum(abs((new_w if s in new_set else 0.0) - (old_w if s in old_set else 0.0)) for s in old_set | new_set)
        bought = sorted(new_set - old_set)
        sold = sorted(old_set - new_set)
        # age is an integer trading-day counter; a buy on date di has age 0.
        state.positions = {s: (state.positions[s] if s in state.positions and s in new_set else di) for s in new_set}
        held = g[g["stock"].isin(new_set)]
        gross = float(held["actual_return_1"].mean()) if not held.empty else np.nan
        benchmark = float(g["actual_return_1"].mean()) if not g.empty else np.nan
        trans_cost = cost_rate * turnover
        net = gross - trans_cost if math.isfinite(gross) else np.nan
        rank_ic = spearman(g["pred_signal"].to_numpy(float), g["actual_return_1"].to_numpy(float))
        for s in sorted(new_set):
            row = g[g["stock"].eq(s)]
            holding_rows.append({
                "as_of_date": date.date().isoformat(), "stock": s,
                "signal": finite(row["pred_signal"].iloc[0]) if not row.empty else None,
                "actual_return_1": finite(row["actual_return_1"].iloc[0]) if not row.empty else None,
                "holding_age_trading_days": di - state.positions[s],
                "held": True, "bought_today": s in bought, "sold_today": False,
            })
        for s in sold:
            holding_rows.append({"as_of_date": date.date().isoformat(), "stock": s, "signal": None, "actual_return_1": None, "holding_age_trading_days": None, "held": False, "bought_today": False, "sold_today": True})
        daily_rows.append({
            "as_of_date": date.date().isoformat(), "n_universe": int(len(g)), "n_holdings": int(len(new_set)),
            "target_k": int(min(k, len(g))), "gross_return": gross, "transaction_cost": trans_cost,
            "net_return": net, "benchmark_ew_return": benchmark, "turnover": float(turnover),
            "bought": int(len(bought)), "sold": int(len(sold)), "forced_sold_missing": int(len(forced_sell)),
            "rank_ic": rank_ic, "missing_held_rows": int(len(new_set) - len(held)),
        })
    daily = pd.DataFrame(daily_rows)
    holdings = pd.DataFrame(holding_rows)
    if daily.empty:
        return daily, holdings, {"available": False, "reason": "no valid daily portfolio periods"}
    daily["net_return"] = pd.to_numeric(daily["net_return"], errors="coerce")
    daily["wealth"] = (1.0 + daily["net_return"].fillna(0.0)).cumprod()
    daily["cumulative_return"] = daily["wealth"] - 1.0
    daily["benchmark_wealth"] = (1.0 + daily["benchmark_ew_return"].fillna(0.0)).cumprod()
    daily["benchmark_cumulative_return"] = daily["benchmark_wealth"] - 1.0
    n = len(daily)
    wealth_final = float(daily["wealth"].iloc[-1])
    bench_final = float(daily["benchmark_wealth"].iloc[-1])
    summary = {
        "available": True, "n_periods": int(n), "test_start": daily["as_of_date"].iloc[0], "test_end": daily["as_of_date"].iloc[-1],
        "n_universe_min": int(daily["n_universe"].min()), "n_universe_max": int(daily["n_universe"].max()),
        "n_holdings_min": int(daily["n_holdings"].min()), "n_holdings_max": int(daily["n_holdings"].max()),
        "final_wealth": wealth_final, "final_cumulative_return": wealth_final - 1.0,
        "annualized_return_252": wealth_final ** (252.0 / n) - 1.0 if wealth_final > 0 else np.nan,
        "max_drawdown": max_drawdown(daily["wealth"]), "mean_daily_gross_return": float(daily["gross_return"].mean()),
        "mean_daily_net_return": float(daily["net_return"].mean()), "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "mean_turnover": float(daily["turnover"].mean()), "total_turnover": float(daily["turnover"].sum()),
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "benchmark_final_cumulative_return": bench_final - 1.0,
        "benchmark_annualized_return_252": bench_final ** (252.0 / n) - 1.0 if bench_final > 0 else np.nan,
        "benchmark_max_drawdown": max_drawdown(daily["benchmark_wealth"]),
    }
    return daily, holdings, summary


def discover_sources(root: Path, method_key: str) -> tuple[pd.DataFrame, dict]:
    exp = root / "train3" / "csi300_csi800_aligned_seed100" / "experiments"
    if method_key == "A_zero_shot":
        files = sorted((exp / "A_zero_shot" / "predictions" / "by_date").glob("*.csv"))
    elif method_key in {"B_linear_head", "C_prediction_adapter", "D_ranking_adapter"}:
        files = [exp / method_key / "predictions.csv"]
    elif method_key == "Kronos_LoRA":
        files = [root / "kronos_lora_csi300_90x10_pilot_partial_seed100" / "predictions.csv"]
    elif method_key == "A_continual":
        files = [exp / "continual_ABC" / "predictions_A.csv"]
    elif method_key == "B_continual":
        files = [exp / "continual_ABC" / "predictions_B.csv"]
    elif method_key == "C_continual":
        files = [exp / "continual_ABC" / "predictions_C.csv"]
    elif method_key == "Adapter_continual":
        files = [exp / "adapter_continual" / "predictions.csv"]
    else:
        raise KeyError(method_key)
    files = [p for p in files if p.exists()]
    if not files:
        return pd.DataFrame(), {"available": False, "reason": "prediction files not found", "files": []}
    return read_csvs(files), {"available": True, "files": [str(p) for p in files]}


def write_json(path: Path, obj):
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="project root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-start", default=DEFAULT_CONFIG["test_start"])
    ap.add_argument("--test-end", default=DEFAULT_CONFIG["test_end"])
    ap.add_argument("--top-k", type=int, default=DEFAULT_CONFIG["top_k"])
    ap.add_argument("--drop-n", type=int, default=DEFAULT_CONFIG["drop_n"])
    ap.add_argument("--min-hold-days", type=int, default=DEFAULT_CONFIG["min_hold_days"])
    ap.add_argument("--cost-rate", type=float, default=DEFAULT_CONFIG["cost_rate"])
    args = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG, test_start=args.test_start, test_end=args.test_end, top_k=args.top_k, drop_n=args.drop_n, min_hold_days=args.min_hold_days, cost_rate=args.cost_rate)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {**cfg, "generated_at": datetime.now(timezone.utc).isoformat(), "source_policy": "strict stock-level reconstruction; no aggregate back-solving"})
    report = {"config": cfg, "methods": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    # A single shared realized-return table makes all pages use identical
    # date/stock returns. Static B contains the full 10-step realized path.
    ref_raw, ref_meta = discover_sources(args.root, "B_linear_head")
    return_reference = normalize_frame(ref_raw, method="return_reference", source=";".join(ref_meta.get("files", [])), kind="reference", test_start=args.test_start, test_end=args.test_end)
    return_reference = return_reference[["as_of_date", "stock", "actual_return_1"]]
    report["return_reference"] = {"source": ref_meta, "rows": int(len(return_reference)), "definition": "actual_return_1 joined by as_of_date and stock"}
    for page_dir, methods in PAGE_METHODS.items():
        page_out = out / page_dir
        page_out.mkdir(parents=True, exist_ok=True)
        for method_key, spec in methods.items():
            raw, source_meta = discover_sources(args.root, method_key)
            method_out = page_out / method_key
            method_out.mkdir(parents=True, exist_ok=True)
            item = {"label": spec["label"], "page": spec["page"], "source": source_meta}
            try:
                frame = normalize_frame(raw, method=method_key, source=";".join(source_meta.get("files", [])), kind=spec["kind"], test_start=args.test_start, test_end=args.test_end, return_reference=return_reference)
                frame.to_csv(method_out / "predictions_normalized.csv", index=False, encoding="utf-8-sig")
                daily, holdings, summary = simulate(frame, cfg)
                daily.to_csv(method_out / "daily_returns.csv", index=False, encoding="utf-8-sig")
                holdings.to_csv(method_out / "portfolio_holdings.csv", index=False, encoding="utf-8-sig")
                write_json(method_out / "summary.json", {**summary, "method": method_key, "label": spec["label"], "page": spec["page"], "source_files": source_meta.get("files", []), "reconstruction": "daily next-step return from stock-level actual_return_1/actual_signal or actual_close_1/current_close"})
                write_json(method_out / "run_config.json", {**cfg, "method": method_key, "label": spec["label"], "page": spec["page"]})
                item.update({"status": "complete", "rows": int(len(frame)), "summary": summary})
            except Exception as exc:
                item.update({"status": "failed", "error": repr(exc)})
                write_json(method_out / "FAILED.json", item)
            report["methods"][method_key] = item
    # Combined files are convenient for the curve generator and downstream papers.
    combined_daily = []
    combined_summary = []
    for page_dir, methods in PAGE_METHODS.items():
        for method_key, spec in methods.items():
            method_out = out / page_dir / method_key
            p = method_out / "daily_returns.csv"
            s = method_out / "summary.json"
            if p.exists() and s.exists():
                d = pd.read_csv(p); d.insert(0, "method", method_key); d.insert(0, "page_dir", page_dir); combined_daily.append(d)
                x = json.loads(s.read_text(encoding="utf-8")); x.update({"method": method_key, "page_dir": page_dir, "label": spec["label"], "page": spec["page"]}); combined_summary.append(x)
    if combined_daily:
        pd.concat(combined_daily, ignore_index=True).to_csv(out / "all_daily_returns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(combined_summary).to_csv(out / "all_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "summary.json", report)
    print(json.dumps(clean_json(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





