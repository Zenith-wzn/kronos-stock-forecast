#!/usr/bin/env python
"""Build a new frozen Kronos hidden cache for the continual-learning ABC experiment.

The cache is independent of every existing zero-shot/Adapter cache.  Kronos is
frozen here: only hidden states are extracted, never trained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def stamps(dates: pd.Index) -> np.ndarray:
    d = pd.Series(pd.to_datetime(dates))
    return pd.DataFrame({"minute": d.dt.minute, "hour": d.dt.hour, "weekday": d.dt.weekday,
                         "day": d.dt.day, "month": d.dt.month}).to_numpy(np.float32)


def normalise_ohlcv(frame: pd.DataFrame) -> np.ndarray:
    cols = ["open", "high", "low", "close", "volume", "amount"]
    x = frame[cols].to_numpy(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return np.clip((x - mean) / (std + 1e-5), -5, 5).astype(np.float32)


def load_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        if "amount" not in frame.columns:
            frame["amount"] = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
        else:
            fallback = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(fallback)
        if "code" in frame.columns and frame["code"].notna().any():
            raw_code = frame["code"].dropna().astype(str).iloc[-1].strip()
        elif "ticker" in frame.columns and frame["ticker"].notna().any():
            raw_code = frame["ticker"].dropna().astype(str).iloc[-1].strip()
        else:
            raw_code = path.stem
        if path.parent.name.lower() == "yahoo_sp500":
            if raw_code.lower().startswith("us."):
                code = "us." + raw_code.split(".", 1)[1]
            elif "ticker" in frame.columns and frame["ticker"].notna().any():
                code = "us." + frame["ticker"].dropna().astype(str).iloc[-1].strip()
            elif raw_code.lower().startswith("us_"):
                code = "us." + raw_code.split("_", 1)[1]
            elif path.stem.lower().startswith("us_"):
                code = "us." + path.stem.split("_", 1)[1]
            else:
                code = "us." + raw_code
        else:
            code = raw_code
        frame["code"] = code
        frames[code] = frame.set_index("date", drop=False)
    if not frames:
        raise RuntimeError(f"no usable stock files under {root}")
    return frames


def load_predictor(args):
    repo = str(Path(args.kronos_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model import Kronos, KronosPredictor, KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer_name)
    model = Kronos.from_pretrained(args.model_name)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
    model.eval()
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return predictor


def hidden_state(predictor, x: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    tokens = predictor.tokenizer.encode(torch.as_tensor(x, device=predictor.device), half=True)
    _, context = predictor.model.decode_s1(tokens[0], tokens[1], torch.as_tensor(timestamps, device=predictor.device))
    return context[:, -1, :]


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--stocks-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--kronos-repo", required=True)
    p.add_argument("--model-name", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer-name", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    p.add_argument("--max-stocks", type=int)
    p.add_argument("--max-train-dates", type=int)
    p.add_argument("--max-val-dates", type=int)
    p.add_argument("--max-test-dates", type=int)
    p.add_argument("--smoke-test-schedule", action="store_true")
    a = p.parse_args()

    out = Path(a.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pair = pd.read_csv(Path(a.panel_dir) / "eligible_stock_date_pairs.csv")
    windows = pd.read_csv(Path(a.panel_dir) / "date_windows.csv")
    summary_path = Path(a.panel_dir) / "panel_summary.json"
    if summary_path.exists():
        panel_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        lookback = int(panel_summary["lookback"])
        pred_len = int(panel_summary["pred_len"])
    else:
        first_window = windows.iloc[0]
        lookback = len(str(first_window["input_dates"]).split("|"))
        pred_len = len(str(first_window["label_dates"]).split("|"))
    if lookback < 1 or pred_len < 1:
        raise ValueError(f"invalid panel window lengths: {lookback}/{pred_len}")
    frames = load_frames(Path(a.stocks_dir))
    predictor = load_predictor(a)
    d_model = int(predictor.model.d_model)

    generated_rows: dict[str, int] = {}
    try:
        for split in a.splits:
            meta = pair[pair["split"] == split].copy()
            if a.max_stocks is not None:
                keep_stocks = sorted(meta["stock"].astype(str).unique())[:a.max_stocks]
                meta = meta[meta["stock"].astype(str).isin(keep_stocks)].copy()
            date_limit = a.max_train_dates if split == "train" else a.max_val_dates if split == "val" else a.max_test_dates if split == "test" else None
            if date_limit is not None:
                all_split_dates = sorted(meta["as_of_date"].astype(str).unique())
                keep_dates = all_split_dates[:date_limit] if split == "test" else all_split_dates[-date_limit:]
                meta = meta[meta["as_of_date"].astype(str).isin(keep_dates)].copy()
            if split == "test" and a.smoke_test_schedule:
                all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(meta["as_of_date"]).dt.normalize().unique()))
                periods = all_dates.to_period("M")
                unique_periods = periods.unique().sort_values()
                if len(unique_periods) < 3:
                    raise RuntimeError("smoke test cache needs at least three calendar months")
                first_month = all_dates[periods == unique_periods[0]]
                second_month = all_dates[periods == unique_periods[1]]
                third_month = all_dates[periods == unique_periods[2]]
                # Keep exactly the boundary dates consumed by the runner smoke schedule.
                smoke_dates = pd.DatetimeIndex([
                    first_month[-1], second_month[0], second_month[-1], third_month[0]
                ]).unique()
                meta = meta[pd.to_datetime(meta["as_of_date"]).dt.normalize().isin(smoke_dates)].copy()
            if meta.empty:
                raise RuntimeError(f"panel split is empty: {split}")
            meta = meta.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
            window = windows[windows["split"] == split].drop_duplicates("as_of_date").set_index("as_of_date")
            meta_path = out / f"meta_{split}.csv"
            n = len(meta)
            generated_rows[split] = int(n)
            hidden_path = out / f"hidden_{split}.npy"
            labels_path = out / f"labels_{split}.npy"
            current_path = out / f"current_{split}.npy"
            completed_path = out / f"progress_{split}.json"
            any_array_exists = hidden_path.exists() or labels_path.exists() or current_path.exists()
            paths_exist = hidden_path.exists() and labels_path.exists() and current_path.exists()
            if any_array_exists and not paths_exist:
                raise RuntimeError(f"cache arrays are incomplete for {split}")
            if completed_path.exists() and not paths_exist:
                raise RuntimeError(f"progress exists but cache arrays are incomplete for {split}")
            if paths_exist:
                if not meta_path.exists():
                    raise RuntimeError(f"cache arrays exist but metadata is missing for {split}")
                old_meta = pd.read_csv(meta_path)
                compare_cols = ["split", "as_of_date", "stock"]
                if len(old_meta) != len(meta) or not old_meta[compare_cols].astype(str).equals(meta[compare_cols].astype(str)):
                    raise RuntimeError(f"existing metadata differs from requested cache rows for {split}")
                hidden = np.load(hidden_path, mmap_mode="r+")
                labels = np.load(labels_path, mmap_mode="r+")
                current = np.load(current_path, mmap_mode="r+")
                expected_shapes = ((n, d_model), (n, pred_len), (n,))
                actual_shapes = (hidden.shape, labels.shape, current.shape)
                if actual_shapes != expected_shapes:
                    raise RuntimeError(
                        f"existing cache shapes differ for {split}: actual={actual_shapes} expected={expected_shapes}"
                    )
            else:
                meta.to_csv(meta_path, index=False)
                hidden = np.lib.format.open_memmap(hidden_path, mode="w+", dtype="float32", shape=(n, d_model))
                labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype="float32", shape=(n, pred_len))
                current = np.lib.format.open_memmap(current_path, mode="w+", dtype="float32", shape=(n,))
            completed = set(json.loads(completed_path.read_text(encoding="utf-8")).get("completed_dates", [])) if completed_path.exists() else set()
            grouped = meta.groupby("as_of_date", sort=True)
            total_dates = meta["as_of_date"].nunique()
            for number, (date, group) in enumerate(grouped, 1):
                if str(date) in completed:
                    continue
                if date not in window.index:
                    raise KeyError(f"missing date window for {split}/{date}")
                w = window.loc[date]
                input_dates = pd.to_datetime(str(w["input_dates"]).split("|"))
                label_dates = pd.to_datetime(str(w["label_dates"]).split("|"))
                indices = group.index.to_numpy(dtype=np.int64)
                stock_codes = group["stock"].astype(str).tolist()
                for start in range(0, len(stock_codes), a.batch_size):
                    batch_stocks = stock_codes[start:start + a.batch_size]
                    xs, ts, ys, closes = [], [], [], []
                    for stock in batch_stocks:
                        frame = frames[stock]
                        context = frame.loc[input_dates]
                        future = frame.loc[label_dates]
                        if len(context) != lookback or len(future) != pred_len:
                            raise ValueError(f"bad {lookback}/{pred_len} window for {stock}/{date}")
                        close_t = float(context["close"].iloc[-1])
                        xs.append(normalise_ohlcv(context))
                        ts.append(stamps(input_dates))
                        ys.append((future["close"].to_numpy(np.float32) / close_t - 1).astype(np.float32))
                        closes.append(close_t)
                    with torch.no_grad():
                        h = hidden_state(predictor, np.stack(xs), np.stack(ts)).float().cpu().numpy()
                    target_indices = indices[start:start + len(batch_stocks)]
                    hidden[target_indices] = h
                    labels[target_indices] = np.stack(ys)
                    current[target_indices] = np.asarray(closes, np.float32)
                hidden.flush(); labels.flush(); current.flush()
                completed.add(str(date))
                atomic_json(completed_path, {"status": "running", "split": split, "completed_dates": sorted(completed), "completed_count": len(completed), "total_dates": int(total_dates), "rows": n, "d_model": d_model, "elapsed_seconds": time.time() - started})
                print(f"[CONTINUAL CACHE {split}] {number}/{total_dates} date={date} rows={len(group)}", flush=True)
            atomic_json(completed_path, {"status": "complete", "split": split, "completed_dates": sorted(completed), "completed_count": len(completed), "total_dates": int(total_dates), "rows": n, "d_model": d_model, "elapsed_seconds": time.time() - started})
        atomic_json(out / "cache_summary.json", {"status": "complete", "experiment_type": "continual_learning_ranking_head", "d_model": d_model, "lookback": lookback, "pred_len": pred_len, "batch_size": a.batch_size, "generated_splits": list(a.splits), "generated_rows": generated_rows, "full_panel_rows": {split: int((pair["split"] == split).sum()) for split in ("train", "val", "test")}, "elapsed_seconds": time.time() - started})
    except Exception as exc:
        atomic_json(out / "FAILED.json", {"status": "failed", "experiment_type": "continual_learning_ranking_head", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()


