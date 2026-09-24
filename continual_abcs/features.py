"""Feature, label, and cache utilities for the frozen-Kronos A/B/C study."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
KRONOS_COLUMNS = PRICE_COLUMNS + ("amount",)


def _dates(values: Sequence[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise").normalize())


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" not in out.columns:
        out["date"] = out.index
    out = out.reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" not in out.columns:
        out["amount"] = out[list(PRICE_COLUMNS[:4])].mean(axis=1) * out["volume"]
    else:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
        out["amount"] = out["amount"].fillna(out[list(PRICE_COLUMNS[:4])].mean(axis=1) * out["volume"])
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.set_index("date", drop=False)


def make_kronos_input(frame: pd.DataFrame, input_dates: Sequence[object]) -> pd.DataFrame:
    """Return exactly the requested historical rows; never forward-fill future data."""
    dates = _dates(input_dates)
    if len(dates) == 0 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("input_dates must be non-empty, unique and increasing")
    out = _normalise(frame)
    if not dates.isin(out.index).all():
        missing = [str(x.date()) for x in dates if x not in out.index]
        raise ValueError(f"input dates missing: {missing}")
    selected = out.loc[dates, list(KRONOS_COLUMNS)].copy()
    if len(selected) != len(dates) or not selected.index.equals(dates):
        raise ValueError("input dates are not an exact ordered slice")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("input contains invalid OHLCV values")
    return selected.reset_index(drop=True)


def make_target(frame: pd.DataFrame, label_dates: Sequence[object]) -> np.ndarray:
    """Build Eq.13: mean future close / close at the end of the input window - 1."""
    dates = _dates(label_dates)
    if len(dates) == 0 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("label_dates must be non-empty, unique and increasing")
    out = _normalise(frame)
    if not dates.isin(out.index).all():
        raise ValueError("label dates missing from frame")
    start_pos = out.index.get_indexer([dates[0]])[0]
    if start_pos <= 0:
        raise ValueError("cannot infer as-of close before label window")
    if not out.index[start_pos - 1] < dates[0]:
        raise ValueError("label window is not after the as-of date")
    close_t = float(out.iloc[start_pos - 1]["close"])
    future = out.loc[dates, "close"].to_numpy(dtype=float)
    if not np.isfinite(close_t) or close_t <= 0 or not np.isfinite(future).all() or (future <= 0).any():
        raise ValueError("invalid close values for target")
    return np.asarray([future.mean() / close_t - 1.0], dtype=np.float32)


def build_signal_summary(prediction_frame: pd.DataFrame, current_close: float | None = None) -> np.ndarray:
    """Summarise a Kronos path and expose the Eq.13 raw signal as element zero."""
    out = prediction_frame.copy()
    close_col = "pred_close" if "pred_close" in out.columns else "close"
    if close_col not in out.columns:
        raise ValueError("prediction_frame must contain close or pred_close")
    close = pd.to_numeric(out[close_col], errors="coerce").to_numpy(dtype=float)
    if current_close is None and "current_close" in out.columns:
        current_close = float(pd.to_numeric(out["current_close"], errors="coerce").iloc[0])
    if current_close is None or not np.isfinite(current_close) or current_close <= 0:
        raise ValueError("current_close is required and must be positive")
    if len(close) == 0 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("prediction close path is invalid")
    returns = close / current_close - 1.0
    return np.asarray(
        [returns.mean(), returns[-1], returns.std(ddof=0), returns.min(), returns.max(),
         np.diff(returns).mean() if len(returns) > 1 else 0.0], dtype=np.float32
    )


class FeatureCacheWriter:
    """Write a split cache atomically enough for resumable GPU jobs."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        split: str,
        metadata: pd.DataFrame,
        hidden: np.ndarray,
        target: np.ndarray,
        raw_signal: np.ndarray,
    ) -> None:
        split = str(split)
        hidden = np.asarray(hidden, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        raw_signal = np.asarray(raw_signal, dtype=np.float32).reshape(-1)
        n = len(metadata)
        if hidden.ndim != 2 or target.ndim != 2 or len(hidden) != n or len(target) != n or len(raw_signal) != n:
            raise ValueError("metadata and cache arrays must have the same row count")
        meta = metadata.reset_index(drop=True).copy()
        meta["raw_signal"] = raw_signal
        for name, arr in (("hidden", hidden), ("target", target)):
            path = self.cache_dir / f"{name}_{split}.npy"
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("wb") as fh:
                np.save(fh, arr)
            tmp.replace(path)
        meta.to_csv(self.cache_dir / f"meta_{split}.csv", index=False)


def load_feature_cache(cache_dir: str | Path, split: str, *, mmap: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    root = Path(cache_dir)
    hidden_path = root / f"hidden_{split}.npy"
    target_path = root / f"labels_{split}.npy"
    if not target_path.exists():
        target_path = root / f"target_{split}.npy"
    meta_path = root / f"meta_{split}.csv"
    if not hidden_path.exists() or not target_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"incomplete feature cache for split={split}")
    mode = "r" if mmap else None
    hidden = np.load(hidden_path, mmap_mode=mode)
    target = np.load(target_path, mmap_mode=mode)
    meta = pd.read_csv(meta_path)
    if len(hidden) != len(target) or len(hidden) != len(meta):
        raise ValueError(f"cache row mismatch for split={split}")
    return hidden, target, meta

