"""Independent CSI300 A/B/C continual-learning experiment runner.

This file is intentionally separate from the existing zero-shot and Adapter
experiments.  It consumes a frozen Kronos hidden-state cache and runs exactly
three groups:

A: raw Kronos zero-shot signal, no learned head.
B: one ranking head trained before the test period and frozen during test.
C: the same B checkpoint, then monthly test-then-train updates.  A prediction
   is emitted first on a selected month-end; only labels already mature on that
   date are used for the update; the accepted checkpoint is active next test day.

The script never imports or edits the existing Adapter/zero-shot runners.
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from continual_abcs.panel import month_end_update_dates
from continual_abcs.head import NumpyRankingHead, fit_numpy_ranking_head

TEST_START = "2025-07-01"
TEST_END = "2026-06-05"
LOOKBACK = 90
PRED_LEN = 10
BATCH_SIZE = 16


@dataclass
class Config:
    cache_dir: str
    output_dir: str
    raw_signal_file: str | None = None
    panel_dir: str | None = None
    train_val_cache_dir: str | None = None
    test_start: str = TEST_START
    test_end: str = TEST_END
    lookback: int = LOOKBACK
    pred_len: int = PRED_LEN
    batch_size: int = BATCH_SIZE
    epochs: int = 5
    update_epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 100
    device: str = "cuda:0"
    backend: str = "auto"
    # Strict MAE gate: a candidate may not weaken the current champion.
    validation_tolerance: float = 0.0
    max_test_dates: int | None = None
    smoke_test_schedule: bool = False
    max_stocks: int | None = None
    max_train_dates: int | None = None
    max_val_dates: int | None = None
    replay_max_rows: int | None = 120_000
    replay_recent_months: int = 24
    replay_recent_fraction: float = 0.70

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start).normalize()

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_end).normalize()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(root: Path, names: Iterable[str]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {list(names)} exists under {root}")


def _load_split(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    hidden_path = _first_existing(cache_dir, [f"hidden_{split}.npy"])
    target_path = _first_existing(cache_dir, [f"labels_{split}.npy", f"target_{split}.npy"])
    meta_path = _first_existing(cache_dir, [f"meta_{split}.csv"])
    hidden = np.load(hidden_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    if hidden.ndim != 2 or target.ndim not in (1, 2):
        raise ValueError(f"unexpected cache shapes for {split}: {hidden.shape}, {target.shape}")
    if len(hidden) != len(target) or len(hidden) != len(meta):
        raise ValueError(f"cache row mismatch for {split}")
    required = {"as_of_date", "stock"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"meta_{split}.csv missing {sorted(missing)}")
    meta = meta.copy()
    meta["as_of_date"] = pd.to_datetime(meta["as_of_date"], errors="raise").dt.normalize()
    meta["stock"] = meta["stock"].astype(str)
    meta["_row_id"] = np.arange(len(meta), dtype=np.int64)
    return hidden, target, meta


def _attach_label_end(meta: pd.DataFrame, panel_dir: str | None) -> pd.DataFrame:
    result = meta.copy()
    if "label_end_date" in result.columns:
        result["label_end_date"] = pd.to_datetime(result["label_end_date"], errors="raise").dt.normalize()
        return result
    if not panel_dir:
        raise ValueError("label_end_date is absent from cache metadata; --panel-dir is required")
    windows_path = Path(panel_dir) / "date_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"missing date windows: {windows_path}")
    windows = pd.read_csv(windows_path)
    windows["as_of_date"] = pd.to_datetime(windows["as_of_date"], errors="raise").dt.normalize()
    if "label_end_date" not in windows.columns:
        raise ValueError("date_windows.csv must contain label_end_date")
    windows["label_end_date"] = pd.to_datetime(windows["label_end_date"], errors="raise").dt.normalize()
    lookup = windows[["as_of_date", "label_end_date"]].drop_duplicates("as_of_date")
    result = result.drop(columns=["label_end_date"], errors="ignore").merge(lookup, on="as_of_date", how="left", validate="many_to_one")
    if result["label_end_date"].isna().any():
        raise ValueError("some cache rows have no label_end_date in date_windows.csv")
    return result


def _verify_cache_keys(meta: pd.DataFrame, panel_dir: str | None, split: str, *, exact: bool = True) -> None:
    """Verify cache keys against the locked panel; smoke runs may use a strict subset."""
    if not panel_dir:
        return
    pair_path = Path(panel_dir) / "eligible_stock_date_pairs.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"missing locked panel pairs: {pair_path}")
    pair = pd.read_csv(pair_path, usecols=["stock", "as_of_date", "split"])
    expected = pair[pair["split"] == split][["as_of_date", "stock"]].copy()
    expected["as_of_date"] = pd.to_datetime(expected["as_of_date"], errors="raise").dt.normalize()
    expected["stock"] = expected["stock"].astype(str)
    actual = meta[["as_of_date", "stock"]].copy()
    if expected.duplicated().any() or actual.duplicated().any():
        raise ValueError(f"duplicate stock/date keys while verifying {split} cache")
    expected = expected.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    actual = actual.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    exp_keys = set(map(tuple, expected.astype(str).to_numpy()))
    act_keys = set(map(tuple, actual.astype(str).to_numpy()))
    missing = exp_keys - act_keys
    extra = act_keys - exp_keys
    if extra or (exact and missing) or len(actual) == 0:
        raise ValueError(
            f"{split} cache keys are not valid for locked panel: exact={exact} "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing_examples={list(sorted(missing))[:5]} "
            f"extra_examples={list(sorted(extra))[:5]}"
        )


def _filter_split(
    hidden: np.ndarray,
    target: np.ndarray,
    meta: pd.DataFrame,
    *,
    stocks: set[str] | None = None,
    max_dates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    keep = np.ones(len(meta), dtype=bool)
    if stocks is not None:
        keep &= meta["stock"].isin(stocks).to_numpy()
    if max_dates is not None:
        dates = sorted(meta.loc[keep, "as_of_date"].unique())[-max_dates:]
        keep &= meta["as_of_date"].isin(dates).to_numpy()
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise ValueError("split filter removed every row")
    return np.asarray(hidden[idx]), np.asarray(target[idx]), meta.iloc[idx].reset_index(drop=True)


def _canonical_stock(value: object) -> str:
    text = str(value).strip().lower().replace("_", ".")
    if text.startswith(("sh", "sz")) and "." not in text and len(text) >= 8:
        text = text[:2] + "." + text[2:]
    return text


def _normalise_signal_file(path: Path, *, pred_len: int) -> pd.DataFrame:
    if path.is_dir():
        rolling = path / "rolling_predictions_all.csv"
        preferred = path / "predictions" / "by_date"
        if rolling.exists():
            files = [rolling]
        elif preferred.exists():
            files = sorted(preferred.glob("*.csv"))
        else:
            files = sorted(path.glob("*.csv")) or sorted(path.rglob("*.csv"))
        frames: list[pd.DataFrame] = []
        for file in files:
            try:
                frame = pd.read_csv(file)
            except Exception:
                continue
            columns = {str(c).strip() for c in frame.columns}
            has_direct = bool(columns & {"raw_signal", "pred_signal", "pred_path_mean_return", "pred_path_mean_return_mean"})
            has_path = {"pred_close", "current_close", "step"}.issubset(columns)
            if (columns & {"as_of_date", "date", "signal_date"}) and (columns & {"stock", "code", "symbol"}) and (has_direct or has_path):
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"no signal-bearing CSV files under {path}")
        raw = pd.concat(frames, ignore_index=True, sort=False)
    else:
        raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "model" in raw.columns:
        kronos = raw[raw["model"].astype(str).str.casefold() == "kronos"]
        if not kronos.empty:
            raw = kronos.copy()
    date_col = next((c for c in ("as_of_date", "date", "signal_date") if c in raw.columns), None)
    stock_col = next((c for c in ("stock", "code", "symbol") if c in raw.columns), None)
    if not date_col or not stock_col:
        raise ValueError("raw signal file needs date/as_of_date and stock/code")

    signal_col = next((c for c in ("raw_signal", "pred_signal", "pred_path_mean_return", "pred_path_mean_return_mean") if c in raw.columns), None)
    if signal_col:
        out = pd.DataFrame({
            "as_of_date": pd.to_datetime(raw[date_col], errors="coerce").dt.normalize(),
            "stock": raw[stock_col].map(_canonical_stock),
            "raw_signal": pd.to_numeric(raw[signal_col], errors="coerce"),
        }).dropna(subset=["as_of_date", "raw_signal"])
        out = out.drop_duplicates(["as_of_date", "stock"], keep="last")
    elif {"pred_close", "current_close", "step"}.issubset(raw.columns):
        work = pd.DataFrame({
            "as_of_date": pd.to_datetime(raw[date_col], errors="coerce").dt.normalize(),
            "stock": raw[stock_col].map(_canonical_stock),
            "step": pd.to_numeric(raw["step"], errors="coerce"),
            "pred_close": pd.to_numeric(raw["pred_close"], errors="coerce"),
            "current_close": pd.to_numeric(raw["current_close"], errors="coerce"),
        }).dropna()
        work = work[(work["step"] >= 1) & (work["step"] <= pred_len)].copy()
        work["step"] = work["step"].astype(int)
        if work.duplicated(["as_of_date", "stock", "step"]).any():
            raise ValueError("zero-shot rolling predictions contain duplicate date/stock/step rows")
        counts = work.groupby(["as_of_date", "stock"])["step"].nunique()
        bad = counts[counts != pred_len]
        if not bad.empty:
            raise ValueError(f"zero-shot paths must contain exactly {pred_len} steps per key; bad_examples={bad.head().to_dict()}")
        grouped = work.groupby(["as_of_date", "stock"], as_index=False).agg(
            pred_close_mean=("pred_close", "mean"), current_close=("current_close", "first")
        )
        if (grouped["current_close"] <= 0).any():
            raise ValueError("zero-shot current_close must be positive")
        grouped["raw_signal"] = grouped["pred_close_mean"] / grouped["current_close"] - 1.0
        out = grouped[["as_of_date", "stock", "raw_signal"]]
    else:
        raise ValueError("raw signal file needs a direct signal column or pred_close/current_close/step")
    if not np.isfinite(out["raw_signal"].to_numpy(dtype=float)).all():
        raise ValueError("raw signal contains non-finite values")
    return out

def _load_raw_signal(
    test_meta: pd.DataFrame,
    raw_signal_file: str | None,
    *,
    pred_len: int,
    strict_keys: bool = False,
) -> pd.Series:
    key = test_meta["as_of_date"].dt.strftime("%Y-%m-%d") + "|" + test_meta["stock"].map(_canonical_stock)
    if "raw_signal" in test_meta.columns:
        signal = pd.to_numeric(test_meta["raw_signal"], errors="coerce")
        if signal.notna().all():
            return pd.Series(signal.to_numpy(dtype=float), index=key)
    if not raw_signal_file:
        raise ValueError("A requires raw signals: add raw_signal to meta_test.csv or pass --raw-signal-file")
    raw = _normalise_signal_file(Path(raw_signal_file), pred_len=pred_len)
    expected_dates = set(test_meta["as_of_date"].dt.normalize())
    raw = raw[raw["as_of_date"].isin(expected_dates)].copy()
    raw_key = raw["as_of_date"].dt.strftime("%Y-%m-%d") + "|" + raw["stock"].map(_canonical_stock)
    if strict_keys:
        expected_keys = set(key.astype(str))
        actual_keys = set(raw_key.astype(str))
        if expected_keys != actual_keys:
            raise ValueError(
                "raw signal keys do not exactly match the formal test cache: "
                f"expected={len(expected_keys)} actual={len(actual_keys)} "
                f"missing_examples={sorted(expected_keys - actual_keys)[:5]} "
                f"extra_examples={sorted(actual_keys - expected_keys)[:5]}"
            )
    lookup = pd.Series(raw["raw_signal"].to_numpy(dtype=float), index=raw_key)
    values = key.map(lookup)
    if values.isna().any():
        missing = test_meta.loc[values.isna(), ["as_of_date", "stock"]].head(10).to_dict("records")
        raise ValueError(f"raw signal is missing for {int(values.isna().sum())} common test rows; examples={missing}")
    return pd.Series(values.to_numpy(dtype=float), index=key)


def _target_scalar(target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    return y if y.ndim == 1 else y.mean(axis=1)


def _mean_daily_rankic(scores: np.ndarray, target: np.ndarray, meta: pd.DataFrame) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    true = _target_scalar(target).reshape(-1)
    if len(scores) != len(true) or len(scores) != len(meta):
        raise ValueError("validation scores, labels, and metadata are not aligned")
    values: list[float] = []
    for _, group in meta.groupby("as_of_date", sort=True):
        ix = group.index.to_numpy(dtype=np.int64)
        if len(ix) < 2:
            continue
        pred_rank = pd.Series(scores[ix]).rank(method="average").to_numpy(dtype=float)
        true_rank = pd.Series(true[ix]).rank(method="average").to_numpy(dtype=float)
        if np.std(pred_rank) < 1e-12 or np.std(true_rank) < 1e-12:
            continue
        values.append(float(np.corrcoef(pred_rank, true_rank)[0, 1]))
    return float(np.mean(values)) if values else -math.inf


def _mean_absolute_error(scores: np.ndarray, target: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    true = _target_scalar(target).reshape(-1)
    if len(scores) != len(true):
        raise ValueError("validation scores and labels are not aligned")
    if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(true).all():
        return math.inf
    return float(np.mean(np.abs(scores - true)))


def _accept_candidate_mae(candidate_mae: float, champion_mae: float, tolerance: float) -> bool:
    """Accept a finite candidate when its validation MAE does not worsen the champion."""
    return bool(
        np.isfinite(candidate_mae)
        and np.isfinite(champion_mae)
        and np.isfinite(tolerance)
        and tolerance == 0.0
        and candidate_mae <= champion_mae
    )


def _select_test_rows(meta: pd.DataFrame, c: Config) -> pd.DataFrame:
    m = meta[(meta["as_of_date"] >= c.start) & (meta["as_of_date"] <= c.end)].copy()
    if m.empty:
        raise ValueError(f"no test rows in locked interval {c.test_start}..{c.test_end}")
    dates = pd.DatetimeIndex(sorted(m["as_of_date"].unique()))
    if c.smoke_test_schedule:
        periods = dates.to_period("M")
        unique_periods = periods.unique().sort_values()
        if len(unique_periods) < 3:
            raise ValueError("smoke schedule needs at least three calendar months")
        first_month = dates[periods == unique_periods[0]]
        second_month = dates[periods == unique_periods[1]]
        third_month = dates[periods == unique_periods[2]]
        dates = pd.DatetimeIndex(
            [first_month[-1], second_month[0], second_month[-1], third_month[0]]
        ).unique()
    elif c.max_test_dates is not None:
        dates = dates[: c.max_test_dates]
    m = m[m["as_of_date"].isin(dates)].copy()
    if c.max_stocks is not None:
        stocks = sorted(m["stock"].unique())[: c.max_stocks]
        m = m[m["stock"].isin(stocks)].copy()
    return m.sort_values(["as_of_date", "stock"]).reset_index(drop=True)


def _row_indices(meta: pd.DataFrame, selected: pd.DataFrame) -> np.ndarray:
    # Selected metadata carries the original cache row id.
    if "_row_id" not in selected:
        raise ValueError("selected metadata lost _row_id")
    return selected["_row_id"].to_numpy(dtype=np.int64)


def _date_batches(meta: pd.DataFrame, batch_size: int, seed: int, shuffle: bool) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for _, g in meta.groupby("as_of_date", sort=True):
        indices = g.index.to_numpy(dtype=np.int64).copy()
        if shuffle:
            rng.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batches.append(indices[i : i + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TorchScalarHead:
    def __init__(self, dim: int, device: str, lr: float, weight_decay: float, batch_size: int):
        import torch
        from torch import nn

        self.torch = torch
        self.device = device
        self.model = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1)).to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = int(batch_size)

    def clone(self) -> "TorchScalarHead":
        out = TorchScalarHead(self.model[1].in_features, self.device, self.lr, self.weight_decay, self.batch_size)
        out.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        return out

    def score(self, x: np.ndarray, *, train: bool = False) -> np.ndarray:
        self.model.train(train)
        with self.torch.no_grad():
            return self.model(self.torch.as_tensor(np.asarray(x), dtype=self.torch.float32, device=self.device)).squeeze(-1).detach().cpu().numpy()

    def fit(self, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, epochs: int, seed: int) -> list[dict[str, float]]:
        self.model.train()
        opt = self.torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        hist: list[dict[str, float]] = []
        y = _target_scalar(y).astype(np.float32)
        for epoch in range(1, epochs + 1):
            losses = []
            for batch in _date_batches(meta, self.batch_size, seed + epoch, True):
                xb = self.torch.as_tensor(np.asarray(x[batch]), dtype=self.torch.float32, device=self.device)
                yb = self.torch.as_tensor(y[batch], dtype=self.torch.float32, device=self.device)
                pred = self.model(xb).squeeze(-1)
                loss = self.torch.nn.functional.smooth_l1_loss(pred, yb)
                # Pairwise loss is intentionally computed within one date batch only.
                if len(batch) > 1:
                    dy = yb[:, None] - yb[None, :]
                    mask = dy != 0
                    if bool(mask.any()):
                        dp = pred[:, None] - pred[None, :]
                        rank = self.torch.nn.functional.softplus(-self.torch.sign(dy[mask]) * dp[mask]).mean()
                        loss = loss + 0.2 * rank
                opt.zero_grad(set_to_none=True)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            hist.append({"epoch": float(epoch), "loss": float(np.mean(losses)) if losses else math.nan})
        return hist

    def validation_rankic(self, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> float:
        return _mean_daily_rankic(self.score(x), y, meta)

    def save(self, path: Path, version: int, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"model": self.model.state_dict(), "version": version, "metadata": metadata}, path)


def _fit_head(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    c: Config,
    *,
    warm: Any | None = None,
    seed: int,
    epochs: int | None = None,
) -> tuple[Any, list[dict[str, float]]]:
    backend = c.backend
    if backend == "auto":
        backend = "torch" if _torch_available() else "numpy"
    if backend == "torch":
        if not _torch_available():
            raise RuntimeError("backend=torch requested but PyTorch is unavailable")
        head = warm.clone() if warm is not None else TorchScalarHead(x.shape[1], c.device, c.lr, c.weight_decay, c.batch_size)
        return head, head.fit(x, y, meta, c.epochs if epochs is None else epochs, seed)
    if backend == "numpy":
        return fit_numpy_ranking_head(x, y, ridge=1e-2), []
    raise ValueError(f"unknown backend={backend}")


def _validation_rankic(head: Any, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> float:
    if isinstance(head, NumpyRankingHead):
        return _mean_daily_rankic(head.score(x), y, meta)
    return head.validation_rankic(x, y, meta)


def _validation_mae(head: Any, x: np.ndarray, y: np.ndarray) -> float:
    return _mean_absolute_error(_predict(head, x), y)


def _predict(head: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(head.score(x), dtype=float).reshape(-1)


def _save_checkpoint(
    head: Any,
    path: Path,
    version: int,
    c: Config,
    validation_mae: float,
    validation_rankic: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_type": "continual_learning_ranking_head",
        "version": version,
        "validation_metric": "path_mae",
        "validation_mae": validation_mae,
        "validation_tolerance": c.validation_tolerance,
        "validation_rankic": validation_rankic,
        "test_start": c.test_start,
        "test_end": c.test_end,
        "lookback": c.lookback,
        "pred_len": c.pred_len,
        "seed": c.seed,
    }
    if isinstance(head, NumpyRankingHead):
        np.savez(path.with_suffix(".npz"), mean=head.state.mean, scale=head.state.scale,
                 weight=head.state.weight, bias=head.state.bias, ridge=head.state.ridge,
                 metadata=json.dumps(metadata))
    else:
        head.save(path, version, metadata)

def _sample_historical_by_year(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if n >= len(frame):
        return frame.copy()
    work = frame.copy()
    work["_replay_year"] = work["label_end_date"].dt.year
    years = sorted(work["_replay_year"].unique())
    base, extra = divmod(n, len(years))
    chosen: list[int] = []
    leftovers: list[int] = []
    for pos, year in enumerate(years):
        idx = work.index[work["_replay_year"] == year].to_numpy(dtype=np.int64).copy()
        rng.shuffle(idx)
        quota = min(len(idx), base + (1 if pos < extra else 0))
        chosen.extend(idx[:quota].tolist())
        leftovers.extend(idx[quota:].tolist())
    if len(chosen) < n:
        remaining = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)].tolist())
    return frame.loc[chosen].copy()


def _select_replay_rows(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    update_date: pd.Timestamp,
    c: Config,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic 70/30 recent/history replay set from mature labels."""
    if not 0.0 <= c.replay_recent_fraction <= 1.0:
        raise ValueError("replay_recent_fraction must be between 0 and 1")
    train_pool = train_meta.copy()
    train_pool["_source"] = "train"
    train_pool["_source_row"] = np.arange(len(train_pool), dtype=np.int64)
    test_pool = test_meta[
        (test_meta["as_of_date"] < update_date)
        & (test_meta["label_end_date"] <= update_date)
    ].copy()
    test_pool["_source"] = "test"
    test_pool["_source_row"] = test_pool.index.to_numpy(dtype=np.int64)
    pool = pd.concat([train_pool, test_pool], ignore_index=True, sort=False)
    pool = pool[pool["label_end_date"] <= update_date].copy()
    if pool.empty:
        raise ValueError(f"no mature replay rows at {update_date.date()}")

    cutoff = update_date - pd.DateOffset(months=c.replay_recent_months)
    recent = pool[pool["label_end_date"] >= cutoff].copy()
    history = pool[pool["label_end_date"] < cutoff].copy()
    cap = len(pool) if c.replay_max_rows is None else min(int(c.replay_max_rows), len(pool))
    desired_recent = int(round(cap * c.replay_recent_fraction))
    desired_history = cap - desired_recent
    recent_n = min(desired_recent, len(recent))
    history_n = min(desired_history, len(history))
    spare = cap - recent_n - history_n
    if spare > 0:
        add_recent = min(spare, len(recent) - recent_n)
        recent_n += add_recent
        spare -= add_recent
    if spare > 0:
        history_n += min(spare, len(history) - history_n)

    rng = np.random.default_rng(seed)
    if recent_n < len(recent):
        recent_idx = rng.choice(recent.index.to_numpy(dtype=np.int64), size=recent_n, replace=False)
        recent_selected = recent.loc[recent_idx].copy()
    else:
        recent_selected = recent.copy()
    history_selected = _sample_historical_by_year(history, history_n, rng)
    selected = pd.concat([recent_selected, history_selected], ignore_index=True, sort=False)
    selected = selected.sort_values(["as_of_date", "stock", "_source"]).reset_index(drop=True)
    stats = {
        "pool_rows": int(len(pool)),
        "recent_pool_rows": int(len(recent)),
        "historical_pool_rows": int(len(history)),
        "selected_rows": int(len(selected)),
        "selected_recent_rows": int(len(recent_selected)),
        "selected_historical_rows": int(len(history_selected)),
        "recent_cutoff": cutoff.date().isoformat(),
        "recent_fraction_target": float(c.replay_recent_fraction),
        "historical_year_counts": {
            str(int(year)): int(count)
            for year, count in history_selected["label_end_date"].dt.year.value_counts().sort_index().items()
        },
    }
    return selected, stats


def _materialize_replay(
    selected: pd.DataFrame,
    train_h: np.ndarray,
    train_y: np.ndarray,
    test_h: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    for source, hidden, target in (("train", train_h, train_y), ("test", test_h, test_y)):
        part = selected[selected["_source"] == source].copy()
        if part.empty:
            continue
        idx = part["_source_row"].to_numpy(dtype=np.int64)
        feature_parts.append(np.asarray(hidden[idx]))
        target_parts.append(np.asarray(target[idx]))
        meta_parts.append(part)
    if not feature_parts:
        raise ValueError("replay selection produced no materialized rows")
    x = np.concatenate(feature_parts, axis=0)
    y = np.concatenate(target_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True, sort=False)
    order = np.lexsort((meta["stock"].astype(str).to_numpy(), meta["as_of_date"].to_numpy()))
    return x[order], y[order], meta.iloc[order].reset_index(drop=True)


def _daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("as_of_date", sort=True):
        g = group.dropna(subset=["pred_signal", "actual_signal"])
        if len(g) < 2:
            continue
        rankic = g["pred_signal"].corr(g["actual_signal"], method="spearman")
        ic = g["pred_signal"].corr(g["actual_signal"], method="pearson")
        q = max(1, len(g) // 5)
        ordered = g.sort_values("pred_signal")
        rows.append({"as_of_date": date, "n_stocks": len(g), "rankic": rankic, "pearson_ic": ic, "q5_q1": ordered.tail(q).actual_signal.mean() - ordered.head(q).actual_signal.mean(), "pred_mean": g.pred_signal.mean(), "actual_mean": g.actual_signal.mean()})
    return pd.DataFrame(rows)


def _run(c: Config) -> dict[str, Any]:
    seed_everything(c.seed)
    if (c.test_start, c.test_end, c.lookback, c.pred_len) != (TEST_START, TEST_END, LOOKBACK, PRED_LEN):
        raise ValueError(f"locked CSI800-aligned experiment requires test={TEST_START}..{TEST_END} and {LOOKBACK}/{PRED_LEN} windows")
    if c.batch_size <= 0 or c.update_epochs < 1 or c.update_epochs > 3:
        raise ValueError("batch_size must be positive and update_epochs must be in [1, 3]")
    if not np.isfinite(c.validation_tolerance) or c.validation_tolerance < 0:
        raise ValueError("validation_tolerance must be finite and nonnegative")

    test_cache = Path(c.cache_dir)
    base_cache = Path(c.train_val_cache_dir or c.cache_dir)
    formal_requested = (not c.smoke_test_schedule and c.max_test_dates is None and c.max_stocks is None)
    out = Path(c.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    train_h, train_y, train_m = _load_split(base_cache, "train")
    val_h, val_y, val_m = _load_split(base_cache, "val")
    test_h_all, test_y_all, test_m_all = _load_split(test_cache, "test")
    _verify_cache_keys(train_m, c.panel_dir, "train", exact=formal_requested)
    _verify_cache_keys(val_m, c.panel_dir, "val", exact=formal_requested)
    _verify_cache_keys(test_m_all, c.panel_dir, "test", exact=formal_requested)
    train_m = _attach_label_end(train_m, c.panel_dir)
    test_m_all = _attach_label_end(test_m_all, c.panel_dir)

    test_m = _select_test_rows(test_m_all, c)
    test_idx = _row_indices(test_m_all, test_m)
    test_h = np.asarray(test_h_all[test_idx])
    test_y = np.asarray(test_y_all[test_idx])
    test_m = test_m.reset_index(drop=True)
    keep_stocks = set(test_m["stock"].astype(str).unique())
    train_h, train_y, train_m = _filter_split(
        train_h, train_y, train_m, stocks=keep_stocks, max_dates=c.max_train_dates
    )
    val_h, val_y, val_m = _filter_split(
        val_h, val_y, val_m, stocks=keep_stocks, max_dates=c.max_val_dates
    )
    train_m = train_m.reset_index(drop=True)
    val_m = val_m.reset_index(drop=True)

    formal_run = formal_requested
    raw = _load_raw_signal(test_m, c.raw_signal_file, pred_len=c.pred_len, strict_keys=formal_run)
    test_m["raw_signal"] = raw.to_numpy(dtype=float)
    test_m["label_end_date"] = pd.to_datetime(test_m["label_end_date"]).dt.normalize()
    train_m["label_end_date"] = pd.to_datetime(train_m["label_end_date"]).dt.normalize()
    test_dates = pd.DatetimeIndex(sorted(test_m.as_of_date.unique()))
    if len(test_dates) == 0:
        raise ValueError("no selected test dates")
    if not c.smoke_test_schedule and c.max_test_dates is None:
        if test_dates[0] < c.start or test_dates[-1] > c.end:
            raise ValueError(
                f"formal test calendar must stay within {c.start.date()}..{c.end.date()}, "
                f"got {len(test_dates)} dates from {test_dates[0].date()} to {test_dates[-1].date()}"
            )
    updates = month_end_update_dates(test_dates)
    update_set = set(updates)

    # B is fitted once. C starts from an exact copy of the B checkpoint.
    b_head, b_hist = _fit_head(train_h, train_y, train_m, c, seed=c.seed, epochs=c.epochs)
    b_val = _validation_rankic(b_head, val_h, val_y, val_m)
    b_mae = _validation_mae(b_head, val_h, val_y)
    if not np.isfinite(b_mae):
        raise ValueError("initial validation MAE is not finite")
    c_head = b_head.clone() if hasattr(b_head, "clone") else copy.deepcopy(b_head)
    version = 0
    champion_val = float(b_val)
    champion_mae = float(b_mae)
    checkpoint_dir = out / "checkpoints"
    _save_checkpoint(b_head, checkpoint_dir / "B_initial.pt", 0, c, b_mae, b_val)
    _save_checkpoint(c_head, checkpoint_dir / "C_version_000.pt", 0, c, b_mae, b_val)
    rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    matured_seen: set[int] = set()
    c_history: list[dict[str, Any]] = []

    b_pred = _predict(b_head, test_h)
    for pos, date in enumerate(test_dates):
        current = test_m[test_m.as_of_date == date].copy()
        current_indices = current.index.to_numpy(dtype=np.int64)
        # A/B/C use the same rows and are all predicted before any C update on date.
        c_pred = _predict(c_head, test_h[current_indices])
        for group, pred in (
            ("A", current.raw_signal.to_numpy(dtype=float)),
            ("B", b_pred[current_indices]),
            ("C", c_pred),
        ):
            for local, (_, row) in enumerate(current.iterrows()):
                rows.append(
                    {
                        "experiment_type": "continual_learning_ranking_head",
                        "group": group,
                        "seed": c.seed,
                        "as_of_date": date.date().isoformat(),
                        "stock": str(row.stock),
                        "model_version": 0 if group != "C" else version,
                        "pred_signal": float(pred[local]),
                        "actual_signal": float(_target_scalar(test_y[current_indices[local] : current_indices[local] + 1])[0]),
                        "label_end_date": row.label_end_date.date().isoformat(),
                        "updated_after_prediction": False,
                    }
                )

        if date in update_set:
            version_before = version
            champion_val_before = champion_val
            champion_mae_before = champion_mae
            mature_mask = (test_m.label_end_date <= date) & (test_m.as_of_date < date)
            mature_indices = test_m.index[mature_mask].to_numpy(dtype=np.int64)
            new_indices = [int(i) for i in mature_indices if int(i) not in matured_seen]
            matured_seen.update(new_indices)
            accepted = False
            candidate_val = math.nan
            candidate_mae = math.nan
            replay_stats: dict[str, Any] = {}
            update_decision = "skipped_no_new_matured_labels"
            hist: list[dict[str, float]] = []
            if new_indices:
                selected, replay_stats = _select_replay_rows(
                    train_m, test_m, date, c, seed=c.seed + pos + 10_000
                )
                x_update, y_update, update_meta = _materialize_replay(
                    selected, train_h, train_y, test_h, test_y
                )
                candidate, hist = _fit_head(
                    x_update,
                    y_update,
                    update_meta,
                    c,
                    warm=c_head,
                    seed=c.seed + pos + 1,
                    epochs=c.update_epochs,
                )
                candidate_val = _validation_rankic(candidate, val_h, val_y, val_m)
                candidate_mae = _validation_mae(candidate, val_h, val_y)
                accepted = _accept_candidate_mae(candidate_mae, champion_mae, c.validation_tolerance)
                update_decision = "accepted" if accepted else "rejected_validation_mae_gate"
                if accepted:
                    c_head = candidate
                    version += 1
                    champion_val = float(candidate_val)
                    champion_mae = float(candidate_mae)
                    _save_checkpoint(
                        c_head,
                        checkpoint_dir / f"C_version_{version:03d}.pt",
                        version,
                        c,
                        candidate_mae,
                        candidate_val,
                    )
                    c_history.append(
                        {
                            "version": version,
                            "validation_metric": "path_mae",
                            "validation_mae": candidate_mae,
                            "validation_rankic": candidate_val,
                            "training_rows": int(len(update_meta)),
                            "replay": replay_stats,
                            "history": hist,
                        }
                    )
            update_rows.append(
                {
                    "update_date": date.date().isoformat(),
                    "model_version_before": version_before,
                    "model_version_after": version,
                    "validation_metric": "path_mae",
                    "champion_validation_mae_before": champion_mae_before,
                    "candidate_validation_mae": candidate_mae,
                    "champion_validation_mae_after": champion_mae,
                    "champion_validation_rankic_before": champion_val_before,
                    "candidate_validation_rankic": candidate_val,
                    "champion_validation_rankic_after": champion_val,
                    "validation_tolerance": c.validation_tolerance,
                    "accepted": accepted,
                    "decision": update_decision,
                    "matured_test_rows_available": int(len(mature_indices)),
                    "new_matured_test_rows": int(len(new_indices)),
                    "selected_replay_rows": int(replay_stats.get("selected_rows", 0)),
                    "selected_recent_rows": int(replay_stats.get("selected_recent_rows", 0)),
                    "selected_historical_rows": int(replay_stats.get("selected_historical_rows", 0)),
                    "recent_cutoff": replay_stats.get("recent_cutoff"),
                    "next_test_date": test_dates[pos + 1].date().isoformat() if pos + 1 < len(test_dates) else None,
                }
            )

            print(
                f"update={date.date()} candidate_mae={candidate_mae:.8f} "
                f"champion_mae={champion_mae:.8f} tolerance={c.validation_tolerance} "
                f"decision={update_decision} version={version}", flush=True,
            )

    predictions = pd.DataFrame(rows)
    # Mark C rows at update dates explicitly; their model_version remains the pre-update version.
    for date in update_set:
        mask = (predictions.group == "C") & (predictions.as_of_date == date.date().isoformat())
        predictions.loc[mask, "updated_after_prediction"] = True
    atomic_csv(out / "predictions.csv", predictions)
    atomic_csv(out / "update_log.csv", pd.DataFrame(update_rows))
    for group in ("A", "B", "C"):
        group_df = predictions[predictions.group == group].copy()
        atomic_csv(out / f"predictions_{group}.csv", group_df)
        atomic_csv(out / f"daily_metrics_{group}.csv", _daily_metrics(group_df))
    common = predictions.pivot_table(
        index=["as_of_date", "stock", "actual_signal", "label_end_date"],
        columns="group",
        values="pred_signal",
        aggfunc="first",
    ).reset_index()
    common = common.dropna(subset=["A", "B", "C"])
    atomic_csv(out / "common_predictions.csv", common)

    cache_files: dict[str, str] = {}
    for label, cache_root in (("train_val", base_cache), ("test", test_cache)):
        for file in cache_root.glob("meta_*.csv"):
            cache_files[f"{label}/{file.name}"] = sha256_file(file)
    config = asdict(c) | {
        "experiment_type": "continual_learning_ranking_head",
        "groups": ["A", "B", "C"],
        "validation_metric": "path_mae",
        "update_policy": "month_end_test_then_train_next_test_date",
        "replay_policy": "70pct_recent_24m_30pct_historical_year_stratified",
        "resolved_test_dates": [d.date().isoformat() for d in test_dates],
        "update_dates": [d.date().isoformat() for d in updates],
        "cache_files": cache_files,
    }
    atomic_json(out / "run_config.json", config)
    summary = {
        "status": "complete",
        "validation_metric": "path_mae",
        "validation_tolerance": c.validation_tolerance,
        "experiment_type": "continual_learning_ranking_head",
        "groups": ["A", "B", "C"],
        "test_start": c.test_start,
        "test_end": c.test_end,
        "test_dates": len(test_dates),
        "test_rows": len(test_m),
        "common_rows": len(common),
        "update_dates": len(updates),
        "accepted_updates": int(sum(bool(x["accepted"]) for x in update_rows)),
        "final_c_version": version,
        "initial_validation_rankic": b_val,
        "final_champion_validation_rankic": champion_val,
        "initial_validation_mae": b_mae,
        "final_champion_validation_mae": champion_mae,
        "backend": c.backend if c.backend != "auto" else ("torch" if _torch_available() else "numpy"),
        "seed": c.seed,
        "initial_training_history": b_hist,
        "continual_history": c_history,
    }
    atomic_json(out / "summary.json", summary)
    return summary

def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--raw-signal-file")
    p.add_argument("--panel-dir")
    p.add_argument("--train-val-cache-dir")
    p.add_argument("--test-start", default=TEST_START)
    p.add_argument("--test-end", default=TEST_END)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--pred-len", type=int, default=PRED_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    p.add_argument("--validation-tolerance", type=float, default=0.0)
    p.add_argument("--max-test-dates", type=int)
    p.add_argument("--smoke-test-schedule", action="store_true")
    p.add_argument("--max-stocks", type=int)
    p.add_argument("--max-train-dates", type=int)
    p.add_argument("--max-val-dates", type=int)
    p.add_argument("--replay-max-rows", type=int, default=120_000)
    p.add_argument("--replay-recent-months", type=int, default=24)
    p.add_argument("--replay-recent-fraction", type=float, default=0.70)
    a = p.parse_args()
    return Config(**vars(a))


def main() -> None:
    c = parse_args()
    started = time.time()
    try:
        summary = _run(c)
        summary["elapsed_seconds"] = time.time() - started
        atomic_json(Path(c.output_dir) / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        out = Path(c.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "FAILED.json", {"status": "failed", "error": repr(exc), "experiment_type": "continual_learning_ranking_head"})
        raise


if __name__ == "__main__":
    main()










