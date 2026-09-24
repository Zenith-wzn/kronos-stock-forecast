from __future__ import annotations

"""Train and evaluate frozen-Kronos cross-sectional ranking reward heads.

This module has two explicit execution modes:

* ``run_experiment`` uses PyTorch when available and trains S1-S4 using only
  train rows. Validation selects the checkpoint; test is inference only.
* ``run_numpy_experiment`` is a deterministic CPU smoke fallback. It is clearly
  marked as a fallback and never claims to be a production PyTorch run.

The cache contract is intentionally strict. Missing or malformed market caches
produce auditable ``无法验证`` artifacts instead of inferred values.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .features import load_feature_cache
from .head import (
    NumpyRankingHead,
    RankingHead,
    date_balanced_pairwise_loss_torch,
    fit_numpy_ranking_head,
    make_same_date_pairs,
    pairwise_rank_loss_numpy,
)
from .portfolio import PortfolioConfig, MARKET_PORTFOLIO_CONFIGS, backtest_long_only, run_cost_sensitivity

MARKETS = ("CSI800", "CSI300", "S&P500")
SPLITS = ("train", "val", "test")
MODELS = ("S0", "S1", "S2", "S3", "S4")
REQUIRED_META = {
    "dataset", "split", "stock", "as_of_date", "label_start_date", "label_end_date",
    "current_close", "entry_price", "exit_price", "actual_endpoint_return", "actual_path_mean_return", "next_day_return",
}


@dataclass(frozen=True)
class SortingConfig:
    train_end: str = "2024-12-15"
    val_start: str = "2025-01-02"
    val_end: str = "2025-06-14"
    test_start: str = "2025-07-01"
    test_end: str = "2026-06-05"
    lookback: int = 90
    pred_len: int = 10
    pair_scale: float = 2.0
    huber_weight: float = 0.2
    max_pairs_per_date: int = 4096
    min_cross_section: int = 3
    hidden_dim: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_dates: int = 16
    dates_per_epoch: int = 256
    predict_batch_size: int = 65536
    initial_capital: float = 1_000_000.0
    seeds: tuple[int, ...] = (100, 101, 102, 103, 104)


try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


def _json_clean(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_clean(value.item())
        return [_json_clean(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {str(p): sha256(p) for p in paths if p.exists()}


def git_commit(root: Path | None = None) -> str:
    """Return the current git revision without making git availability fatal."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root or Path.cwd()),
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip() or "无法验证"
    except Exception:
        return "无法验证"


def _write_metric_status_csv(out: Path, payload: dict[str, Any]) -> None:
    rows = []
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(_json_clean(value), ensure_ascii=False, separators=(",", ":"))
        rows.append({"metric": str(key), "value": value if value is not None else None,
                     "metric_status": payload.get("metric_status", payload.get("status", "无法验证")),
                     "unavailable_reason": payload.get("unavailable_reason", "")})
    pd.DataFrame(rows, columns=["metric", "value", "metric_status", "unavailable_reason"]).to_csv(
        out / "metric_status.csv", index=False, encoding="utf-8-sig"
    )


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _date_set(meta: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.to_datetime(meta["as_of_date"], errors="coerce").dt.normalize().dropna())


def validate_cache(
    cache_dir: str | Path,
    dataset: str,
    *,
    config: SortingConfig = SortingConfig(),
) -> dict[str, Any]:
    """Validate all cache arrays/metadata and fixed temporal boundaries."""
    root = Path(cache_dir)
    result: dict[str, Any] = {
        "dataset": dataset, "cache_dir": str(root.resolve()), "status": "可验证", "warnings": [], "splits": {},
        "survivorship_warning": "当前成分股池回溯，存在幸存者偏差和当前成分股前视偏差；结果不是 point-in-time 无偏历史组合结果",
    }
    dates_by_split: dict[str, set[pd.Timestamp]] = {}
    boundaries = {
        "train": (None, pd.Timestamp(config.train_end)),
        "val": (pd.Timestamp(config.val_start), pd.Timestamp(config.val_end)),
        "test": (pd.Timestamp(config.test_start), pd.Timestamp(config.test_end)),
    }
    for split in SPLITS:
        try:
            hidden, target, meta = load_feature_cache(root, split, mmap=True)
        except Exception as exc:
            result["status"] = "无法验证"
            result["warnings"].append(f"{split}: {exc}")
            continue
        missing = sorted(REQUIRED_META - set(meta.columns))
        if missing:
            result["status"] = "无法验证"
            result["warnings"].append(f"{split}: meta 缺少字段 {missing}")
        if hidden.ndim != 2 or len(hidden) != len(target) or len(hidden) != len(meta):
            result["status"] = "无法验证"
            result["warnings"].append(f"{split}: hidden/label/meta 行数或维度不一致")
        dates = pd.to_datetime(meta.get("as_of_date"), errors="coerce").dt.normalize()
        stocks = meta.get("stock", pd.Series(index=meta.index, dtype=str)).astype(str)
        keys = pd.MultiIndex.from_arrays([dates, stocks], names=["as_of_date", "stock"])
        duplicate_keys = int(keys.duplicated().sum())
        counts = dates.value_counts(dropna=True)
        small_dates = int((counts < config.min_cross_section).sum()) if len(counts) else 0
        if dates.isna().any() or duplicate_keys or small_dates:
            result["status"] = "无法验证"
            result["warnings"].append(
                f"{split}: duplicate_keys={duplicate_keys}, invalid_dates={int(dates.isna().sum())}, small_dates={small_dates}"
            )
        hidden_finite = True
        for start in range(0, len(hidden), 65536):
            if not np.isfinite(np.asarray(hidden[start:start + 65536])).all():
                hidden_finite = False
                break
        if not hidden_finite:
            result["status"] = "无法验证"
            result["warnings"].append(f"{split}: hidden 含 NaN/Inf")
        lower, upper = boundaries[split]
        valid_dates = dates.dropna()
        if lower is not None and len(valid_dates) and valid_dates.min() < lower:
            result["status"] = "无法验证"; result["warnings"].append(f"{split}: 日期早于 {lower.date()}")
        if upper is not None and len(valid_dates) and valid_dates.max() > upper:
            result["status"] = "无法验证"; result["warnings"].append(f"{split}: 日期晚于 {upper.date()}")
        if split == "train" and len(valid_dates) and valid_dates.max() > pd.Timestamp(config.train_end):
            result["status"] = "无法验证"; result["warnings"].append("train: as_of_date 晚于 train_end")
        if "label_end_date" in meta:
            label_end = pd.to_datetime(meta["label_end_date"], errors="coerce").dt.normalize()
            if label_end.isna().any() or (label_end <= dates).any():
                result["status"] = "无法验证"; result["warnings"].append(f"{split}: label_end_date 未晚于 as_of_date")
            # test_end is the final *prediction* date.  A 10-day label for that
            # prediction necessarily ends later, so maturity is checked against
            # the actual run date instead of incorrectly truncating at test_end.
            run_date = pd.Timestamp.now().normalize()
            if split == "test" and (label_end > run_date).any():
                result["status"] = "无法验证"; result["warnings"].append(
                    f"test: 存在截至 {run_date.date()} 尚未成熟的标签"
                )
        dates_by_split[split] = set(valid_dates)
        result["splits"][split] = {
            "rows": int(len(meta)), "hidden_shape": list(hidden.shape), "target_shape": list(np.asarray(target).shape),
            "dates": int(dates.nunique()), "stocks": int(stocks.nunique()), "duplicate_keys": duplicate_keys,
            "min_cross_section": int(counts.min()) if len(counts) else 0, "max_cross_section": int(counts.max()) if len(counts) else 0,
            "invalid_price_rows": int(sum(pd.to_numeric(meta.get(c, pd.Series(np.nan, index=meta.index)), errors="coerce").isna().sum() for c in ("current_close", "entry_price", "exit_price"))),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = dates_by_split.get(left, set()) & dates_by_split.get(right, set())
        if overlap:
            result["status"] = "无法验证"; result["warnings"].append(f"{left}/{right} 日期交集 {len(overlap)}")
    if set(result["splits"]) != set(SPLITS):
        result["status"] = "无法验证"
    return result


def _target_vector(target: np.ndarray, meta: pd.DataFrame, *, label: str = "endpoint") -> np.ndarray:
    """Extract endpoint/path labels, preferring explicit metadata columns."""
    col = "actual_endpoint_return" if label in ("endpoint", "actual_endpoint_return") else "actual_path_mean_return"
    if col in meta.columns:
        return pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
    y = np.asarray(target, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim != 2:
        raise ValueError("target must be one- or two-dimensional")
    if label in ("endpoint", "actual_endpoint_return") and y.shape[1] >= 10:
        return y[:, 9]
    return np.nanmean(y, axis=1)


def train_standardizer(hidden: np.ndarray, target: np.ndarray, *, chunk_size: int = 65536) -> dict[str, np.ndarray]:
    """Fit train-only statistics without materialising a multi-GB float64 copy."""
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    if getattr(hidden, "ndim", 0) != 2 or len(hidden) == 0 or len(hidden) != len(y):
        raise ValueError("training arrays are not aligned")
    if not np.isfinite(y).all():
        raise ValueError("training targets contain NaN/Inf")
    total = np.zeros(hidden.shape[1], dtype=np.float64)
    total_sq = np.zeros(hidden.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(hidden), int(chunk_size)):
        block = np.asarray(hidden[start:start + int(chunk_size)], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("training hidden contains NaN/Inf")
        total += block.sum(axis=0, dtype=np.float64)
        total_sq += np.square(block, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(block)
    x_mean64 = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(x_mean64), 0.0)
    x_scale64 = np.sqrt(variance)
    x_scale64[x_scale64 < 1e-8] = 1.0
    y_mean = float(np.mean(y, dtype=np.float64))
    y_scale = float(np.std(y, dtype=np.float64)); y_scale = y_scale if y_scale >= 1e-8 else 1.0
    return {"x_mean": x_mean64.astype(np.float32), "x_scale": x_scale64.astype(np.float32),
            "y_mean": np.asarray(y_mean, dtype=np.float32), "y_scale": np.asarray(y_scale, dtype=np.float32)}


def apply_standardizer(hidden: np.ndarray, target: np.ndarray | None, stats: dict[str, np.ndarray]):
    x = (np.asarray(hidden, dtype=np.float32) - stats["x_mean"]) / stats["x_scale"]
    if target is None:
        return x.astype(np.float32, copy=False), None
    y = (np.asarray(target, dtype=np.float32) - float(stats["y_mean"])) / float(stats["y_scale"])
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y, method=method))


def _bootstrap_mean(values: np.ndarray, *, seed: int, n_boot: int = 1000, block: int = 5) -> tuple[float, float]:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), len(x)))
    starts = np.arange(len(x))
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample: list[float] = []
        while len(sample) < len(x):
            start = int(rng.choice(starts))
            sample.extend(x[(start + np.arange(block)) % len(x)].tolist())
        means[i] = np.mean(sample[:len(x)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cross_section_metrics(
    predictions: pd.DataFrame, *, score_col: str, target_col: str, top_k: int,
    bootstrap_seed: int = 0, compute_bootstrap: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for date, group in predictions.groupby("as_of_date", sort=True):
        g = group.dropna(subset=[score_col, target_col]).copy()
        if len(g) < 3:
            skipped += 1; continue
        g = g.sort_values([score_col, "stock"], ascending=[True, True], kind="mergesort")
        q = max(1, len(g) // 5)
        top = g.tail(min(top_k, len(g))); bottom = g.head(q)
        rows.append({
            "as_of_date": date, "n": len(g), "ic": _corr(g[score_col], g[target_col], "pearson"),
            "rankic": _corr(g[score_col], g[target_col], "spearman"),
            "q5_q1": float(top.iloc[:0][target_col].mean()) if False else float(g.tail(q)[target_col].mean() - bottom[target_col].mean()),
            "top_k_return": float(top[target_col].mean()),
            "top_k_excess_return": float(top[target_col].mean() - g[target_col].mean()),
            "bottom_k_return": float(bottom[target_col].mean()),
            "long_short_return": float(top[target_col].mean() - bottom[target_col].mean()),
            "direction_accuracy": float(((g[score_col] > 0) == (g[target_col] > 0)).mean()),
        })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, {"metric_status": "无法验证", "unavailable_reason": "no valid cross-sectional dates", "skipped_dates": skipped}
    rank = pd.to_numeric(daily["rankic"], errors="coerce").dropna()
    from .evaluate import newey_west_mean_t
    summary: dict[str, Any] = {
        "metric_status": "可验证", "unavailable_reason": "", "n_dates": int(len(daily)), "skipped_dates": skipped,
        "valid_stock_count_mean": float(daily["n"].mean()), "ic_mean": float(daily["ic"].mean()),
        "ic_std": float(daily["ic"].std(ddof=1)) if len(daily) > 1 else np.nan,
        "rankic_mean": float(rank.mean()) if len(rank) else np.nan,
        "rankic_std": float(rank.std(ddof=1)) if len(rank) > 1 else np.nan,
        "rankic_median": float(rank.median()) if len(rank) else np.nan,
        "rankic_positive_ratio": float((rank > 0).mean()) if len(rank) else np.nan,
        "rankic_nw_t": float(newey_west_mean_t(rank.to_numpy())) if len(rank) else np.nan,
        "rankic_bootstrap_ci_95": (_bootstrap_mean(rank.to_numpy(), seed=bootstrap_seed)
                                    if compute_bootstrap else (np.nan, np.nan)),
    }
    for col in ("q5_q1", "top_k_return", "top_k_excess_return", "bottom_k_return", "long_short_return", "direction_accuracy"):
        summary[col if col == "top_k_return" else f"{col}_mean"] = float(pd.to_numeric(daily[col], errors="coerce").mean())
        if col in ("q5_q1", "top_k_excess_return", "long_short_return"):
            summary[f"{col}_bootstrap_ci_95"] = (_bootstrap_mean(daily[col].to_numpy(), seed=bootstrap_seed + len(col))
                                                   if compute_bootstrap else (np.nan, np.nan))
    return daily, summary


def _raw_signal(meta: pd.DataFrame) -> np.ndarray:
    if "raw_signal" not in meta.columns:
        return np.full(len(meta), np.nan, dtype=float)
    return pd.to_numeric(meta["raw_signal"], errors="coerce").to_numpy(dtype=float)


def _fit_numpy_model(model: str, x: np.ndarray, y: np.ndarray, dates: np.ndarray, *, seed: int, config: SortingConfig) -> NumpyRankingHead:
    """Deterministic train-only CPU proxy for smoke tests.

    This is not a claim of MLP optimisation.  It provides distinct, auditable
    S1-S4 smoke paths when PyTorch is absent; formal results must use Torch.
    """
    if model == "S1":
        return fit_numpy_ranking_head(x, y, ridge=1e-2)
    if model in ("S2", "S3", "S4"):
        z = np.column_stack([x, np.tanh(x), x * x])
        return fit_numpy_ranking_head(z, y, ridge=1e-2)
    raise ValueError(f"unsupported numpy model {model}")


def _numpy_predict(head: NumpyRankingHead, model: str, x: np.ndarray) -> np.ndarray:
    if model == "S1":
        return head.score(x)
    if model in ("S2", "S3", "S4"):
        return head.score(np.column_stack([x, np.tanh(x), x * x]))
    raise ValueError(model)


def _selection_score(summary: dict[str, Any], val_ref: dict[str, tuple[float, float]] | None = None) -> float:
    values = [summary.get("rankic_mean"), summary.get("q5_q1_mean"), summary.get("top_k_excess_return_mean")]
    if not all(v is not None and np.isfinite(float(v)) for v in values):
        return float("nan")
    ref = val_ref or {"q5_q1": (0.0, 1.0), "top": (0.0, 1.0)}
    q = (float(values[1]) - ref["q5_q1"][0]) / (ref["q5_q1"][1] or 1.0)
    t = (float(values[2]) - ref["top"][0]) / (ref["top"][1] or 1.0)
    return 0.5 * float(values[0]) + 0.3 * q + 0.2 * t


def _add_mean_aliases(summary: dict[str, Any]) -> dict[str, Any]:
    """Expose the public schema while retaining concise internal column names."""
    out = dict(summary)
    for key in ("q5_q1", "top_k_excess_return", "bottom_k_return", "long_short_return", "direction_accuracy"):
        if key in out and f"{key}_mean" not in out:
            out[f"{key}_mean"] = out[key]
    return out


def selection_score_from_validation_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute selection scores from validation rows without using test data."""
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    if "model" not in out.columns or "selection_score" in out.columns:
        return out
    q = pd.to_numeric(out.get("q5_q1_mean"), errors="coerce")
    t = pd.to_numeric(out.get("top_k_excess_return_mean"), errors="coerce")
    q_good, t_good = q[np.isfinite(q)], t[np.isfinite(t)]
    q_ref = (float(q_good.min()), max(float(q_good.max() - q_good.min()), 1e-12)) if len(q_good) else (0.0, 1.0)
    t_ref = (float(t_good.min()), max(float(t_good.max() - t_good.min()), 1e-12)) if len(t_good) else (0.0, 1.0)
    out["selection_score"] = [
        _selection_score(row.to_dict(), {"q5_q1": q_ref, "top": t_ref})
        for _, row in out.iterrows()
    ]
    return out


def _torch_loss(model: str, scores, y, dates: np.ndarray, *, config: SortingConfig, pair_seed: int = 0):
    # S1 ends in Linear(..., 1), while RankingHead already squeezes its output.
    # Flatten both sides to prevent accidental [N, 1] versus [N] broadcasting.
    scores = scores.reshape(-1)
    y = y.reshape(-1)
    if scores.numel() != y.numel() or scores.numel() != len(dates):
        raise ValueError("dates, scores and targets must be aligned and non-empty")
    if scores.numel() == 0:
        raise ValueError("empty training batch is not allowed")
    rank = model in ("S3", "S4")
    huber = model in ("S1", "S2", "S4")
    loss_rank = date_balanced_pairwise_loss_torch(
        scores, y, dates, scale=config.pair_scale, min_items=config.min_cross_section,
        max_pairs_per_date=config.max_pairs_per_date, seed=pair_seed
    ) if rank else scores.sum() * 0.0
    loss_huber = F.huber_loss(scores, y) if huber else scores.sum() * 0.0
    if rank and huber:
        return loss_rank + config.huber_weight * loss_huber, float(loss_rank.detach().cpu()), float(loss_huber.detach().cpu())
    return loss_rank + loss_huber, float(loss_rank.detach().cpu()), float(loss_huber.detach().cpu())


def _date_batches(dates: np.ndarray, *, batch_dates: int, seed: int, dates_per_epoch: int | None = None):
    """Yield non-empty row indices grouped by normalized datetime64[ns] dates.

    datetime64.tolist() can turn nanosecond timestamps into integer epoch values.
    Comparing those integers with the original datetime array yields all-False
    masks, so the complete selection remains in NumPy datetime space.
    """
    d = np.asarray(dates).astype("datetime64[ns]")
    if d.ndim != 1:
        d = d.reshape(-1)
    unique = np.unique(d)
    rng = np.random.default_rng(seed)
    if dates_per_epoch is not None and 0 < int(dates_per_epoch) < len(unique):
        selected = rng.choice(len(unique), size=int(dates_per_epoch), replace=False)
        unique = unique[np.sort(selected)]
    else:
        unique = unique[rng.permutation(len(unique))]
    step = max(1, int(batch_dates))
    for start in range(0, len(unique), step):
        selected_dates = unique[start:start + step]
        indices = np.flatnonzero(np.isin(d, selected_dates))
        if len(indices):
            yield indices.astype(np.int64, copy=False)


def _require_finite_loss(loss, *, dataset: str, model: str, seed: int, epoch: int, batch_no: int, rows: int) -> None:
    """Fail fast instead of allowing a corrupt optimization run to continue."""
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(
            f"non-finite loss: dataset={dataset} model={model} seed={seed} "
            f"epoch={epoch} batch={batch_no} rows={rows}"
        )


def _standardize_hidden_block(block: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    x = np.asarray(block, dtype=np.float32)
    if stats is None:
        return x
    return ((x - stats["x_mean"]) / stats["x_scale"]).astype(np.float32, copy=False)


def _predict_torch(
    net, x: np.ndarray, device, *, batch_size: int = 65536,
    stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Predict from a memmap in bounded, lazily standardized batches."""
    net.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            block = _standardize_hidden_block(x[start:start + int(batch_size)], stats)
            batch = torch.as_tensor(block, dtype=torch.float32, device=device)
            out.append(net(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _torch_fit_model(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    val_x: np.ndarray | None = None,
    val_target: np.ndarray | None = None,
    val_meta: pd.DataFrame | None = None,
    dataset: str = "",
    seed: int,
    config: SortingConfig,
    stats: dict[str, np.ndarray] | None = None,
):
    """Fit one frozen-backbone head using date mini-batches and val-only selection."""
    if torch is None:
        raise ImportError("PyTorch is required for production S1-S4 training")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(x.shape[1])
    if model == "S1":
        net = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 1)).to(device)
    else:
        net = RankingHead(input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    candidates: list[tuple[int, dict[str, Any]]] = []

    def train_one_epoch(epoch: int):
        net.train()
        total = 0.0; n_batches = 0; rank_total = 0.0; huber_total = 0.0
        for batch_no, indices in enumerate(_date_batches(
            dates, batch_dates=config.batch_dates, seed=seed + epoch,
            dates_per_epoch=config.dates_per_epoch,
        )):
            if len(indices) == 0:
                continue
            xb_np = _standardize_hidden_block(x[indices], stats)
            xb = torch.as_tensor(xb_np, dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[indices], dtype=torch.float32, device=device).reshape(-1)
            db = dates[indices]
            optimizer.zero_grad(set_to_none=True)
            scores = net(xb).reshape(-1)
            loss, rank_loss, huber_loss = _torch_loss(
                model, scores, yb, db, config=config, pair_seed=seed + epoch * 10000 + batch_no,
            )
            _require_finite_loss(
                loss, dataset=dataset, model=model, seed=seed, epoch=epoch,
                batch_no=batch_no, rows=len(indices),
            )
            loss.backward(); optimizer.step()
            total += float(loss.detach().cpu()); rank_total += rank_loss; huber_total += huber_loss; n_batches += 1
        return total / max(n_batches, 1), rank_total / max(n_batches, 1), huber_total / max(n_batches, 1), n_batches

    for epoch in range(1, config.epochs + 1):
        train_loss, rank_loss, huber_loss, n_batches = train_one_epoch(epoch)
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss,
                                  "pairwise_loss": rank_loss, "huber_loss": huber_loss,
                                  "train_batches": n_batches, "device": str(device)}
        if val_x is not None and val_target is not None and val_meta is not None:
            val_score = _predict_torch(net, val_x, device, batch_size=config.predict_batch_size, stats=stats)
            vf = _frame(val_meta, dataset=dataset, split="val", model=model, seed=seed,
                        score=val_score, endpoint=np.asarray(val_target), path=np.full(len(val_meta), np.nan))
            _, metric = cross_section_metrics(
                vf, score_col="prediction_score", target_col="actual_endpoint_return",
                top_k=MARKET_PORTFOLIO_CONFIGS[dataset].top_k, bootstrap_seed=seed,
                compute_bootstrap=False,
            )
            metric = _add_mean_aliases(metric)
            record.update({"val_rankic_mean": metric.get("rankic_mean"),
                           "val_q5_q1_mean": metric.get("q5_q1_mean"),
                           "val_top_k_excess_return_mean": metric.get("top_k_excess_return_mean")})
            candidates.append((epoch, metric))
        history.append(record)
        print(f"[{dataset}][seed={seed}][{model}] epoch={epoch}/{config.epochs} "
              f"loss={train_loss:.6g} batches={n_batches}", flush=True)

    best_epoch = config.epochs; best_selection = float("nan")
    if candidates:
        qvals = np.asarray([m.get("q5_q1_mean", np.nan) for _, m in candidates], dtype=float)
        tvals = np.asarray([m.get("top_k_excess_return_mean", np.nan) for _, m in candidates], dtype=float)
        def ref(values: np.ndarray) -> tuple[float, float]:
            good = values[np.isfinite(values)]
            if len(good) == 0: return (0.0, 1.0)
            lo, hi = float(good.min()), float(good.max())
            return (lo, max(hi - lo, 1e-12))
        qref, tref = ref(qvals), ref(tvals)
        scored = [(epoch, _selection_score(metric, {"q5_q1": qref, "top": tref})) for epoch, metric in candidates]
        valid_scored = [(e, v) for e, v in scored if np.isfinite(v)]
        if valid_scored:
            best_epoch, best_selection = max(valid_scored, key=lambda item: (item[1], -item[0]))
        for record in history:
            if record["epoch"] == best_epoch:
                record["selection_score"] = best_selection

    # Refit from the deterministic initial state through the selected epoch.
    set_seed(seed)
    if model == "S1":
        net = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 1)).to(device)
    else:
        net = RankingHead(input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    for epoch in range(1, best_epoch + 1):
        train_one_epoch(epoch)
    return net, history, device, int(best_epoch), float(best_selection)

def _frame(meta: pd.DataFrame, *, dataset: str, split: str, model: str, seed: int, score: np.ndarray, endpoint: np.ndarray, path: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame({"dataset": dataset, "split": split, "model": model, "seed": seed,
                      "as_of_date": pd.to_datetime(meta["as_of_date"], errors="coerce").dt.normalize(),
                      "stock": meta["stock"].astype(str), "prediction_score": score,
                      "actual_endpoint_return": endpoint, "actual_path_mean_return": path})
    f["cross_sectional_zscore"] = f.groupby("as_of_date")["prediction_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 1e-8 else 1.0)
    )
    f["percentile_rank"] = f.groupby("as_of_date")["prediction_score"].rank(pct=True)
    for col in ("current_close", "entry_price", "exit_price", "next_day_return"):
        if col in meta.columns:
            f[col] = pd.to_numeric(meta[col], errors="coerce").to_numpy()
    return f


def _write_unavailable(out: Path, gate: dict[str, Any], *, dataset: str, seed: int, config: SortingConfig, reason: str | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    message = reason or "; ".join(gate.get("warnings", [])) or "sorting reward experiment unavailable"
    payload = dict(gate)
    if reason:
        payload.setdefault("warnings", []).append(reason)
    payload.update({"dataset": dataset, "seed": seed, "status": "无法验证",
                    "metric_status": "无法验证", "unavailable_reason": message})
    (out / "metric_status.json").write_text(json.dumps(_json_clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metric_status_csv(out, payload)
    (out / "config.json").write_text(json.dumps(_json_clean(asdict(config)), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\ntorch={getattr(torch, '__version__', 'unavailable')}\n",
        encoding="utf-8",
    )
    (out / "git_commit.txt").write_text(git_commit(Path(__file__).resolve().parents[1]) + "\n", encoding="utf-8")
    return payload


def run_experiment(
    cache_dir: str | Path,
    output_dir: str | Path,
    dataset: str,
    *,
    seed: int = 100,
    config: SortingConfig = SortingConfig(),
    label: str = "endpoint",
    allow_numpy_fallback: bool = False,
) -> dict[str, Any]:
    """Run one market/seed. Test is never used in fitting or selection."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    gate = validate_cache(cache_dir, dataset, config=config)
    if gate["status"] != "可验证":
        return _write_unavailable(out, gate, dataset=dataset, seed=seed, config=config)
    htr, ttr, mtr = load_feature_cache(cache_dir, "train", mmap=True)
    hval, tval, mval = load_feature_cache(cache_dir, "val", mmap=True)
    hte, tte, mte = load_feature_cache(cache_dir, "test", mmap=True)
    ytr = _target_vector(ttr, mtr, label=label); yval = _target_vector(tval, mval, label=label); yte = _target_vector(tte, mte, label=label)
    path_val = _target_vector(tval, mval, label="path_mean"); path_te = _target_vector(tte, mte, label="path_mean")
    if len(ytr) < config.min_cross_section or not np.isfinite(ytr).all():
        return _write_unavailable(out, gate, dataset=dataset, seed=seed, config=config,
                                  reason="训练集有效行不足或训练标签含 NaN/Inf")
    if "next_day_return" not in mte.columns:
        return _write_unavailable(out, gate, dataset=dataset, seed=seed, config=config,
                                  reason="test meta 缺少 next_day_return；不能将 10 日 Endpoint 标签当作每日组合收益")
    stats = train_standardizer(htr, ytr)
    ytrz = ((np.asarray(ytr, dtype=np.float32) - float(stats["y_mean"])) / float(stats["y_scale"])).astype(np.float32, copy=False)
    train_dates = pd.to_datetime(mtr["as_of_date"]).dt.normalize().to_numpy()
    all_prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []; val_rows: list[dict[str, Any]] = []; test_rows: list[dict[str, Any]] = []; portfolio_summaries: list[dict[str, Any]] = []
    model_heads: dict[str, Any] = {}

    for model in MODELS:
        print(f"[{dataset}][seed={seed}] starting {model}", flush=True)
        if model == "S0":
            train_score = None; val_score = _raw_signal(mval); test_score = _raw_signal(mte)
            model_source = "raw_signal"
        elif torch is not None:
            net, hist, device, best_epoch, best_selection = _torch_fit_model(
                model, htr, ytrz, train_dates, val_x=hval, val_target=yval, val_meta=mval,
                dataset=dataset, seed=seed, config=config, stats=stats
            )
            model_source = "pytorch"
            history_rows.extend({"dataset": dataset, "model": model, "seed": seed,
                                 "best_epoch": best_epoch, "best_selection_score": best_selection, **h} for h in hist)
            train_score = None
            val_score = _predict_torch(net, hval, device, batch_size=config.predict_batch_size, stats=stats)
            test_score = _predict_torch(net, hte, device, batch_size=config.predict_batch_size, stats=stats)
            # Save state only after val evaluation below; selection is val-only.
            model_heads[model] = (net, device)
        elif allow_numpy_fallback:
            xtr, _ = apply_standardizer(htr, None, stats); xval, _ = apply_standardizer(hval, None, stats); xte, _ = apply_standardizer(hte, None, stats)
            head = _fit_numpy_model(model, xtr, ytrz, train_dates, seed=seed, config=config)
            train_score = None; val_score = _numpy_predict(head, model, xval); test_score = _numpy_predict(head, model, xte)
            model_heads[model] = head; model_source = "numpy_smoke_fallback"
            history_rows.append({"dataset": dataset, "model": model, "seed": seed, "epoch": 1,
                                 "train_loss": pairwise_rank_loss_numpy(_numpy_predict(head, model, xtr), ytrz),
                                 "model_source": model_source})
        else:
            reason = "PyTorch 未安装，无法执行正式 S1-S4 训练"
            for split in ("train", "val", "test"):
                test_rows.append({"dataset": dataset, "model": model, "seed": seed, "split": split, "metric_status": "无法验证", "unavailable_reason": reason})
            portfolio_summaries.append({"dataset": dataset, "model": model, "seed": seed, "metric_status": "无法验证", "unavailable_reason": reason})
            continue

        # Persist validation/test predictions only.  Training scores are not
        # needed for model selection and would multiply output size by millions
        # of rows across five seeds.
        val_frame = _frame(mval, dataset=dataset, split="val", model=model, seed=seed, score=np.asarray(val_score), endpoint=yval, path=path_val)
        test_frame = _frame(mte, dataset=dataset, split="test", model=model, seed=seed, score=np.asarray(test_score), endpoint=yte, path=path_te)
        all_prediction_frames.extend((val_frame, test_frame))
        _, val_metric = cross_section_metrics(val_frame, score_col="prediction_score", target_col="actual_endpoint_return", top_k=MARKET_PORTFOLIO_CONFIGS[dataset].top_k, bootstrap_seed=seed)
        val_metric = _add_mean_aliases(val_metric)
        val_metric["selection_score"] = _selection_score(val_metric)
        val_metric.update({"dataset": dataset, "model": model, "seed": seed, "split": "val", "model_source": model_source})
        val_rows.append(val_metric)
        _, test_metric = cross_section_metrics(test_frame, score_col="prediction_score", target_col="actual_endpoint_return", top_k=MARKET_PORTFOLIO_CONFIGS[dataset].top_k, bootstrap_seed=seed)
        test_metric = _add_mean_aliases(test_metric)
        test_metric.update({"dataset": dataset, "model": model, "seed": seed, "split": "test", "model_source": model_source})
        test_rows.append(test_metric)
        portfolio = backtest_long_only(test_frame, return_col="next_day_return", config=MARKET_PORTFOLIO_CONFIGS[dataset], dataset=dataset, model=model, seed=seed)
        portfolio_summaries.append(dict(portfolio["summary"], signal_mode="long_only_top_k"))
        portfolio["daily"].to_csv(out / f"portfolio_daily_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        portfolio["holdings"].to_csv(out / f"holdings_daily_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        portfolio["trades"].to_csv(out / f"trades_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        base_portfolio_config = MARKET_PORTFOLIO_CONFIGS[dataset]
        cross_section_size = int(test_frame.groupby("as_of_date")["stock"].nunique().median()) if not test_frame.empty else 0
        sensitivity_top_ks = tuple(dict.fromkeys((
            base_portfolio_config.top_k,
            max(1, round(cross_section_size * 0.10)) if cross_section_size else base_portfolio_config.top_k,
            max(1, round(cross_section_size * 0.20)) if cross_section_size else base_portfolio_config.top_k,
        )))
        endpoint_sens = run_cost_sensitivity(
            test_frame, return_col="next_day_return", base_config=base_portfolio_config,
            top_ks=sensitivity_top_ks, dataset=dataset, model=model, seed=seed,
        )
        endpoint_sens["label_mode"] = "Endpoint"
        endpoint_sens["label_sensitivity_status"] = "可验证"
        unavailable_label_rows = []
        for label_mode in ("Path", "Path + Endpoint ensemble"):
            unavailable_label_rows.append({
                "dataset": dataset, "model": model, "seed": seed,
                "label_mode": label_mode, "metric_status": "无法验证",
                "label_sensitivity_status": "无法验证",
                "unavailable_reason": "该标签模式需要独立训练并由验证集锁定 checkpoint；主 Endpoint 模型结果不得冒充标签敏感性",
            })
        sens = pd.concat([endpoint_sens, pd.DataFrame(unavailable_label_rows)], ignore_index=True, sort=False)
        sens["top_k_label"] = sens.get("sensitivity_top_k", pd.Series(index=sens.index, dtype=float)).map({
            base_portfolio_config.top_k: "baseline",
            sensitivity_top_ks[1] if len(sensitivity_top_ks) > 1 else base_portfolio_config.top_k: "10%",
            sensitivity_top_ks[2] if len(sensitivity_top_ks) > 2 else base_portfolio_config.top_k: "20%",
        })
        sens.to_csv(out / f"cost_sensitivity_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        if model == "S4" and torch is not None and model in model_heads:
            net = model_heads[model][0]
            if hasattr(net, "state_dict"):
                torch.save({"model": model, "state_dict": net.state_dict(),
                           "seed": seed, "best_epoch": int(best_epoch),
                           "selection_score": float(best_selection) if np.isfinite(best_selection) else None,
                           "config": asdict(config), "stats": _json_clean(stats)},
                          out / f"checkpoint_{model}_seed{seed}.pt")
        elif model != "S0" and model in model_heads and isinstance(model_heads[model], NumpyRankingHead):
            state = model_heads[model].state
            np.savez(out / f"checkpoint_{model}_seed{seed}.npz", mean=state.mean, scale=state.scale, weight=state.weight, bias=state.bias)

    # Equal-weight investable-universe benchmark, evaluated with the same
    # one-day return and cost convention.  It is not a ranking model.
    benchmark_frame = _frame(
        mte, dataset=dataset, split="test", model="MARKET_EW", seed=seed,
        score=np.zeros(len(mte), dtype=np.float32), endpoint=yte, path=path_te,
    )
    benchmark_top_k = max(1, int(benchmark_frame.groupby("as_of_date")["stock"].nunique().max()))
    benchmark_cfg = PortfolioConfig(
        top_k=benchmark_top_k, drop_n=benchmark_top_k, min_holding_days=1,
        cost_rate=MARKET_PORTFOLIO_CONFIGS[dataset].cost_rate,
        initial_capital=MARKET_PORTFOLIO_CONFIGS[dataset].initial_capital,
        annualization_days=MARKET_PORTFOLIO_CONFIGS[dataset].annualization_days,
    )
    benchmark = backtest_long_only(
        benchmark_frame, return_col="next_day_return", config=benchmark_cfg,
        dataset=dataset, model="MARKET_EW", seed=seed,
    )
    portfolio_summaries.append(dict(benchmark["summary"], signal_mode="equal_weight_market"))
    benchmark["daily"].to_csv(out / f"portfolio_daily_MARKET_EW_seed{seed}.csv", index=False, encoding="utf-8-sig")
    benchmark["holdings"].to_csv(out / f"holdings_daily_MARKET_EW_seed{seed}.csv", index=False, encoding="utf-8-sig")
    benchmark["trades"].to_csv(out / f"trades_MARKET_EW_seed{seed}.csv", index=False, encoding="utf-8-sig")

    pred = pd.concat(all_prediction_frames, ignore_index=True) if all_prediction_frames else pd.DataFrame()
    pred.to_csv(out / "sorting_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_rows).to_csv(out / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(out / "test_cross_section_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(portfolio_summaries).to_csv(out / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    # Keep full daily/holdings/trades for S4 and raw S0 when available.
    for model in ("S0", "S4"):
        frame = pred[(pred["split"] == "test") & (pred["model"] == model)] if not pred.empty else pd.DataFrame()
        if not frame.empty:
            pf = backtest_long_only(frame, return_col="next_day_return", config=MARKET_PORTFOLIO_CONFIGS[dataset], dataset=dataset, model=model, seed=seed)
            pf["daily"].to_csv(out / f"portfolio_daily_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
            pf["holdings"].to_csv(out / f"holdings_daily_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
            pf["trades"].to_csv(out / f"trades_{model}_seed{seed}.csv", index=False, encoding="utf-8-sig")
    manifest = {"dataset": dataset, "seed": seed, "cache_gate": gate, "config": asdict(config), "label": label,
                "formal_execution": bool(torch is not None),
                "model_order": MODELS, "model_sources": {m: ("raw_signal" if m == "S0" else ("pytorch" if torch is not None else "numpy_smoke_fallback")) for m in MODELS},
                "test_usage": "inference_and_final_evaluation_only",
                "training_sampling": {"dates_per_epoch": config.dates_per_epoch, "batch_dates": config.batch_dates,
                                      "note": "每个 epoch 对训练日期做固定 seed 的无放回抽样；不是每 epoch 全日期"},
                "portfolio_pnl_source": "next_day_return; 10-day Endpoint is ranking label only",
                "survivorship_warning": gate["survivorship_warning"]}
    (out / "data_manifest.json").write_text(json.dumps(_json_clean(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(_json_clean(asdict(config)), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "environment.txt").write_text(f"python={platform.python_version()}\nplatform={platform.platform()}\ntorch={getattr(torch, '__version__', 'unavailable')}\n", encoding="utf-8")
    (out / "git_commit.txt").write_text(git_commit(Path(__file__).resolve().parents[1]) + "\n", encoding="utf-8")
    status = "可验证" if all_prediction_frames and torch is not None else ("smoke_only" if all_prediction_frames and allow_numpy_fallback else "无法验证")
    result = {"status": status, "metric_status": "可验证" if status == "可验证" else "无法验证", "dataset": dataset, "seed": seed, "rows": int(len(pred)), "models": list(MODELS), "output_dir": str(out.resolve()), "unavailable_reason": "" if status == "可验证" else ("仅 CPU smoke fallback；不是正式 PyTorch S1-S4 结果" if status == "smoke_only" else "PyTorch 未安装，正式 S1-S4 结果无法验证")}
    (out / "metric_status.json").write_text(json.dumps(_json_clean(result), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metric_status_csv(out, result)
    return result


def run_numpy_experiment(cache_dir: str | Path, output_dir: str | Path, dataset: str, *, seed: int = 100, config: SortingConfig = SortingConfig(), label: str = "endpoint") -> dict[str, Any]:
    return run_experiment(cache_dir, output_dir, dataset, seed=seed, config=config, label=label, allow_numpy_fallback=True)


def _completed_seed_result(seed_out: Path, dataset: str, seed: int) -> dict[str, Any] | None:
    """Return an existing formally completed seed result for safe queue resume.

    A directory is skipped only when its status is verifiable and all required
    seed-level artifacts are present and non-empty. Partial or smoke outputs are
    rerun, so resuming never promotes incomplete results.
    """
    status_path = seed_out / "metric_status.json"
    required = (
        "sorting_predictions.csv", "training_history.csv", "validation_metrics.csv",
        "test_cross_section_metrics.csv", "portfolio_summary.csv", "config.json",
        "data_manifest.json", "metric_status.csv", "environment.txt", "git_commit.txt",
    )
    if not status_path.exists() or any(not (seed_out / name).exists() or (seed_out / name).stat().st_size == 0 for name in required):
        return None
    try:
        result = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if result.get("status") != "可验证" or result.get("dataset") != dataset or int(result.get("seed", -1)) != int(seed):
        return None
    result["output_dir"] = str(seed_out.resolve())
    result["resumed_from_existing"] = True
    return result

def run_all_markets_seeds(
    cache_root: str | Path,
    output_root: str | Path = "train3/sorting_reward_v1",
    *,
    markets: Sequence[str] = MARKETS,
    seeds: Sequence[int] = (100, 101, 102, 103, 104),
    config: SortingConfig = SortingConfig(),
    label: str = "endpoint",
    allow_numpy_fallback: bool = False,
) -> dict[str, Any]:
    """Run the formal market × seed matrix and write auditable aggregates.

    Each seed has an isolated directory.  Aggregate files are concatenations,
    never replacements for seed-level outputs.  A market is marked formally
    verifiable only if every requested seed completed on the PyTorch path.
    Missing caches, partial seeds and smoke fallbacks remain ``无法验证`` at the
    aggregate level so the dashboard cannot accidentally promote them.
    """
    root = Path(cache_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    requested_seeds = tuple(int(x) for x in seeds)
    market_results: dict[str, Any] = {}
    for dataset in markets:
        market_out = out_root / dataset
        market_out.mkdir(parents=True, exist_ok=True)
        cache_dir = root / dataset
        seed_results: list[dict[str, Any]] = []
        rank_frames: list[pd.DataFrame] = []
        val_frames: list[pd.DataFrame] = []
        port_frames: list[pd.DataFrame] = []
        history_frames: list[pd.DataFrame] = []
        daily_frames: list[pd.DataFrame] = []
        holding_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        sensitivity_frames: list[pd.DataFrame] = []
        for seed in requested_seeds:
            seed_out = market_out / f"seed{seed}"
            result = _completed_seed_result(seed_out, dataset, seed)
            if result is None:
                result = run_experiment(
                    cache_dir, seed_out, dataset, seed=seed, config=config, label=label,
                    allow_numpy_fallback=allow_numpy_fallback,
                )
            else:
                print(f"[{dataset}][seed={seed}] resume: using completed seed output", flush=True)
            seed_results.append(result)
            for filename, bucket in (("test_cross_section_metrics.csv", rank_frames),
                                     ("validation_metrics.csv", val_frames),
                                     ("portfolio_summary.csv", port_frames),
                                     ("training_history.csv", history_frames)):
                path = seed_out / filename
                if path.exists() and path.stat().st_size > 0:
                    try:
                        frame = pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        frame = pd.DataFrame()
                    if not frame.empty:
                        frame["seed_dir"] = str(seed_out)
                        bucket.append(frame)
            for pattern, bucket in (("portfolio_daily_*.csv", daily_frames),
                                    ("holdings_daily_*.csv", holding_frames),
                                    ("trades_*.csv", trade_frames),
                                    ("cost_sensitivity_*.csv", sensitivity_frames)):
                for path in sorted(seed_out.glob(pattern)):
                    try:
                        frame = pd.read_csv(path) if path.stat().st_size > 0 else pd.DataFrame()
                    except pd.errors.EmptyDataError:
                        frame = pd.DataFrame()
                    if not frame.empty:
                        frame["seed_dir"] = str(seed_out)
                        bucket.append(frame)
        def concat_write(frames: list[pd.DataFrame], filename: str) -> int:
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            frame.to_csv(market_out / filename, index=False, encoding="utf-8-sig")
            return int(len(frame))
        counts = {
            "test_cross_section_metrics.csv": concat_write(rank_frames, "test_cross_section_metrics.csv"),
            "validation_metrics.csv": concat_write(val_frames, "validation_metrics.csv"),
            "portfolio_summary.csv": concat_write(port_frames, "portfolio_summary.csv"),
            "training_history.csv": concat_write(history_frames, "training_history.csv"),
            "portfolio_daily.csv": concat_write(daily_frames, "portfolio_daily.csv"),
            "holdings_daily.csv": concat_write(holding_frames, "holdings_daily.csv"),
            "trades.csv": concat_write(trade_frames, "trades.csv"),
            "cost_sensitivity.csv": concat_write(sensitivity_frames, "cost_sensitivity.csv"),
        }
        formal = (torch is not None and len(seed_results) == len(requested_seeds)
                  and all(r.get("status") == "可验证" for r in seed_results))
        reasons = [str(r.get("unavailable_reason", "")) for r in seed_results if r.get("status") != "可验证"]
        market_status = "可验证" if formal else "无法验证"
        gate = validate_cache(cache_dir, dataset, config=config)
        manifest = {
            "dataset": dataset, "cache_dir": str(cache_dir.resolve()),
            "seeds": list(requested_seeds), "seed_results": seed_results,
            "cache_gate": gate, "config": asdict(config), "label": label,
            "model_order": list(MODELS),
            "test_usage": "inference_and_final_evaluation_only",
            "survivorship_warning": gate.get("survivorship_warning"),
            "formal_status_rule": "all requested seeds must be formal PyTorch results",
        }
        (market_out / "data_manifest.json").write_text(json.dumps(_json_clean(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        (market_out / "config.json").write_text(json.dumps(_json_clean(asdict(config)), ensure_ascii=False, indent=2), encoding="utf-8")
        status_payload = {
            "dataset": dataset, "status": market_status, "metric_status": market_status,
            "seeds": list(requested_seeds), "seed_results": seed_results,
            "cache_gate": gate, "rows": counts,
            "unavailable_reason": "；".join(dict.fromkeys(x for x in reasons if x)) if not formal else "",
            "survivorship_warning": gate.get("survivorship_warning"),
        }
        (market_out / "metric_status.json").write_text(json.dumps(_json_clean(status_payload), ensure_ascii=False, indent=2), encoding="utf-8")
        _write_metric_status_csv(market_out, status_payload)
        (market_out / "git_commit.txt").write_text(git_commit(Path(__file__).resolve().parents[1]) + "\n", encoding="utf-8")
        market_results[dataset] = status_payload
    all_rank = []
    all_port = []
    for dataset in markets:
        market_out = out_root / dataset
        for filename, target in (("test_cross_section_metrics.csv", all_rank), ("portfolio_summary.csv", all_port)):
            path = market_out / filename
            if path.exists() and path.stat().st_size > 0:
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                if not frame.empty:
                    target.append(frame)
    pd.concat(all_rank, ignore_index=True).to_csv(out_root / "aggregate_sorting_metrics.csv", index=False, encoding="utf-8-sig") if all_rank else pd.DataFrame().to_csv(out_root / "aggregate_sorting_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_port, ignore_index=True).to_csv(out_root / "aggregate_portfolio_summary.csv", index=False, encoding="utf-8-sig") if all_port else pd.DataFrame().to_csv(out_root / "aggregate_portfolio_summary.csv", index=False, encoding="utf-8-sig")
    # Explicit mean/std over seeds: no cross-market pooling and no replacement of
    # the seed-level rows.  Only numeric metric columns are summarized.
    summary_rows: list[dict[str, Any]] = []
    for dataset in markets:
        frame_path = out_root / dataset / "test_cross_section_metrics.csv"
        if not frame_path.exists(): continue
        try:
            frame = pd.read_csv(frame_path) if frame_path.stat().st_size > 0 else pd.DataFrame()
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        for model, group in frame.groupby("model") if not frame.empty and "model" in frame.columns else []:
            numeric = group.select_dtypes(include=[np.number])
            row = {"dataset": dataset, "model": model, "seed_count": int(group["seed"].nunique()) if "seed" in group else 0}
            metric_columns = ("ic_mean", "rankic_mean", "rankic_std", "rankic_median",
                              "rankic_positive_ratio", "rankic_nw_t", "q5_q1_mean",
                              "top_k_return", "top_k_excess_return_mean", "bottom_k_return_mean",
                              "long_short_return_mean", "direction_accuracy_mean")
            for col in metric_columns:
                if col in numeric:
                    row[f"{col}_mean_over_seeds"] = float(numeric[col].mean()) if numeric[col].notna().any() else np.nan
                    row[f"{col}_std_over_seeds"] = float(numeric[col].std(ddof=1)) if numeric[col].notna().sum() > 1 else np.nan
            port_path = out_root / dataset / "portfolio_summary.csv"
            if port_path.exists() and port_path.stat().st_size > 0:
                try: port = pd.read_csv(port_path)
                except pd.errors.EmptyDataError: port = pd.DataFrame()
                port = port[port.get("model", pd.Series(dtype=str)) == model] if not port.empty else port
                for col in ("annualized_net_return", "net_cumulative_return", "gross_cumulative_return",
                            "final_wealth", "absolute_profit", "maximum_drawdown", "sharpe",
                            "sortino", "annualized_turnover", "cumulative_transaction_cost"):
                    if col in port:
                        vals = pd.to_numeric(port[col], errors="coerce")
                        row[f"{col}_mean_over_seeds"] = float(vals.mean()) if vals.notna().any() else np.nan
                        row[f"{col}_std_over_seeds"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            row["metric_status"] = "可验证" if row["seed_count"] == len(requested_seeds) and np.isfinite(row.get("rankic_mean_mean_over_seeds", np.nan)) else "无法验证"
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_root / "multi_seed_summary.csv", index=False, encoding="utf-8-sig")
    root_status = "可验证" if market_results and all(v["status"] == "可验证" for v in market_results.values()) else "无法验证"
    return {"status": root_status, "markets": market_results, "output_dir": str(out_root.resolve()), "seeds": list(requested_seeds)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=False, default="train3/sorting_reward_v1/cache")
    ap.add_argument("--output-dir", default="train3/sorting_reward_v1")
    ap.add_argument("--dataset", required=False, choices=MARKETS)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-numpy-fallback", action="store_true")
    ap.add_argument("--all-markets", action="store_true", help="run CSI800/CSI300/S&P500 for all five seeds")
    args = ap.parse_args(argv)
    if args.all_markets:
        result = run_all_markets_seeds(args.cache_dir, args.output_dir, allow_numpy_fallback=args.allow_numpy_fallback or args.smoke)
    else:
        if not args.dataset:
            ap.error("--dataset is required unless --all-markets is used")
        result = run_experiment(args.cache_dir, args.output_dir, args.dataset, seed=args.seed, allow_numpy_fallback=args.allow_numpy_fallback or args.smoke)
    print(json.dumps(_json_clean(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "可验证" else 2


if __name__ == "__main__":
    raise SystemExit(main())

