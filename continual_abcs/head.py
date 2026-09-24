"""Shared ranking-head utilities for fixed and continual groups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def make_same_date_batches(
    dates: Iterable[object], batch_size: int, seed: int = 0, shuffle: bool = True
) -> list[np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.asarray(list(dates))
    groups: dict[object, list[int]] = {}
    for index, date in enumerate(values.tolist()):
        groups.setdefault(date, []).append(index)
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    group_values = list(groups.values())
    if shuffle:
        rng.shuffle(group_values)
    for indices in group_values:
        indices = list(indices)
        if shuffle:
            rng.shuffle(indices)
        batches.extend(
            np.asarray(indices[i : i + batch_size], dtype=np.int64)
            for i in range(0, len(indices), batch_size)
        )
    return batches


def pairwise_rank_loss_numpy(scores: np.ndarray, targets: np.ndarray, scale: float = 2.0) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    delta_target = targets[:, None] - targets[None, :]
    mask = delta_target != 0
    if not mask.any():
        return 0.0
    delta_score = scores[:, None] - scores[None, :]
    signed = np.sign(delta_target[mask]) * delta_score[mask]
    return float(np.logaddexp(0.0, -scale * signed).mean())


@dataclass(frozen=True)
class NumpyHeadState:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    ridge: float


class NumpyRankingHead:
    """Deterministic ridge ranking head used by CPU smoke tests and cache checks.

    The production GPU path may replace this implementation with the PyTorch head,
    but it must preserve the same feature/label and date-batch semantics.
    """

    def __init__(self, state: NumpyHeadState):
        self.state = state

    @property
    def input_dim(self) -> int:
        return int(self.state.weight.shape[0])

    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("features have an unexpected shape")
        z = (x - self.state.mean) / self.state.scale
        return (z @ self.state.weight + self.state.bias).astype(np.float64)

    def copy(self) -> "NumpyRankingHead":
        s = self.state
        return NumpyRankingHead(NumpyHeadState(s.mean.copy(), s.scale.copy(), s.weight.copy(), s.bias, s.ridge))


def _scalar_targets(targets: np.ndarray) -> np.ndarray:
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.mean(axis=1)
    raise ValueError("targets must be one- or two-dimensional")


def fit_numpy_ranking_head(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float = 1.0e-2,
    warm_start: NumpyRankingHead | None = None,
) -> NumpyRankingHead:
    """Fit a deterministic scalar head to the Eq.13 target.

    The ridge fit is intentionally small and dependency-free. It is used for the
    CPU smoke test; the GPU experiment can use the same cached arrays with the
    PyTorch implementation without changing the schedule.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("features must be a non-empty 2-D array")
    y = _scalar_targets(targets)
    if len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite and aligned")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    reg = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return NumpyRankingHead(
        NumpyHeadState(mean.astype(np.float64), scale.astype(np.float64), coef[:-1], float(coef[-1]), float(ridge))
    )


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class RankingHead(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
            )

        def forward(self, features):
            return self.net(features).squeeze(-1)
else:
    class RankingHead:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("RankingHead requires PyTorch")


@dataclass(frozen=True)
class PairSample:
    """Indices and signed labels for one same-date ranking problem."""
    left: np.ndarray
    right: np.ndarray
    sign: np.ndarray


def make_same_date_pairs(
    dates: Iterable[object],
    targets: np.ndarray,
    *,
    max_pairs_per_date: int = 4096,
    min_items: int = 3,
    seed: int = 0,
) -> PairSample:
    """Create deterministic, within-date ordered pairs only.

    Equal targets are excluded.  Sampling is performed independently per date,
    so a large cross-section cannot dominate the ranking objective.
    """
    if max_pairs_per_date < 1:
        raise ValueError("max_pairs_per_date must be positive")
    y = np.asarray(targets, dtype=float).reshape(-1)
    d = np.asarray(list(dates))
    if len(d) != len(y):
        raise ValueError("dates and targets must have the same length")
    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    sign_parts: list[np.ndarray] = []
    groups: dict[object, list[int]] = {}
    for i, date in enumerate(d.tolist()):
        groups.setdefault(date, []).append(i)
    for indices in groups.values():
        if len(indices) < min_items:
            continue
        ix = np.asarray(indices, dtype=np.int64)
        delta = y[ix, None] - y[ix][None, :]
        valid = np.triu(delta != 0, k=1)
        a, b = np.where(valid)
        if len(a) == 0:
            continue
        if len(a) > max_pairs_per_date:
            chosen = rng.choice(len(a), size=max_pairs_per_date, replace=False)
            a, b = a[chosen], b[chosen]
        li, ri = ix[a], ix[b]
        signs = np.sign(y[li] - y[ri]).astype(np.float32)
        left_parts.append(li); right_parts.append(ri); sign_parts.append(signs)
    if not left_parts:
        empty = np.empty(0, dtype=np.int64)
        return PairSample(empty, empty, np.empty(0, dtype=np.float32))
    return PairSample(np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(sign_parts))


def pairwise_rank_loss_torch(scores, targets, *, scale: float = 2.0):
    """Torch pairwise logistic loss; pairs in the supplied tensors must share a date."""
    if torch is None:
        raise ImportError("pairwise_rank_loss_torch requires PyTorch")
    scores = scores.reshape(-1)
    targets = targets.reshape(-1)
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    delta_target = targets[:, None] - targets[None, :]
    mask = torch.triu(delta_target != 0, diagonal=1)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    delta_score = scores[:, None] - scores[None, :]
    signed = torch.sign(delta_target[mask]) * delta_score[mask]
    return torch.nn.functional.softplus(-float(scale) * signed).mean()


def date_balanced_pairwise_loss_torch(
    scores, targets, dates, *, scale: float = 2.0, min_items: int = 3,
    max_pairs_per_date: int = 4096, seed: int = 0,
):
    """Average deterministic, sampled same-date pair losses equally over dates."""
    if torch is None:
        raise ImportError("date_balanced_pairwise_loss_torch requires PyTorch")
    d = np.asarray(list(dates))
    if len(d) != int(scores.numel()) or len(d) != int(targets.numel()):
        raise ValueError("dates, scores and targets must be aligned")
    pairs = make_same_date_pairs(
        d, targets.detach().cpu().numpy(), max_pairs_per_date=max_pairs_per_date,
        min_items=min_items, seed=seed,
    )
    if len(pairs.left) == 0:
        return scores.sum() * 0.0
    losses = []
    # Keep datetime64 values in NumPy space. datetime64[ns].tolist() may yield
    # integer nanoseconds, which do not compare equal to the original array.
    pair_dates = d[pairs.left]
    for date in np.unique(pair_dates):
        mask = pair_dates == date
        if not np.any(mask):
            continue
        left = torch.as_tensor(pairs.left[mask], device=scores.device, dtype=torch.long)
        right = torch.as_tensor(pairs.right[mask], device=scores.device, dtype=torch.long)
        signs = torch.as_tensor(pairs.sign[mask], device=scores.device, dtype=scores.dtype)
        loss = torch.nn.functional.softplus(-float(scale) * signs * (scores[left] - scores[right])).mean()
        if bool(torch.isfinite(loss)):
            losses.append(loss)
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()
