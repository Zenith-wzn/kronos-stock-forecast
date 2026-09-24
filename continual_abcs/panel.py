"""Strict panel construction for the CSI300 A/B/C experiments."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")


def _as_calendar(values: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, errors="raise").normalize()).drop_duplicates().sort_values()


def _date_hash(values: Sequence[pd.Timestamp]) -> str:
    text = "|".join(pd.Timestamp(x).date().isoformat() for x in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_row(frame: pd.DataFrame, date: pd.Timestamp) -> bool:
    if date not in frame.index:
        return False
    row = frame.loc[date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    try:
        values = np.asarray([row[c] for c in REQUIRED_VALUE_COLUMNS], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all() and (values[:4] > 0).all() and values[4] >= 0)


def build_master_calendar(
    frames: Mapping[str, pd.DataFrame],
    min_cross_section: int = 30,
    calendar_min_fraction: float = 0.5,
) -> pd.DatetimeIndex:
    """Build a deterministic calendar from dates observed by enough stocks."""
    if not frames:
        raise ValueError("frames must not be empty")
    if min_cross_section < 1:
        raise ValueError("min_cross_section must be positive")
    if not 0 < calendar_min_fraction <= 1:
        raise ValueError("calendar_min_fraction must be in (0, 1]")

    counts: dict[pd.Timestamp, int] = {}
    for frame in frames.values():
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce").dropna()).normalize().unique()
        for date in dates:
            counts[date] = counts.get(date, 0) + 1
    threshold = max(min_cross_section, int(np.ceil(len(frames) * calendar_min_fraction)))
    selected = sorted(date for date, count in counts.items() if count >= threshold)
    if not selected:
        raise ValueError(f"no calendar dates meet threshold={threshold}")
    return pd.DatetimeIndex(selected)


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result.columns:
        result = result.copy()
        result["date"] = result.index
    result = result.reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return result.set_index("date", drop=False)


def build_eligible_pairs(
    frames: Mapping[str, pd.DataFrame],
    calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    lookback: int = 90,
    pred_len: int = 10,
    split_dates: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    min_cross_section: int = 1,
) -> pd.DataFrame:
    """Create one row per eligible stock/date using exact calendar positions.

    The validity matrix is computed once. Rows are emitted as per-date blocks
    rather than millions of Python dictionaries; this keeps the 503-stock
    S&P500 preflight bounded while preserving the locked 90->10 semantics.
    """
    if lookback < 1 or pred_len < 1:
        raise ValueError("lookback and pred_len must be positive")
    master = _as_calendar(calendar)
    stocks = sorted(str(stock) for stock in frames)
    normalised = {stock: _normalise_frame(frames[stock]) for stock in stocks}
    validity = np.zeros((len(stocks), len(master)), dtype=bool)
    for row_number, stock in enumerate(stocks):
        frame = normalised[stock].reindex(master)
        try:
            values = frame[list(REQUIRED_VALUE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            validity[row_number] = (
                np.isfinite(values).all(axis=1)
                & (values[:, :4] > 0).all(axis=1)
                & (values[:, 4] >= 0)
            )
        except (KeyError, TypeError, ValueError):
            validity[row_number] = False

    invalid = (~validity).astype(np.int16)
    prefix = np.concatenate(
        [np.zeros((len(stocks), 1), dtype=np.int32), np.cumsum(invalid, axis=1, dtype=np.int32)],
        axis=1,
    )
    split_arrays = []
    if split_dates:
        split_arrays = [(str(name), pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()) for name, (start, end) in split_dates.items()]

    def split_for(date: pd.Timestamp) -> str:
        if not split_arrays:
            return "unspecified"
        for name, start, end in split_arrays:
            if start <= date <= end:
                return name
        return "excluded"

    columns = ["split", "stock", "as_of_date", "input_start_date", "input_end_date", "label_start_date", "label_end_date", "input_dates", "label_dates", "input_calendar_hash", "label_calendar_hash"]
    blocks: list[pd.DataFrame] = []
    for pos, as_of in enumerate(master):
        if pos < lookback - 1 or pos + pred_len >= len(master):
            continue
        split = split_for(as_of)
        if split == "excluded":
            continue
        start = pos - lookback + 1
        end = pos + pred_len + 1
        eligible_idx = np.flatnonzero((prefix[:, end] - prefix[:, start]) == 0)
        if eligible_idx.size < min_cross_section:
            continue
        input_dates = master[start : pos + 1]
        label_dates = master[pos + 1 : end]
        block = pd.DataFrame({
            "split": split,
            "stock": [stocks[i] for i in eligible_idx],
            "as_of_date": as_of.date().isoformat(),
            "input_start_date": input_dates[0].date().isoformat(),
            "input_end_date": input_dates[-1].date().isoformat(),
            "label_start_date": label_dates[0].date().isoformat(),
            "label_end_date": label_dates[-1].date().isoformat(),
            "input_dates": "|".join(x.date().isoformat() for x in input_dates),
            "label_dates": "|".join(x.date().isoformat() for x in label_dates),
            "input_calendar_hash": _date_hash(input_dates),
            "label_calendar_hash": _date_hash(label_dates),
        })
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=columns)
def matured_pairs(pairs: pd.DataFrame, update_date: pd.Timestamp) -> pd.DataFrame:
    """Return only samples whose complete future label is known by update_date."""
    if pairs.empty:
        return pairs.copy()
    result = pairs.copy()
    update = pd.Timestamp(update_date).normalize()
    result["label_end_date"] = pd.to_datetime(result["label_end_date"]).dt.normalize()
    return result[result["label_end_date"] <= update].reset_index(drop=True)


def month_end_update_dates(test_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the final selected test date in each calendar month."""
    dates = _as_calendar(test_dates)
    if len(dates) == 0:
        return []
    frame = pd.DataFrame({"date": dates})
    frame["month"] = frame["date"].dt.to_period("M")
    last_date = dates[-1]
    return [
        pd.Timestamp(group["date"].max())
        for _, group in frame.groupby("month", sort=True)
        if pd.Timestamp(group["date"].max()) < last_date
    ]


def effective_version_for_date(date: pd.Timestamp, update_dates: Sequence[pd.Timestamp]) -> int:
    """Version active before prediction on date; month-end updates activate next date."""
    target = pd.Timestamp(date).normalize()
    return sum(pd.Timestamp(update).normalize() < target for update in update_dates)


