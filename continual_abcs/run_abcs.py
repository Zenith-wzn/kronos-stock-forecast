"""A/B/C test-then-train scheduling primitives."""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .panel import month_end_update_dates


def update_schedule(test_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> list[pd.Timestamp]:
    return month_end_update_dates(test_dates)


def simulate_test_then_train(
    test_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    update_dates: Sequence[pd.Timestamp],
    matured_by_update: Mapping[pd.Timestamp, pd.DataFrame] | None = None,
    *,
    group: str = "C",
) -> pd.DataFrame:
    """Emit the version/update timeline without performing model training."""
    if group not in {"A", "B", "C"}:
        raise ValueError("group must be A, B, or C")
    dates = pd.DatetimeIndex(pd.to_datetime(test_dates)).normalize().sort_values()
    updates = {pd.Timestamp(x).normalize() for x in update_dates}
    rows: list[dict[str, object]] = []
    version = 0
    for date in dates:
        active_version = version
        is_update = group == "C" and date in updates
        rows.append(
            {
                "as_of_date": date,
                "model_version": active_version,
                "updated_after_prediction": bool(is_update),
                "matured_count": int(len((matured_by_update or {}).get(date, pd.DataFrame())) if is_update else 0),
            }
        )
        if is_update:
            version += 1
    return pd.DataFrame(rows)
