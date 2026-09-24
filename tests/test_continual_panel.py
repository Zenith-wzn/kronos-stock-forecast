import unittest
from datetime import datetime
import numpy as np
import pandas as pd

from continual_abcs.panel import (
    build_master_calendar,
    build_eligible_pairs,
    matured_pairs,
    month_end_update_dates,
    effective_version_for_date,
)


def frame(code, dates, missing=()):
    dates = [pd.Timestamp(x) for x in dates if pd.Timestamp(x) not in {pd.Timestamp(m) for m in missing}]
    n = len(dates)
    return pd.DataFrame({
        "date": dates, "code": code,
        "open": np.arange(n, dtype=float) + 10,
        "high": np.arange(n, dtype=float) + 11,
        "low": np.arange(n, dtype=float) + 9,
        "close": np.arange(n, dtype=float) + 10.5,
        "volume": np.ones(n),
    }).set_index("date", drop=False)


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=12, freq="D")
        self.frames = {f"s{i}": frame(f"s{i}", self.dates) for i in range(3)}

    def test_eligibility_requires_exact_90_10_positions(self):
        pairs = build_eligible_pairs(self.frames, self.dates, lookback=3, pred_len=2)
        self.assertEqual(pairs.iloc[0].as_of_date, "2024-01-03")
        self.assertEqual(pairs.iloc[0].input_start_date, "2024-01-01")
        self.assertEqual(pairs.iloc[0].label_end_date, "2024-01-05")
        self.assertEqual(pairs.stock.nunique(), 3)

    def test_missing_position_excludes_only_affected_stock_date(self):
        frames = dict(self.frames)
        frames["s1"] = frame("s1", self.dates, missing=["2024-01-02"])
        pairs = build_eligible_pairs(frames, self.dates, lookback=3, pred_len=2)
        row = pairs[pairs.as_of_date == "2024-01-03"]
        self.assertNotIn("s1", set(row.stock))
        self.assertEqual(len(row), 2)

    def test_label_maturity_is_not_before_label_end(self):
        pairs = build_eligible_pairs(self.frames, self.dates, lookback=3, pred_len=2)
        mature = matured_pairs(pairs, pd.Timestamp("2024-01-05"))
        self.assertTrue((pd.to_datetime(mature.label_end_date) <= pd.Timestamp("2024-01-05")).all())
        self.assertFalse((mature.as_of_date == "2024-01-04").any())

    def test_month_end_update_activates_next_selected_date(self):
        dates = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
        updates = month_end_update_dates(dates)
        self.assertEqual(updates, [pd.Timestamp("2024-01-31")])
        self.assertEqual(effective_version_for_date(pd.Timestamp("2024-01-31"), updates), 0)
        self.assertEqual(effective_version_for_date(pd.Timestamp("2024-02-01"), updates), 1)


if __name__ == "__main__":
    unittest.main()
