from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from continual_abcs.head import (
    make_same_date_pairs,
    pairwise_rank_loss_numpy,
)
from continual_abcs.features import FeatureCacheWriter
from continual_abcs.portfolio import PortfolioConfig, backtest_long_only
from continual_abcs.sorting_reward import SortingConfig, train_standardizer, validate_cache, _date_batches, _json_clean


class SortingRewardTests(unittest.TestCase):
    def test_equal_labels_are_excluded_and_pairs_are_same_date(self):
        dates = np.array(["2025-01-01"] * 3 + ["2025-01-02"] * 3)
        y = np.array([1.0, 1.0, 0.0, 0.4, 0.2, 0.2])
        pairs = make_same_date_pairs(dates, y, max_pairs_per_date=100, min_items=2, seed=7)
        self.assertEqual(len(pairs.left), 4)
        self.assertTrue(np.all(dates[pairs.left] == dates[pairs.right]))
        self.assertTrue(np.all(y[pairs.left] != y[pairs.right]))

    def test_correct_order_has_lower_loss(self):
        y = np.array([3.0, 2.0, 1.0])
        self.assertLess(pairwise_rank_loss_numpy(y, y), pairwise_rank_loss_numpy(y[::-1], y))

    def test_pair_sampling_is_reproducible(self):
        dates = np.array(["2025-01-01"] * 20)
        y = np.arange(20, dtype=float)
        a = make_same_date_pairs(dates, y, max_pairs_per_date=5, seed=100)
        b = make_same_date_pairs(dates, y, max_pairs_per_date=5, seed=100)
        self.assertTrue(np.array_equal(a.left, b.left)); self.assertTrue(np.array_equal(a.right, b.right))

    def test_datetime64_batches_are_non_empty_and_date_local(self):
        dates = np.array(["2025-01-01"] * 2 + ["2025-01-02"] * 3, dtype="datetime64[ns]")
        batches = list(_date_batches(dates, batch_dates=1, seed=100))
        self.assertEqual(sum(len(x) for x in batches), len(dates))
        self.assertTrue(all(len(x) > 0 for x in batches))
        for indices in batches:
            self.assertEqual(len(np.unique(dates[indices])), 1)

    def test_datetime64_batch_sampling_is_reproducible(self):
        dates = np.repeat(np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-11"), dtype="datetime64[D]"), 2).astype("datetime64[ns]")
        a = list(_date_batches(dates, batch_dates=2, seed=7, dates_per_epoch=6))
        b = list(_date_batches(dates, batch_dates=2, seed=7, dates_per_epoch=6))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_json_clean_handles_zero_dimensional_numpy_arrays(self):
        self.assertEqual(_json_clean(np.array(3.5)), 3.5)
        self.assertIsNone(_json_clean(np.array(np.nan)))

    def test_standardizer_uses_train_stats(self):
        stats = train_standardizer(np.array([[0., 0.], [2., 2.]]), np.array([0., 2.]))
        x, y = (np.array([[100., 100.]]), np.array([100.]))
        from continual_abcs.sorting_reward import apply_standardizer
        xz, yz = apply_standardizer(x, y, stats)
        self.assertGreater(float(xz.mean()), 10.0); self.assertGreater(float(yz.mean()), 10.0)

    def test_cache_row_mismatch_fails_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ("train", "val", "test"):
                meta = pd.DataFrame({"dataset": ["CSI800"], "split": [split], "stock": ["s0"], "as_of_date": ["2025-01-01"], "label_start_date": ["2025-01-02"], "label_end_date": ["2025-01-11"], "current_close": [10.], "entry_price": [10.], "exit_price": [11.], "actual_endpoint_return": [.1], "actual_path_mean_return": [.05], "raw_signal": [.1]})
                FeatureCacheWriter(root).write(split, meta, np.ones((1, 4)), np.ones((1, 10)), np.array([.1]))
            gate = validate_cache(root, "CSI800", config=SortingConfig(min_cross_section=1))
            self.assertEqual(gate["status"], "无法验证")
            self.assertTrue(any("日期交集" in x for x in gate["warnings"]))

if __name__ == "__main__":
    unittest.main()
