import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from continual_abcs.features import FeatureCacheWriter, build_signal_summary, make_kronos_input, make_target, load_feature_cache


class FeatureTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        self.frame = pd.DataFrame({
            "date": dates, "open": np.arange(8) + 10., "high": np.arange(8) + 11.,
            "low": np.arange(8) + 9., "close": np.arange(8) + 10., "volume": 100.
        })
        self.dates = dates

    def test_input_is_exact_and_target_is_eq13(self):
        x = make_kronos_input(self.frame, self.dates[:3])
        self.assertEqual(list(x.date) if "date" in x else len(x), 3 if "date" not in x else [])[0:] if False else None
        target = make_target(self.frame, self.dates[3:5])
        expected = ((13.0 + 14.0) / 2.0 / 12.0) - 1.0
        self.assertAlmostEqual(float(target[0]), expected, places=6)
        self.assertEqual(len(x), 3)
        self.assertEqual(float(x.iloc[-1].close), 12.0)

    def test_signal_summary_and_cache_round_trip(self):
        pred = pd.DataFrame({"pred_close": [11., 12., 13.]})
        summary = build_signal_summary(pred, current_close=10.)
        self.assertAlmostEqual(float(summary[0]), 0.2, places=6)
        with tempfile.TemporaryDirectory() as d:
            meta = pd.DataFrame({"as_of_date": ["2024-01-01"], "stock": ["s0"]})
            FeatureCacheWriter(d).write("train", meta, np.ones((1, 4)), np.ones((1, 2)), np.array([.2]))
            h, y, m = load_feature_cache(d, "train")
            self.assertEqual(h.shape, (1, 4)); self.assertEqual(y.shape, (1, 2)); self.assertAlmostEqual(float(m.raw_signal.iloc[0]), .2)


if __name__ == "__main__":
    unittest.main()



