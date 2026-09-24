import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.run_continual_csi300_abc_experiment import (
    Config,
    _accept_candidate_mae,
    _mean_absolute_error,
    _mean_daily_rankic,
    _run,
    _select_replay_rows,
    _select_test_rows,
)


class ContinualExperimentTests(unittest.TestCase):
    def test_mae_gate_boundaries(self):
        champion = 0.02
        for candidate, accepted in [(0.019, True), (0.02, True),
                                    (0.0200000001, False), (0.0205, False),
                                    (champion + 0.001, False), (np.nan, False),
                                    (np.inf, False), (-np.inf, False)]:
            with self.subTest(candidate=candidate):
                self.assertEqual(_accept_candidate_mae(candidate, champion, 0.0), accepted)
        for tolerance in (-0.001, np.nan, np.inf, 0.001):
            self.assertFalse(_accept_candidate_mae(0.019, champion, tolerance))
        self.assertEqual(Config(cache_dir="x", output_dir="y").validation_tolerance, 0.0)

    def test_mae_is_error_of_path_scalar_not_mean_horizon_error(self):
        target = np.array([[0.01, 0.03], [-0.01, 0.01]])
        self.assertAlmostEqual(_mean_absolute_error(np.array([0.02, 0.002]), target), 0.001)
        with self.assertRaises(ValueError):
            _mean_absolute_error(np.array([0.02]), target)
        self.assertFalse(np.isfinite(_mean_absolute_error(np.array([np.nan, 0]), target)))
        self.assertFalse(np.isfinite(_mean_absolute_error(np.array([]), np.array([]))))

    def test_smoke_schedule_spans_month_end_and_next_month(self):
        dates = pd.to_datetime([
            "2026-05-27", "2026-05-28", "2026-05-29", "2026-06-01", "2026-06-30", "2026-07-01"
        ])
        meta = pd.DataFrame({
            "as_of_date": np.repeat(dates, 2),
            "stock": ["A", "B"] * len(dates),
            "_row_id": np.arange(len(dates) * 2),
        })
        c = Config(
            cache_dir="x", output_dir="y", smoke_test_schedule=True,
            test_start="2026-05-01", test_end="2026-07-31",
        )
        selected = _select_test_rows(meta, c)
        self.assertEqual(
            sorted(selected.as_of_date.dt.strftime("%Y-%m-%d").unique().tolist()),
            ["2026-05-29", "2026-06-01", "2026-06-30", "2026-07-01"],
        )

    def test_validation_metric_is_mean_daily_spearman(self):
        meta = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
        })
        target = np.array([1, 2, 3, 30, 20, 10], dtype=float)
        score = np.array([10, 20, 30, 3, 2, 1], dtype=float)
        self.assertAlmostEqual(_mean_daily_rankic(score, target, meta), 1.0)

    def test_replay_is_70_recent_30_historical_and_year_stratified(self):
        train = pd.DataFrame({
            "as_of_date": pd.to_datetime(
                ["2021-06-01"] * 50 + ["2022-06-01"] * 50 + ["2024-06-01"] * 100
            ),
            "label_end_date": pd.to_datetime(
                ["2021-06-15"] * 50 + ["2022-06-15"] * 50 + ["2024-06-15"] * 100
            ),
            "stock": [f"S{i:03d}" for i in range(200)],
        })
        test = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2025-05-01"] * 20),
            "label_end_date": pd.to_datetime(["2025-05-20"] * 20),
            "stock": [f"T{i:03d}" for i in range(20)],
        })
        c = Config(cache_dir="x", output_dir="y", replay_max_rows=100)
        selected, stats = _select_replay_rows(
            train, test, pd.Timestamp("2025-06-30"), c, seed=7
        )
        self.assertEqual(len(selected), 100)
        self.assertEqual(stats["selected_recent_rows"], 70)
        self.assertEqual(stats["selected_historical_rows"], 30)
        self.assertEqual(stats["historical_year_counts"], {"2021": 15, "2022": 15})
        self.assertTrue((selected.label_end_date <= pd.Timestamp("2025-06-30")).all())

    def test_end_to_end_can_load_train_val_from_separate_cache(self):
        self._check_end_to_end(0.0195, True)

    def test_end_to_end_equal_mae_is_not_a_strict_improvement(self):
        self._check_end_to_end(0.02, True)

    def test_end_to_end_rejection_keeps_champion(self):
        self._check_end_to_end(0.022, False)

    def test_end_to_end_nonfinite_candidate_keeps_champion(self):
        self._check_end_to_end(np.inf, False)

    def _check_end_to_end(self, candidate_mae, accepted):
        rng = np.random.default_rng(4)
        stocks = ["S0", "S1", "S2", "S3"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            test_cache = root / "test"
            out = root / "out"
            base.mkdir(); test_cache.mkdir()

            def write_split(cache, split, dates):
                rows = []
                features = []
                labels = []
                for d in pd.to_datetime(dates):
                    for rank, stock in enumerate(stocks):
                        x = rng.normal(size=8)
                        x[0] = rank
                        y = np.full(10, rank / 10.0, dtype=np.float32)
                        rows.append({
                            "split": split,
                            "as_of_date": d.strftime("%Y-%m-%d"),
                            "stock": stock,
                            "label_end_date": (d + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                        })
                        features.append(x)
                        labels.append(y)
                np.save(cache / f"hidden_{split}.npy", np.asarray(features, np.float32))
                np.save(cache / f"labels_{split}.npy", np.asarray(labels, np.float32))
                pd.DataFrame(rows).to_csv(cache / f"meta_{split}.csv", index=False)

            write_split(base, "train", ["2022-01-03", "2023-12-01"])
            write_split(base, "val", ["2024-01-02", "2024-02-01"])
            write_split(test_cache, "test", [
                "2026-03-31", "2026-04-01", "2026-04-30", "2026-05-06"
            ])
            raw = pd.read_csv(test_cache / "meta_test.csv")[["as_of_date", "stock"]]
            raw["raw_signal"] = np.tile(np.arange(4, dtype=float), 4)
            raw_path = root / "raw.csv"
            raw.to_csv(raw_path, index=False)

            config = Config(
                cache_dir=str(test_cache),
                train_val_cache_dir=str(base),
                output_dir=str(out),
                raw_signal_file=str(raw_path),
                backend="numpy",
                smoke_test_schedule=True,
                replay_max_rows=12,
                validation_tolerance=0.0,
            )
            module = "scripts.experiments.run_continual_csi300_abc_experiment"
            # RankIC deliberately deteriorates: it must not control acceptance.
            with patch(module + "._validation_mae", side_effect=[0.02, candidate_mae]), \
                 patch(module + "._validation_rankic", side_effect=[0.5, -0.5]):
                summary = _run(config)
            self.assertEqual(summary["groups"], ["A", "B", "C"])
            pred = pd.read_csv(out / "predictions_C.csv")
            june_end = pred[pred.as_of_date == "2026-04-30"]
            july_first = pred[pred.as_of_date == "2026-05-06"]
            self.assertEqual(june_end.model_version.unique().tolist(), [0])
            self.assertEqual(july_first.model_version.unique().tolist(), [int(accepted)])
            log = pd.read_csv(out / "update_log.csv")
            decision_log = log[log.candidate_validation_mae.notna()].iloc[0]
            self.assertEqual(decision_log.selected_replay_rows, 12)
            self.assertEqual(bool(decision_log.accepted), accepted)
            self.assertEqual(decision_log.validation_metric, "path_mae")
            self.assertEqual(summary["accepted_updates"], int(accepted))
            self.assertEqual(summary["final_champion_validation_mae"], candidate_mae if accepted else 0.02)
            self.assertEqual(summary["final_champion_validation_rankic"], -0.5 if accepted else 0.5)
            with np.load(out / "checkpoints" / f"C_version_{int(accepted):03d}.npz") as checkpoint:
                metadata = json.loads(str(checkpoint["metadata"]))
            self.assertEqual(metadata["validation_metric"], "path_mae")
            self.assertEqual(metadata["validation_tolerance"], 0.0)


if __name__ == "__main__":
    unittest.main()
