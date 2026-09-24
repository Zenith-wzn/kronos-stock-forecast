import unittest
import pandas as pd

from continual_abcs.run_abcs import update_schedule, simulate_test_then_train


class RunnerTests(unittest.TestCase):
    def test_month_end_prediction_uses_old_version_and_next_date_uses_new(self):
        dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
        matured = {
            pd.Timestamp("2024-01-31"): pd.DataFrame({"label_end_date": ["2024-01-30"]}),
            pd.Timestamp("2024-02-01"): pd.DataFrame({"label_end_date": ["2024-02-01"]}),
        }
        schedule = update_schedule(dates)
        out = simulate_test_then_train(dates, schedule, matured)
        self.assertEqual(out.loc[out.as_of_date == "2024-01-31", "model_version"].item(), 0)
        self.assertEqual(out.loc[out.as_of_date == "2024-02-01", "model_version"].item(), 1)
        self.assertTrue(bool(out.loc[out.as_of_date == "2024-01-31", "updated_after_prediction"].item()))

    def test_group_a_and_b_never_update(self):
        dates = pd.to_datetime(["2024-01-31", "2024-02-01"])
        schedule = update_schedule(dates)
        for group in ("A", "B"):
            out = simulate_test_then_train(dates, schedule, {}, group=group)
            self.assertEqual(out.model_version.tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
