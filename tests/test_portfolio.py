from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only


class PortfolioTests(unittest.TestCase):
    def test_top_k_and_initial_cost(self):
        rows = []
        for d in pd.date_range("2025-07-01", periods=3, freq="D"):
            for stock, score, ret in (("a", 3., .02), ("b", 2., .01), ("c", 1., -.02)):
                rows.append({"as_of_date": d, "stock": stock, "prediction_score": score, "actual_endpoint_return": ret})
        result = backtest_long_only(pd.DataFrame(rows), config=PortfolioConfig(top_k=2, cost_rate=.001))
        self.assertEqual(result["daily"].iloc[0]["n_holdings"], 2)
        self.assertAlmostEqual(float(result["daily"].iloc[0]["turnover"]), 1.0)
        self.assertAlmostEqual(float(result["daily"].iloc[0]["transaction_cost"]), .001)

    def test_wealth_and_annualization(self):
        rows = []
        for d in pd.date_range("2025-07-01", periods=10, freq="D"):
            rows.extend(({"as_of_date": d, "stock": "a", "prediction_score": 1., "actual_endpoint_return": .01}, {"as_of_date": d, "stock": "b", "prediction_score": 0., "actual_endpoint_return": .01}))
        result = backtest_long_only(pd.DataFrame(rows), config=PortfolioConfig(top_k=1, cost_rate=0.0))
        self.assertAlmostEqual(result["summary"]["final_wealth"], 1_000_000 * 1.01 ** 10, places=5)
        self.assertAlmostEqual(result["summary"]["annualized_net_return"], (1.01 ** 10) ** (252 / 10) - 1, places=10)

    def test_missing_prices_and_no_future_fill(self):
        rows = [
            {"as_of_date": "2025-07-01", "stock": "a", "prediction_score": 1., "actual_endpoint_return": .01, "entry_price": np.nan},
            {"as_of_date": "2025-07-01", "stock": "b", "prediction_score": 0., "actual_endpoint_return": .01, "entry_price": 10.},
        ]
        result = backtest_long_only(pd.DataFrame(rows), config=PortfolioConfig(top_k=1, cost_rate=0.0))
        # The invalid-price stock is excluded before ranking; no future fill or
        # hidden substitution is allowed.
        self.assertEqual(result["holdings"].iloc[0]["stock"], "b")
        self.assertAlmostEqual(float(result["summary"]["excluded_rows"]), 1.0)

    def test_missing_holding_return_is_zero_not_renormalized(self):
        rows = [
            {"as_of_date": "2025-07-01", "stock": "a", "prediction_score": 2.0, "next_day_return": 0.02},
            {"as_of_date": "2025-07-01", "stock": "b", "prediction_score": 1.0, "next_day_return": np.nan},
        ]
        result = backtest_long_only(
            pd.DataFrame(rows), return_col="next_day_return",
            config=PortfolioConfig(top_k=2, cost_rate=0.0),
        )
        self.assertAlmostEqual(float(result["daily"].iloc[0]["gross_return"]), 0.01)
        self.assertEqual(int(result["daily"].iloc[0]["missing_return_holdings"]), 1)
        missing = result["holdings"].set_index("stock").loc["b"]
        self.assertEqual(float(missing["realized_return"]), 0.0)
        self.assertTrue(bool(missing["return_was_missing"]))

    def test_transaction_cost_amount_uses_dynamic_wealth(self):
        rows = [
            {"as_of_date": "2025-07-01", "stock": "a", "prediction_score": 2.0, "next_day_return": 0.10},
            {"as_of_date": "2025-07-01", "stock": "b", "prediction_score": 1.0, "next_day_return": 0.00},
            {"as_of_date": "2025-07-02", "stock": "a", "prediction_score": 1.0, "next_day_return": 0.00},
            {"as_of_date": "2025-07-02", "stock": "b", "prediction_score": 2.0, "next_day_return": 0.00},
        ]
        result = backtest_long_only(
            pd.DataFrame(rows), return_col="next_day_return",
            config=PortfolioConfig(top_k=1, drop_n=1, min_holding_days=1, cost_rate=0.01),
        )
        daily = result["daily"].reset_index(drop=True)
        self.assertAlmostEqual(float(daily.loc[0, "transaction_cost_amount"]), 10_000.0)
        wealth_after_day_1 = 1_000_000 * (1 + 0.10 - 0.01)
        self.assertAlmostEqual(float(daily.loc[1, "transaction_cost_amount"]), wealth_after_day_1 * 0.01)
        self.assertAlmostEqual(
            float(result["summary"]["cumulative_transaction_cost_amount"]),
            10_000.0 + wealth_after_day_1 * 0.01,
        )

    def test_drop_n_limits_voluntary_exits(self):
        rows = []
        dates = pd.date_range("2025-07-01", periods=2, freq="D")
        for stock, score, ret in (("a", 3., .01), ("b", 2., .01), ("c", 1., .01)):
            rows.append({"as_of_date": dates[0], "stock": stock, "prediction_score": score, "actual_endpoint_return": ret})
        for stock, score, ret in (("c", 3., .01), ("b", 2., .01), ("a", 1., .01)):
            rows.append({"as_of_date": dates[1], "stock": stock, "prediction_score": score, "actual_endpoint_return": ret})
        result = backtest_long_only(pd.DataFrame(rows), config=PortfolioConfig(top_k=1, drop_n=0, min_holding_days=1, cost_rate=0.0))
        # Drop-N=0 permits no voluntary replacement; the incumbent remains.
        self.assertEqual(set(result["holdings"].query("as_of_date == @dates[1].date()")["stock"]), {"a"})

    def test_min_holding_days_locks_position(self):
        rows = []
        dates = pd.date_range("2025-07-01", periods=2, freq="D")
        for stock, score in (("a", 2.), ("b", 1.)):
            rows.append({"as_of_date": dates[0], "stock": stock, "prediction_score": score, "actual_endpoint_return": .01})
        for stock, score in (("b", 2.), ("a", 1.)):
            rows.append({"as_of_date": dates[1], "stock": stock, "prediction_score": score, "actual_endpoint_return": .01})
        result = backtest_long_only(pd.DataFrame(rows), config=PortfolioConfig(top_k=1, drop_n=0, min_holding_days=5, cost_rate=0.0))
        self.assertEqual(set(result["holdings"].query("as_of_date == @dates[1].date()")["stock"]), {"a"})

    def test_max_drawdown_includes_initial_capital(self):
        rows = [
            {"as_of_date": "2025-07-01", "stock": "a", "prediction_score": 1.0, "next_day_return": -0.10},
            {"as_of_date": "2025-07-02", "stock": "a", "prediction_score": 1.0, "next_day_return": 0.05},
        ]
        result = backtest_long_only(
            pd.DataFrame(rows), return_col="next_day_return",
            config=PortfolioConfig(top_k=1, cost_rate=0.0),
        )
        self.assertAlmostEqual(float(result["summary"]["maximum_drawdown"]), -0.10)

    def test_optimized_cost_sensitivity_matches_direct_backtests(self):
        rows = []
        for day, scores in enumerate(((3., 2., 1.), (1., 3., 2.), (2., 1., 3.))):
            date = pd.Timestamp("2025-07-01") + pd.Timedelta(days=day)
            for stock, score, ret in zip(("a", "b", "c"), scores, (.02, -.01, .005)):
                rows.append({"as_of_date": date, "stock": stock, "prediction_score": score,
                             "next_day_return": ret})
        frame = pd.DataFrame(rows)
        base = PortfolioConfig(top_k=2, drop_n=1, min_holding_days=1, cost_rate=.0015)
        from continual_abcs.portfolio import run_cost_sensitivity
        optimized = run_cost_sensitivity(
            frame, return_col="next_day_return", base_config=base,
            costs_bps=(5, 30), holding_days=(1,), top_ks=(2,),
        ).sort_values("sensitivity_cost_bps").reset_index(drop=True)
        for i, bps in enumerate((5, 30)):
            direct = backtest_long_only(
                frame, return_col="next_day_return",
                config=PortfolioConfig(2, 1, 1, bps / 10000.0),
            )["summary"]
            for key in ("final_wealth", "annualized_net_return", "maximum_drawdown",
                        "sharpe", "cumulative_transaction_cost_amount"):
                self.assertAlmostEqual(float(optimized.loc[i, key]), float(direct[key]), places=10)

    def test_output_contract_and_sensitivity_matrix(self):
        rows = []
        for d in pd.date_range("2025-07-01", periods=3, freq="D"):
            rows.extend([
                {"as_of_date": d, "stock": "a", "prediction_score": 2., "actual_endpoint_return": .01},
                {"as_of_date": d, "stock": "b", "prediction_score": 1., "actual_endpoint_return": .00},
            ])
        cfg = PortfolioConfig(top_k=1, cost_rate=.0015)
        result = backtest_long_only(pd.DataFrame(rows), config=cfg)
        self.assertTrue({"portfolio_value", "drawdown"}.issubset(result["daily"].columns))
        self.assertTrue({"price", "transaction_cost", "reason"}.issubset(result["trades"].columns))
        self.assertTrue({"entry_price", "exit_price", "realized_return"}.issubset(result["holdings"].columns))
        from continual_abcs.portfolio import run_cost_sensitivity
        sens = run_cost_sensitivity(pd.DataFrame(rows), base_config=cfg, costs_bps=(5, 10, 15, 30), holding_days=(5, 10, 20), top_ks=(1, 2))
        self.assertEqual(len(sens), 24)



if __name__ == "__main__":
    unittest.main()
