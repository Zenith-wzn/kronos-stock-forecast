from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "results" / "dashboards"


class DashboardAppendixTests(unittest.TestCase):
    def test_appendix_pages_and_data_are_present(self):
        html = (DASH / "all_experiments_dashboard.html").read_text(encoding="utf-8")
        data = json.loads((DASH / "all_experiments_dashboard_data.json").read_text(encoding="utf-8"))
        self.assertIn("sorting_reward", data); self.assertIn("annual_portfolio", data); self.assertIn("sorting_reward_audit", data)
        self.assertIn("排序奖励学习", html); self.assertIn("一年期组合收益", html); self.assertIn("排序奖励审计", html)
        self.assertNotIn("NaN", html); self.assertNotIn("Infinity", html)
        match = re.search(r"window\.__sortingRewardData=(\{.*?\})</script>", html, re.S)
        self.assertIsNotNone(match)
        embedded = json.loads(match.group(1))
        self.assertEqual(embedded["sorting_reward"], data["sorting_reward"])
        marker = "var D = "
        start = html.index(marker) + len(marker)
        main, _ = json.JSONDecoder().raw_decode(html[start:])
        self.assertEqual(main, data)

    def test_update_audit_is_present(self):
        audit = json.loads((DASH / "sorting_reward_dashboard_update_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["old_top_level_keys_preserved"])
        self.assertTrue(audit["html_embedded_json_equal"])
        self.assertTrue(audit.get("html_main_data_equal"))

    def test_formal_market_matrix_and_historical_entries_are_preserved(self):
        data = json.loads((DASH / "all_experiments_dashboard_data.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["experiments"]), 42)
        for dataset in ("CSI800", "CSI300", "S&P500"):
            sorting_market = data["sorting_reward"]["markets"][dataset]
            annual_market = data["annual_portfolio"]["markets"][dataset]
            self.assertEqual(sorting_market["status"], "可验证")
            self.assertEqual(len(sorting_market["rank_rows"]), 25)
            self.assertEqual(len(sorting_market["multi_seed_summary_rows"]), 5)
            self.assertEqual({row["seed_count"] for row in sorting_market["multi_seed_summary_rows"]}, {5})
            self.assertEqual(len(annual_market["rows"]), 30)
            baseline_rows = [row for row in annual_market["rows"] if row["model"] == "MARKET_EW"]
            self.assertEqual(len(baseline_rows), 5)
            self.assertEqual({row["seed"] for row in baseline_rows}, {100, 101, 102, 103, 104})
            self.assertGreater(len(annual_market["sensitivity_rows"]), 0)

    def test_sp500_rmb_profit_is_explicitly_unavailable(self):
        data = json.loads((DASH / "all_experiments_dashboard_data.json").read_text(encoding="utf-8"))
        market = data["annual_portfolio"]["markets"]["S&P500"]
        self.assertEqual(market["currency"], "USD")
        self.assertEqual(market["fx_conversion_status"], "无法验证")
        for row in market["rows"]:
            self.assertIsNone(row["absolute_profit_rmb"])
            self.assertEqual(row["rmb_profit_status"], "无法验证")
            self.assertTrue(row["rmb_profit_unavailable_reason"])

    def test_new_summary_and_currency_labels_are_rendered(self):
        html = (DASH / "all_experiments_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("五 seed 稳定性（均值 ± seed 标准差）", html)
        self.assertIn("MARKET_EW", html)
        self.assertIn("人民币绝对盈利", html)
        self.assertIn("未验证 USD/CNY 汇率", html)

    def test_screenshot_style_parameters_and_metric_help_are_rendered(self):
        html = (DASH / "all_experiments_dashboard.html").read_text(encoding="utf-8")
        for label in (
            "训练截止", "验证期", "测试期", "Lookback", "预测长度",
            "主标签", "诊断标签", "Backbone", "Pairwise scale",
            "Huber 权重", "组合成本", "初始资金",
        ):
            self.assertIn(label, html)
        self.assertIn("一、横截面排序指标", html)
        self.assertIn("二、组合收益指标", html)
        self.assertIn("appendix-metric-help", html)
        self.assertIn("data-sr-metric", html)
        self.assertIn("指标含义", html)
        self.assertIn("计算方式", html)
        self.assertIn("投资意义", html)
        self.assertIn("优劣方向", html)
        self.assertIn("来源字段", html)
        self.assertIn("modal.classList.add('open')", html)

    def test_model_comparison_highlighting_and_summaries_are_rendered(self):
        html = (DASH / "all_experiments_dashboard.html").read_text(encoding="utf-8")
        for model in ("S0", "S1", "S2", "S3", "S4", "MARKET_EW"):
            self.assertIn(model, html)
        for label in (
            "排序能力最优", "Top-K 超额最优", "分层分离最优",
            "年化净收益最优", "回撤控制最优", "Sharpe 最优", "换手最低",
        ):
            self.assertIn(label, html)
        self.assertIn("appendix-table td.best", html)
        self.assertIn("appendix-table td.worst", html)

    def test_metric_definitions_include_investment_meaning_direction_and_source(self):
        data = json.loads((DASH / "all_experiments_dashboard_data.json").read_text(encoding="utf-8"))
        definitions = data["sorting_reward"]["metric_definitions"]
        for key in (
            "rankic_mean", "ic_mean", "q5_q1", "top_k_excess_return",
            "annualized_net_return", "maximum_drawdown", "sharpe", "turnover",
        ):
            definition = definitions[key]
            self.assertTrue(definition["formula"])
            self.assertTrue(definition["investment_meaning"])
            self.assertTrue(definition["direction"])
            self.assertTrue(definition["source_field"])


if __name__ == "__main__":
    unittest.main()
