from __future__ import annotations

"""Deterministic long-only cross-sectional portfolio backtesting utilities.

The backtester deliberately separates *selection* (scores) from *realisation*
(forward returns).  Forward returns are only consumed after holdings have been
selected, which makes the function suitable for out-of-sample evaluation.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int
    drop_n: int = 0
    min_holding_days: int = 1
    cost_rate: float = 0.0015
    initial_capital: float = 1_000_000.0
    annualization_days: float = 252.0


MARKET_PORTFOLIO_CONFIGS: dict[str, PortfolioConfig] = {
    "CSI800": PortfolioConfig(top_k=200, drop_n=10, min_holding_days=5),
    "CSI300": PortfolioConfig(top_k=50, drop_n=5, min_holding_days=5),
    "S&P500": PortfolioConfig(top_k=50, drop_n=5, min_holding_days=5),
}


def _max_drawdown(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float((v / v.cummax() - 1.0).min())


def _sharpe(returns: pd.Series, annualization_days: float) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna()
    if len(x) < 2 or x.std(ddof=1) <= 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(annualization_days))


def _sortino(returns: pd.Series, annualization_days: float) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna()
    if len(x) < 1:
        return float("nan")
    downside = x[x < 0]
    if downside.empty:
        return float("nan")
    denom = float(np.sqrt(np.mean(np.square(downside))))
    return float(x.mean() / denom * np.sqrt(annualization_days)) if denom > 0 else float("nan")


def _loss_streak(returns: pd.Series) -> int:
    best = current = 0
    for value in pd.to_numeric(returns, errors="coerce").dropna():
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def select_top_k(group: pd.DataFrame, score_col: str, top_k: int) -> list[str]:
    """Return a deterministic score-ranked list without using realised returns."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    missing = {"stock", score_col} - set(group.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    g = group[["stock", score_col]].copy()
    g["stock"] = g["stock"].astype(str)
    g[score_col] = pd.to_numeric(g[score_col], errors="coerce")
    g = g.dropna(subset=[score_col]).drop_duplicates("stock", keep="last")
    return g.sort_values([score_col, "stock"], ascending=[False, True], kind="mergesort")["stock"].head(top_k).tolist()


def _target_holdings(
    ranked: list[str],
    previous: set[str],
    *,
    top_k: int,
    drop_n: int,
    min_holding_days: int,
    age: Mapping[str, int],
) -> set[str]:
    """Build the next portfolio without using realised returns.

    ``drop_n`` is an exit budget: at most ``drop_n`` currently held,
    sufficiently old names may be voluntarily replaced on a rebalance date.
    Positions younger than ``min_holding_days`` are locked when they remain in
    the observable universe.  Names that disappear from the day's ranked
    universe are mandatory exits because they cannot be traded or valued.
    """
    if top_k < 1 or drop_n < 0 or min_holding_days < 1:
        raise ValueError("invalid holding policy")
    ranked = list(dict.fromkeys(str(s) for s in ranked))
    previous = set(str(s) for s in previous)
    if not previous:
        return set(ranked[:top_k])

    desired = ranked[:top_k]
    desired_set = set(desired)
    rank_pos = {stock: index for index, stock in enumerate(ranked)}

    # A held name outside today's observable universe cannot be retained.
    missing = previous - set(ranked)
    # Young names still present in the universe are locked.
    locked = {
        stock for stock in previous
        if stock in rank_pos and stock not in desired_set
        and age.get(stock, 0) < min_holding_days
    }

    voluntary = previous - missing - locked - desired_set
    # Replace the lowest-ranked eligible incumbents first, subject to Drop-N.
    exits = set(sorted(voluntary, key=lambda stock: (-rank_pos[stock], stock))[:drop_n])
    retained = (previous - missing - exits)

    # Retained positions take precedence.  Fill remaining capacity in score
    # order; this guarantees Drop-N=0 keeps the incumbent portfolio unchanged.
    target = set(retained)
    for stock in desired:
        if len(target) >= top_k:
            break
        target.add(stock)
    return target


def _month_returns(daily: pd.DataFrame) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    x = daily.dropna(subset=["net_return"]).copy()
    if x.empty:
        return pd.Series(dtype=float)
    x["month"] = pd.to_datetime(x["as_of_date"]).dt.to_period("M").astype(str)
    return x.groupby("month")["net_return"].apply(lambda s: float((1.0 + s).prod() - 1.0))


def _rescore_summary_for_cost(
    daily: pd.DataFrame,
    base_summary: Mapping[str, object],
    *,
    config: PortfolioConfig,
    cost_rate: float,
) -> dict[str, object]:
    """Recompute cost-dependent metrics without rerunning holding decisions.

    Scores, Top-K, Drop-N and holding constraints do not depend on transaction
    cost.  Sensitivity analysis therefore evaluates each holding path once and
    deterministically rescales its turnover for every requested cost rate.
    """
    if daily.empty:
        out = dict(base_summary)
        out.update({"metric_status": "无法验证", "unavailable_reason": "no valid portfolio dates",
                    "cost_rate": float(cost_rate)})
        return out
    x = daily.copy()
    x["as_of_date"] = pd.to_datetime(x["as_of_date"], errors="coerce")
    gross = pd.to_numeric(x["gross_return"], errors="coerce")
    turnover = pd.to_numeric(x["turnover"], errors="coerce").fillna(0.0)
    x["transaction_cost"] = turnover * float(cost_rate)
    x["net_return"] = gross - x["transaction_cost"]
    if x["net_return"].notna().sum() == 0:
        out = dict(base_summary)
        out.update({"metric_status": "无法验证", "unavailable_reason": "no valid portfolio dates",
                    "cost_rate": float(cost_rate)})
        return out

    wealth = float(config.initial_capital)
    values: list[float] = []
    cost_amounts: list[float] = []
    for row in x.itertuples(index=False):
        net_return = float(row.net_return) if np.isfinite(row.net_return) else 0.0
        cost_return = float(row.transaction_cost) if np.isfinite(row.transaction_cost) else 0.0
        cost_amounts.append(wealth * cost_return)
        wealth *= 1.0 + net_return
        values.append(wealth)
    x["transaction_cost_amount"] = cost_amounts
    x["portfolio_value"] = values
    peaks = np.maximum.accumulate(np.r_[float(config.initial_capital), np.asarray(values, dtype=float)])[1:]
    x["drawdown"] = np.asarray(values, dtype=float) / peaks - 1.0

    net = pd.to_numeric(x["net_return"], errors="coerce").dropna()
    gross_valid = gross.dropna()
    final = float(values[-1])
    n = int(len(net))
    cumulative = final / config.initial_capital - 1.0
    annual = (final / config.initial_capital) ** (config.annualization_days / n) - 1.0 if final > 0 and n else np.nan
    gross_cumulative = float((1.0 + gross_valid).prod() - 1.0) if not gross_valid.empty else np.nan
    cost_sum = float(x["transaction_cost"].fillna(0.0).sum())
    months = _month_returns(x)
    out = dict(base_summary)
    out.update({
        "metric_status": "可验证", "unavailable_reason": "", "n_dates": n,
        "top_k": config.top_k, "drop_n": config.drop_n,
        "min_holding_days": config.min_holding_days, "cost_rate": float(cost_rate),
        "initial_capital": config.initial_capital, "final_wealth": final,
        "absolute_profit": final - config.initial_capital,
        "gross_cumulative_return": gross_cumulative,
        "net_cumulative_return": cumulative, "annualized_net_return": annual,
        "maximum_drawdown": float(x["drawdown"].min()),
        "sharpe": _sharpe(net, config.annualization_days),
        "sortino": _sortino(net, config.annualization_days),
        "win_day_ratio": float((net > 0).mean()),
        "turnover_mean": float(turnover.mean()), "turnover_total": float(turnover.sum()),
        "annualized_turnover": float(turnover.mean() * config.annualization_days),
        "cumulative_transaction_cost": cost_sum,
        "cumulative_transaction_cost_amount": float(np.sum(cost_amounts)),
        "cost_over_gross_profit": float(cost_sum / gross_cumulative)
            if np.isfinite(gross_cumulative) and gross_cumulative > 0 else np.nan,
        "worst_day": float(net.min()),
        "worst_month": float(months.min()) if not months.empty else np.nan,
        "longest_loss_streak": _loss_streak(net),
    })
    return out


def backtest_long_only(
    predictions: pd.DataFrame,
    *,
    score_col: str = "prediction_score",
    return_col: str = "actual_endpoint_return",
    config: PortfolioConfig,
    dataset: str = "",
    model: str = "",
    seed: int | None = None,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Evaluate a daily Top-K equal-weight portfolio.

    The default turnover is half the L1 change in portfolio weights.  Thus a
    full initial investment has turnover 1.0 and a complete replacement has
    turnover close to 1.0; transaction cost is ``turnover * cost_rate``.
    """
    if config.top_k < 1 or config.drop_n < 0 or config.min_holding_days < 1:
        raise ValueError("invalid portfolio configuration")
    required = {"as_of_date", "stock", score_col, return_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    df = predictions.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize()
    df["stock"] = df["stock"].astype(str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    # Portfolio PnL must use the next-trading-day realized return when it is
    # available.  The 10-day endpoint label is a ranking target, not a daily
    # cash-flow return; callers can still explicitly pass another audited
    # return column for synthetic tests or non-daily portfolios.
    df = df.dropna(subset=["as_of_date"]).sort_values(["as_of_date", "stock"], kind="mergesort")

    previous: set[str] = set()
    age: dict[str, int] = {}
    entry_dates: dict[str, pd.Timestamp] = {}
    entry_indices: dict[str, int] = {}
    completed_holding_days: list[int] = []
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    excluded_rows = 0

    for eval_index, (date, group) in enumerate(df.groupby("as_of_date", sort=True)):
        score_valid = group.dropna(subset=[score_col]).copy()
        # Missing/non-positive entry prices make a row non-tradable.  Exit
        # prices are retained for audit but are not required to select a
        # holding because endpoint returns may be supplied directly by the
        # cache.  Never forward-fill prices or infer tradability from a
        # future return.
        if "entry_price" in score_valid.columns:
            entry_values = pd.to_numeric(score_valid["entry_price"], errors="coerce")
            tradable = entry_values.notna() & (entry_values > 0)
            excluded_rows += int((~tradable).sum())
            score_valid = score_valid.loc[tradable].copy()
        excluded_rows += int(len(group) - len(group.dropna(subset=[score_col])))
        # Keep the full observable ranking so Drop-N and minimum-holding
        # constraints can decide which incumbents may be replaced.
        ranked = select_top_k(score_valid, score_col, max(config.top_k, len(score_valid)))
        target = _target_holdings(ranked, previous, top_k=config.top_k, drop_n=config.drop_n,
                                  min_holding_days=config.min_holding_days, age=age)
        if not target:
            daily_rows.append({"dataset": dataset, "model": model, "seed": seed, "as_of_date": date,
                               "gross_return": np.nan, "transaction_cost": 0.0, "net_return": np.nan,
                               "turnover": 0.0, "n_holdings": 0, "n_buys": 0, "n_sells": len(previous),
                               "cash_weight": 1.0, "missing_return_holdings": 0})
            for stock in previous:
                if stock in entry_indices:
                    completed_holding_days.append(max(0, eval_index - entry_indices[stock]))
            previous, age, entry_dates, entry_indices = set(), {}, {}, {}
            continue

        weights = {s: 1.0 / len(target) for s in target}
        old_weights = {s: 1.0 / len(previous) for s in previous} if previous else {}
        union = set(weights) | set(old_weights)
        old_cash = max(0.0, 1.0 - sum(old_weights.values()))
        new_cash = max(0.0, 1.0 - sum(weights.values()))
        turnover = 0.5 * (sum(abs(weights.get(s, 0.0) - old_weights.get(s, 0.0)) for s in union) + abs(new_cash - old_cash))
        transaction_cost = turnover * config.cost_rate
        by_stock = score_valid.drop_duplicates("stock", keep="last").set_index("stock")
        realised_by_stock = {
            s: (
                float(by_stock.loc[s, return_col])
                if s in by_stock.index and np.isfinite(by_stock.loc[s, return_col])
                else 0.0
            )
            for s in target
        }
        missing_return_count = sum(
            1 for s in target
            if s not in by_stock.index or not np.isfinite(by_stock.loc[s, return_col])
        )
        # A missing next-day return is treated as a cash-like 0% return for
        # that holding.  We never silently re-normalize the remaining names,
        # because doing so would introduce an ex-post weight change.
        excluded_rows += missing_return_count
        gross = float(sum(weights[s] * realised_by_stock[s] for s in target))
        net = gross - transaction_cost if np.isfinite(gross) else np.nan
        buys = sorted(set(target) - previous)
        sells = sorted(previous - set(target))
        for stock in sells:
            if stock in entry_indices:
                completed_holding_days.append(max(0, eval_index - entry_indices[stock]))
                entry_indices.pop(stock, None)
                entry_dates.pop(stock, None)
        for stock in buys:
            entry_dates[stock] = date
            entry_indices[stock] = eval_index
        for stock in sorted(target):
            row = by_stock.loc[stock] if stock in by_stock.index else None
            rank = ranked.index(stock) + 1 if stock in ranked else np.nan
            rec = {"dataset": dataset, "model": model, "seed": seed, "as_of_date": date,
                   "stock": stock, "weight": weights[stock], "score": float(row[score_col]) if row is not None else np.nan,
                   "rank": rank, "realized_return": (float(row[return_col]) if row is not None and np.isfinite(row[return_col]) else 0.0),
                   "return_was_missing": bool(row is None or not np.isfinite(row[return_col]))}
            for col in ("entry_price", "exit_price"):
                if col in group.columns:
                    rec[col] = float(row[col]) if row is not None and pd.notna(row[col]) else np.nan
            holding_rows.append(rec)
        for stock in buys + sells:
            delta = abs(weights.get(stock, 0.0) - old_weights.get(stock, 0.0))
            trade_rows.append({"dataset": dataset, "model": model, "seed": seed, "trade_date": date,
                               "stock": stock, "side": "BUY" if stock in buys else "SELL",
                               "old_weight": old_weights.get(stock, 0.0), "new_weight": weights.get(stock, 0.0),
                               "trade_weight": delta,
                               "price": (float(by_stock.loc[stock].get("entry_price" if stock in buys else "exit_price"))
                               if stock in by_stock.index and ("entry_price" if stock in buys else "exit_price") in by_stock.columns
                               and pd.notna(by_stock.loc[stock].get("entry_price" if stock in buys else "exit_price")) else np.nan),
                               "transaction_cost": delta * config.cost_rate, "reason": "rebalance" if previous else "initial_build"})
        daily_rows.append({"dataset": dataset, "model": model, "seed": seed, "as_of_date": date,
                           "gross_return": gross, "transaction_cost": transaction_cost, "net_return": net,
                           "turnover": turnover, "n_holdings": len(target), "n_buys": len(buys),
                           "n_sells": len(sells), "cash_weight": max(0.0, 1.0 - sum(weights.values())),
                           "missing_return_holdings": missing_return_count})
        age = {s: age.get(s, 0) + 1 for s in target}
        previous = set(target)

    daily_columns = ["dataset", "model", "seed", "as_of_date", "gross_return", "transaction_cost",
                     "transaction_cost_amount", "net_return", "turnover", "n_holdings", "n_buys", "n_sells", "cash_weight",
                     "missing_return_holdings", "portfolio_value", "drawdown"]
    holding_columns = ["dataset", "model", "seed", "as_of_date", "stock", "weight", "score", "rank",
                       "entry_price", "exit_price", "realized_return", "return_was_missing"]
    trade_columns = ["dataset", "model", "seed", "trade_date", "stock", "side", "old_weight",
                     "new_weight", "trade_weight", "price", "transaction_cost", "reason"]
    daily = pd.DataFrame(daily_rows, columns=daily_columns)
    holdings = pd.DataFrame(holding_rows, columns=holding_columns)
    trades = pd.DataFrame(trade_rows, columns=trade_columns)
    if daily.empty or daily["net_return"].notna().sum() == 0:
        summary = {"dataset": dataset, "model": model, "seed": seed, "metric_status": "无法验证",
                   "unavailable_reason": "no valid portfolio dates", "excluded_rows": excluded_rows,
                   "initial_capital": config.initial_capital, "top_k": config.top_k,
                   "drop_n": config.drop_n, "min_holding_days": config.min_holding_days,
                   "cost_rate": config.cost_rate}
        return {"daily": daily, "holdings": holdings, "trades": trades, "summary": summary}

    daily["as_of_date"] = pd.to_datetime(daily["as_of_date"])
    daily["net_return"] = pd.to_numeric(daily["net_return"], errors="coerce")
    daily["transaction_cost"] = pd.to_numeric(daily["transaction_cost"], errors="coerce").fillna(0.0)
    wealth = float(config.initial_capital)
    values: list[float] = []
    cost_amounts: list[float] = []
    for row in daily.itertuples(index=False):
        net_return = float(row.net_return) if np.isfinite(row.net_return) else 0.0
        cost_rate_return = float(row.transaction_cost) if np.isfinite(row.transaction_cost) else 0.0
        cost_amounts.append(wealth * cost_rate_return)
        wealth *= 1.0 + net_return
        values.append(wealth)
    daily["transaction_cost_amount"] = cost_amounts
    daily["portfolio_value"] = values
    peaks = np.maximum.accumulate(np.r_[float(config.initial_capital), np.asarray(values, dtype=float)])[1:]
    daily["drawdown"] = np.asarray(values, dtype=float) / peaks - 1.0
    net = daily["net_return"].dropna()
    gross = daily["gross_return"].dropna()
    final = float(daily["portfolio_value"].iloc[-1])
    n = int(len(net))
    cumulative = final / config.initial_capital - 1.0
    annual = (final / config.initial_capital) ** (config.annualization_days / n) - 1.0 if final > 0 else np.nan
    months = _month_returns(daily)
    gross_cumulative = float((1.0 + gross).prod() - 1.0) if not gross.empty else np.nan
    cost_sum = float(pd.to_numeric(daily["transaction_cost"], errors="coerce").fillna(0.0).sum())
    last_eval_index = max(0, len(daily) - 1)
    open_holding_days = [max(0, last_eval_index - start_index) for start_index in entry_indices.values()]
    holding_durations = completed_holding_days + open_holding_days
    summary = {"dataset": dataset, "model": model, "seed": seed, "metric_status": "可验证", "unavailable_reason": "",
               "n_dates": n, "top_k": config.top_k, "drop_n": config.drop_n, "min_holding_days": config.min_holding_days,
               "cost_rate": config.cost_rate, "initial_capital": config.initial_capital, "final_wealth": final,
               "absolute_profit": final - config.initial_capital, "gross_cumulative_return": gross_cumulative,
               "net_cumulative_return": cumulative, "annualized_net_return": annual,
               "maximum_drawdown": float(daily["drawdown"].min()), "sharpe": _sharpe(net, config.annualization_days),
               "sortino": _sortino(net, config.annualization_days), "win_day_ratio": float((net > 0).mean()),
               "turnover_mean": float(daily["turnover"].mean()), "turnover_total": float(daily["turnover"].sum()),
               "annualized_turnover": float(daily["turnover"].mean() * config.annualization_days),
               "cumulative_transaction_cost": cost_sum,
               "cumulative_transaction_cost_amount": float(daily["transaction_cost_amount"].sum()),
               "cost_over_gross_profit": float(cost_sum / gross_cumulative) if np.isfinite(gross_cumulative) and gross_cumulative > 0 else np.nan,
               "trade_count": int(len(trades)),
               "average_holding_days": float(np.mean(holding_durations)) if holding_durations else np.nan,
               "worst_day": float(net.min()), "worst_month": float(months.min()) if not months.empty else np.nan,
               "longest_loss_streak": _loss_streak(net), "excluded_rows": excluded_rows}
    return {"daily": daily, "holdings": holdings, "trades": trades, "summary": summary}


def run_cost_sensitivity(
    predictions: pd.DataFrame, *, score_col: str = "prediction_score", return_col: str = "actual_endpoint_return",
    base_config: PortfolioConfig, costs_bps: Iterable[float] = (5, 10, 15, 30),
    holding_days: Iterable[int] = (5, 10, 20), top_ks: Iterable[int] | None = None,
    dataset: str = "", model: str = "", seed: int | None = None,
) -> pd.DataFrame:
    """Return the full fixed matrix while evaluating each holding path once."""
    ks = tuple(top_ks) if top_ks is not None else (base_config.top_k,)
    costs = tuple(float(x) for x in costs_bps)
    rows = []
    for top_k in ks:
        for hold in holding_days:
            path_config = PortfolioConfig(
                int(top_k), base_config.drop_n, int(hold), 0.0,
                base_config.initial_capital, base_config.annualization_days,
            )
            result = backtest_long_only(
                predictions, score_col=score_col, return_col=return_col,
                config=path_config, dataset=dataset, model=model, seed=seed,
            )
            for bps in costs:
                summary_config = PortfolioConfig(
                    int(top_k), base_config.drop_n, int(hold), float(bps) / 10000.0,
                    base_config.initial_capital, base_config.annualization_days,
                )
                row = _rescore_summary_for_cost(
                    result["daily"], result["summary"], config=summary_config,
                    cost_rate=float(bps) / 10000.0,
                )
                row.update({"sensitivity_top_k": int(top_k), "sensitivity_cost_bps": float(bps),
                            "sensitivity_holding_days": int(hold)})
                rows.append(row)
    return pd.DataFrame(rows)

