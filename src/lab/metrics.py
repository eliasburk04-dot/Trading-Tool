from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_peak = equity_curve.cummax()
    return (equity_curve / running_peak) - 1.0


def _daily_returns(equity_curve: pd.Series) -> pd.Series:
    daily = equity_curve.resample("B").last().ffill()
    return daily.pct_change().dropna()


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series, initial_capital: float) -> Dict[str, float]:
    if equity_curve.empty:
        equity_curve = pd.Series([initial_capital], index=[pd.Timestamp.utcnow().tz_localize(None)])

    daily_returns = _daily_returns(equity_curve)
    drawdown = compute_drawdown(equity_curve)
    years = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 1.0 / 365.25)
    total_return = equity_curve.iloc[-1] / max(equity_curve.iloc[0], 1e-9) - 1.0
    cagr = (equity_curve.iloc[-1] / max(equity_curve.iloc[0], 1e-9)) ** (1.0 / years) - 1.0
    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0:
        sharpe = daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(252.0)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std(ddof=0) > 0:
            sortino = daily_returns.mean() / downside.std(ddof=0) * np.sqrt(252.0)

    if trades.empty:
        return {
            "trade_count": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(drawdown.min() if len(drawdown) else 0.0),
            "avg_trade": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "total_return": float(total_return),
        }

    winning_trades = trades[trades["net_pnl"] > 0]
    losing_trades = trades[trades["net_pnl"] <= 0]
    gross_profit = winning_trades["net_pnl"].sum()
    gross_loss = abs(losing_trades["net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    holding_period = trades["exit_time"] - trades["entry_time"]
    span_seconds = max((equity_curve.index[-1] - equity_curve.index[0]).total_seconds(), 1.0)
    exposure = holding_period.dt.total_seconds().sum() / span_seconds
    turnover = (trades["entry_notional"].sum() + trades["exit_notional"].sum()) / max(initial_capital, 1e-9)

    return {
        "trade_count": float(len(trades)),
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "profit_factor": float(profit_factor),
        "expectancy": float(trades["return_pct"].mean()),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min() if len(drawdown) else 0.0),
        "avg_trade": float(trades["net_pnl"].mean()),
        "exposure": float(exposure),
        "turnover": float(turnover),
        "total_return": float(total_return),
    }


def combine_equity_curves(curves: Iterable[pd.Series]) -> pd.Series:
    prepared: List[pd.Series] = []
    for curve in curves:
        if curve.empty:
            continue
        daily = curve.resample("B").last().ffill()
        normalized = daily / max(float(daily.iloc[0]), 1e-9)
        prepared.append(normalized)
    if not prepared:
        return pd.Series(dtype=float)
    combined = pd.concat(prepared, axis=1).ffill().dropna(how="all")
    return combined.mean(axis=1)
