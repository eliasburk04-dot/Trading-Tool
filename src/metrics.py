"""
Performance-metrics module.

Computes standard quant metrics from a trades DataFrame produced by
``run_backtest``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_equity_curve(
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Compound equity curve indexed by exit time (+ start point)."""
    if trades.empty:
        return pd.Series([initial_capital], index=[pd.Timestamp.now()])
    eq = [initial_capital]
    for pnl in trades["pnl_pct"].values:
        eq.append(eq[-1] * (1.0 + pnl))
    times = [trades["entry_time"].iloc[0]] + list(trades["exit_time"])
    return pd.Series(eq, index=times)


def _daily_returns(equity: pd.Series) -> pd.Series:
    """Resample equity to business-day frequency and compute returns."""
    daily = equity.resample("B").last().ffill()
    return daily.pct_change().dropna()


def compute_metrics(
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> dict[str, object]:
    """
    Return a dict of labelled performance metrics.

    Keys: Total Trades, Win Rate, Avg Win, Avg Loss, Profit Factor,
    Expectancy, Total Return, CAGR, Sharpe, Sortino, Max Drawdown,
    Calmar, Trades/Month, Avg Hold Time, Exposure %.
    """
    empty: dict[str, object] = {
        "Total Trades": 0, "Win Rate": "—", "Avg Win": "—", "Avg Loss": "—",
        "Profit Factor": "—", "Expectancy": "—", "Total Return": "—",
        "CAGR": "—", "Sharpe": "—", "Sortino": "—", "Max Drawdown": "—",
        "Calmar": "—", "Trades/Month": "—", "Avg Hold Time": "—",
        "Exposure %": "—",
    }
    if trades.empty or len(trades) == 0:
        return empty

    n = len(trades)
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]

    winrate = len(wins) / n
    avg_win  = wins["pnl_pct"].mean()   if len(wins)   else 0.0
    avg_loss = losses["pnl_pct"].mean() if len(losses)  else 0.0
    pf_denom = abs(losses["pnl_pct"].sum()) if len(losses) else 1e-10
    profit_factor = wins["pnl_pct"].sum() / pf_denom if pf_denom > 0 else np.inf
    expectancy = trades["pnl_pct"].mean()

    # equity curve
    eq = build_equity_curve(trades, initial_capital)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0

    first = trades["entry_time"].iloc[0]
    last  = trades["exit_time"].iloc[-1]
    secs  = (last - first).total_seconds()
    years = max(secs / (365.25 * 86400), 1e-6)

    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0

    # drawdown
    peak = np.maximum.accumulate(eq.values)
    dd   = (eq.values - peak) / peak
    max_dd = dd.min()

    # daily Sharpe & Sortino
    dr = _daily_returns(eq)
    if len(dr) > 1 and dr.std() > 0:
        sharpe = dr.mean() / dr.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    neg = dr[dr < 0]
    down_std = neg.std() if len(neg) > 1 else 1e-10
    sortino = dr.mean() / down_std * np.sqrt(252) if down_std > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    trades_per_mo = n / max(years * 12, 1e-6)

    # avg hold time
    hold = trades["exit_time"] - trades["entry_time"]
    avg_hold = hold.mean()

    # exposure (fraction of time in a trade)
    total_hold = hold.sum().total_seconds()
    exposure = total_hold / max(secs, 1)

    return {
        "Total Trades":  n,
        "Win Rate":      f"{winrate:.1%}",
        "Avg Win":       f"{avg_win:.4f}",
        "Avg Loss":      f"{avg_loss:.4f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Expectancy":    f"{expectancy:.5f}",
        "Total Return":  f"{total_ret:.1%}",
        "CAGR":          f"{cagr:.1%}",
        "Sharpe":        f"{sharpe:.2f}",
        "Sortino":       f"{sortino:.2f}",
        "Max Drawdown":  f"{max_dd:.1%}",
        "Calmar":        f"{calmar:.2f}",
        "Trades/Month":  f"{trades_per_mo:.1f}",
        "Avg Hold Time": str(avg_hold),
        "Exposure %":    f"{exposure:.1%}",
    }


def metrics_to_series(m: dict) -> pd.Series:
    """Convert metrics dict to a labelled pandas Series (handy for tables)."""
    return pd.Series(m)


def numeric_metrics(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> dict[str, float]:
    """
    Return *numeric* (float) versions of the key metrics – used by the
    parameter sweep to rank configurations.
    """
    if trades.empty:
        return {"sharpe": -99.0, "cagr": -99.0, "pf": 0.0,
                "winrate": 0.0, "total_return": -1.0, "max_dd": -1.0,
                "n_trades": 0}

    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    winrate = len(wins) / n
    pf_denom = abs(losses["pnl_pct"].sum()) if len(losses) else 1e-10
    pf = wins["pnl_pct"].sum() / pf_denom if pf_denom > 0 else 999.0

    eq = build_equity_curve(trades, initial_capital)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    secs = (trades["exit_time"].iloc[-1] - trades["entry_time"].iloc[0]).total_seconds()
    years = max(secs / (365.25 * 86400), 1e-6)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0

    peak = np.maximum.accumulate(eq.values)
    dd = (eq.values - peak) / peak
    max_dd = dd.min()

    dr = _daily_returns(eq)
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "pf": float(pf),
        "winrate": float(winrate),
        "total_return": float(total_ret),
        "max_dd": float(max_dd),
        "n_trades": int(n),
    }
