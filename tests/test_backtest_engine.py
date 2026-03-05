from __future__ import annotations

import pandas as pd

from src.lab.backtest import BacktestConfig, run_backtest


def test_backtest_costs_reduce_pnl() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "Open": [100, 101, 103, 104, 105, 106],
            "High": [101, 104, 105, 106, 107, 108],
            "Low": [99, 100, 102, 103, 104, 105],
            "Close": [100, 103, 104, 105, 106, 107],
            "Volume": [10_000] * 6,
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "entry": [False, True, False, False, False, False],
            "exit": [False, False, False, True, False, False],
        },
        index=index,
    )

    no_costs = BacktestConfig(
        initial_capital=10_000,
        position_fraction=0.5,
        max_exposure=0.5,
        transaction_cost_bps=0,
        slippage_bps=0,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        trailing_stop_pct=0.0,
        max_drawdown_circuit_breaker=0.5,
        max_concurrent_trades=1,
        cooldown_bars_after_loss=0,
    )
    with_costs = BacktestConfig(
        initial_capital=10_000,
        position_fraction=0.5,
        max_exposure=0.5,
        transaction_cost_bps=20,
        slippage_bps=10,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        trailing_stop_pct=0.0,
        max_drawdown_circuit_breaker=0.5,
        max_concurrent_trades=1,
        cooldown_bars_after_loss=0,
    )

    result_no_costs = run_backtest(prices, signals, no_costs, instrument="Gold")
    result_with_costs = run_backtest(prices, signals, with_costs, instrument="Gold")

    assert len(result_no_costs.trades) == 1
    assert len(result_with_costs.trades) == 1
    assert result_with_costs.trades["net_pnl"].iloc[0] < result_no_costs.trades["net_pnl"].iloc[0]


def test_backtest_enforces_cooldown_after_loss() -> None:
    index = pd.date_range("2024-02-01", periods=7, freq="B")
    prices = pd.DataFrame(
        {
            "Open": [100, 100, 98, 97, 98, 99, 100],
            "High": [101, 100.5, 98.5, 98, 99, 100, 101],
            "Low": [99, 96, 95.5, 96.5, 97.5, 98.5, 99.5],
            "Close": [100, 97, 96, 97.5, 98.5, 99.5, 100.5],
            "Volume": [8_000] * 7,
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "entry": [False, True, False, True, False, False, False],
            "exit": [False, False, False, False, False, False, False],
        },
        index=index,
    )
    config = BacktestConfig(
        initial_capital=10_000,
        position_fraction=0.5,
        max_exposure=0.5,
        transaction_cost_bps=0,
        slippage_bps=0,
        stop_loss_pct=0.02,
        take_profit_pct=0.10,
        trailing_stop_pct=0.0,
        max_drawdown_circuit_breaker=0.5,
        max_concurrent_trades=1,
        cooldown_bars_after_loss=3,
    )

    result = run_backtest(prices, signals, config, instrument="Gold")

    assert len(result.trades) == 1
    assert result.trades["exit_reason"].iloc[0] == "stop_loss"
