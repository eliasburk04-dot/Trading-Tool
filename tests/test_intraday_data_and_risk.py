from __future__ import annotations

import pandas as pd

from src.lab.backtest import BacktestConfig, run_backtest
from src.lab.execution.engine import FillModel
from src.lab.execution.state import ExecutionState
from src.lab.risk.limits import DailyRiskState, pre_trade_risk_check
from src.lab.sessions import SessionRules, assess_bar_quality, normalize_market_bars
from src.lab.timeframes import Timeframe, parse_timeframes


def test_parse_timeframes_and_normalize_intraday_bars(sample_intraday_prices: pd.DataFrame) -> None:
    parsed = parse_timeframes("H1,M15")
    assert parsed == [Timeframe.H1, Timeframe.M15]

    rules = SessionRules()
    shuffled = sample_intraday_prices.iloc[::-1].copy()
    shuffled = pd.concat([shuffled, shuffled.iloc[[0]]])

    normalized = normalize_market_bars(shuffled, Timeframe.H1, rules)

    assert normalized.index.is_monotonic_increasing
    assert normalized.index.tz is not None
    assert normalized.index.tz.key == rules.timezone
    assert normalized.index.duplicated().sum() == 0


def test_assess_bar_quality_detects_gap(sample_intraday_prices: pd.DataFrame) -> None:
    rules = SessionRules()
    frame = sample_intraday_prices.iloc[:-1].copy()
    frame = frame.drop(frame.index[3])

    quality = assess_bar_quality(frame, Timeframe.H1, rules)

    assert quality["gap_count"] >= 1
    assert quality["warning_count"] >= 1


def test_pre_trade_risk_check_blocks_after_daily_loss_and_trade_cap() -> None:
    state = ExecutionState(
        cash=100_000.0,
        realized_pnl=0.0,
        open_positions={},
        last_processed_at=None,
        risk=DailyRiskState(
            session_date="2024-01-03",
            realized_pnl=-1_500.0,
            consecutive_losses=2,
            trades_taken=4,
        ),
    )
    decision = pre_trade_risk_check(
        timestamp=pd.Timestamp("2024-01-03T15:00:00", tz="America/New_York"),
        state=state,
        risk_config={
            "initial_capital": 100_000.0,
            "daily_loss_limit_pct": 0.01,
            "max_trades_per_day": 4,
            "max_exposure": 0.20,
            "max_notional": 25_000.0,
            "max_consecutive_losses_per_day": 2,
        },
        proposed_notional=10_000.0,
    )

    assert decision.allowed is False
    assert decision.reason in {"daily_loss_limit", "max_trades_per_day", "consecutive_losses"}


def test_fill_model_is_deterministic_given_seed() -> None:
    model_a = FillModel(spread_bps=2.0, slippage_bps=1.0, seed=7)
    model_b = FillModel(spread_bps=2.0, slippage_bps=1.0, seed=7)

    fill_a = model_a.fill_price(side="buy", open_price=100.0, high_price=101.0, low_price=99.0, limit_price=100.2)
    fill_b = model_b.fill_price(side="buy", open_price=100.0, high_price=101.0, low_price=99.0, limit_price=100.2)

    assert fill_a == fill_b


def test_backtest_flattens_end_of_day_for_intraday() -> None:
    index = pd.date_range(
        "2024-01-02 09:30",
        periods=7,
        freq="1h",
        tz="America/New_York",
    )
    prices = pd.DataFrame(
        {
            "Open": [100, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
            "High": [100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
            "Low": [99.9, 100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "Close": [100.1, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8],
            "Volume": [1_000] * 7,
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "entry": [False, False, False, False, False, True, False],
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
        take_profit_pct=0.08,
        trailing_stop_pct=0.0,
        max_drawdown_circuit_breaker=0.5,
        max_concurrent_trades=1,
        cooldown_bars_after_loss=0,
        timeframe=Timeframe.H1,
        max_trades_per_day=4,
        daily_loss_limit_pct=0.03,
        max_consecutive_losses_per_day=3,
        flatten_end_of_day=True,
        allow_overnight=False,
        time_stop_bars=0,
    )

    result = run_backtest(prices, signals, config, instrument="Nasdaq")

    assert len(result.trades) == 1
    assert result.trades["exit_reason"].iloc[0] == "end_of_day"
