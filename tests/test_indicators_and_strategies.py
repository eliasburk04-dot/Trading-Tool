from __future__ import annotations

import pandas as pd

from src.lab.indicators import compute_rsi, sanitize_ohlcv
from src.lab.strategies import get_strategy


def test_sanitize_ohlcv_drops_nans_and_sorts(sample_prices: pd.DataFrame) -> None:
    frame = sample_prices.iloc[:20].copy()
    frame.iloc[2, frame.columns.get_loc("Close")] = float("nan")
    frame = frame.iloc[::-1]

    sanitized = sanitize_ohlcv(frame)

    assert sanitized.index.is_monotonic_increasing
    assert sanitized["Close"].isna().sum() == 0
    assert len(sanitized) == 19


def test_rsi_stays_bounded_with_missing_values(sample_prices: pd.DataFrame) -> None:
    close = sample_prices["Close"].iloc[:40].copy()
    close.iloc[5] = float("nan")
    close.iloc[12] = float("nan")

    rsi = compute_rsi(close, period=5)

    bounded = rsi.dropna()
    assert not bounded.empty
    assert bounded.between(0, 100).all()


def test_ma_crossover_generates_entries_and_exits() -> None:
    close = [10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8, 9]
    index = pd.date_range("2024-01-01", periods=len(close), freq="B")
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": [value + 0.5 for value in close],
            "Low": [value - 0.5 for value in close],
            "Close": close,
            "Volume": [1_000] * len(close),
        },
        index=index,
    )

    strategy = get_strategy("ma_crossover")
    signals = strategy.generate_signals(
        frame,
        {
            "fast_window": 2,
            "slow_window": 4,
        },
    )

    assert int(signals["entry"].sum()) >= 1
    assert int(signals["exit"].sum()) >= 1
