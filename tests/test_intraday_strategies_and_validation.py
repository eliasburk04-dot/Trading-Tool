from __future__ import annotations

import copy

import pandas as pd

from src.lab.config import DEFAULT_CONFIG
from src.lab.research import evaluate_guardrails, generate_purged_walk_forward_windows
from src.lab.strategies import get_strategy
from src.lab.timeframes import Timeframe


def test_vwap_mean_reversion_supports_intraday_and_generates_signal() -> None:
    index = pd.date_range("2024-01-02 09:30", periods=12, freq="15min", tz="America/New_York")
    close = [100, 100.2, 100.1, 99.8, 99.2, 98.8, 99.0, 99.4, 99.8, 100.0, 100.1, 100.2]
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": [value + 0.15 for value in close],
            "Low": [value - 0.15 for value in close],
            "Close": close,
            "Volume": [1_000, 1_100, 1_050, 1_200, 1_500, 1_700, 1_600, 1_400, 1_300, 1_250, 1_200, 1_150],
        },
        index=index,
    )

    strategy = get_strategy("vwap_mean_reversion")
    signals = strategy.generate_signals(
        frame,
        {
            "atr_window": 3,
            "deviation_atr": 0.6,
            "stall_bars": 2,
            "time_exit_bars": 4,
        },
    )

    assert Timeframe.M15 in strategy.supports_timeframes()
    assert strategy.session_rules()["flatten_time"] == "15:55"
    assert int(signals["entry"].sum()) >= 1


def test_vwap_mean_reversion_handles_zero_volume_without_na_boolean_errors() -> None:
    index = pd.date_range("2024-01-02 09:30", periods=8, freq="1h", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "Open": [100.0, 100.2, 99.8, 99.5, 99.7, 100.1, 100.4, 100.2],
            "High": [100.3, 100.4, 100.0, 99.8, 100.0, 100.4, 100.6, 100.4],
            "Low": [99.8, 100.0, 99.4, 99.2, 99.5, 99.9, 100.1, 100.0],
            "Close": [100.1, 100.1, 99.6, 99.4, 99.9, 100.2, 100.3, 100.1],
            "Volume": [0, 0, 1_200, 1_500, 1_300, 1_250, 1_180, 1_140],
        },
        index=index,
    )

    strategy = get_strategy("vwap_mean_reversion")
    signals = strategy.generate_signals(
        frame,
        {
            "atr_window": 3,
            "deviation_atr": 0.6,
            "stall_bars": 2,
            "time_exit_bars": 4,
        },
    )

    assert signals["entry"].dtype == bool
    assert signals["exit"].dtype == bool
    assert signals.isna().sum().sum() == 0


def test_opening_range_breakout_generates_entry_after_break() -> None:
    index = pd.date_range("2024-01-02 09:30", periods=12, freq="15min", tz="America/New_York")
    close = [100.0, 100.1, 100.0, 100.05, 100.1, 100.2, 100.8, 101.0, 101.1, 101.0, 100.9, 100.8]
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": [value + 0.2 for value in close],
            "Low": [value - 0.2 for value in close],
            "Close": close,
            "Volume": [1_000] * len(close),
        },
        index=index,
    )

    strategy = get_strategy("opening_range_breakout")
    signals = strategy.generate_signals(
        frame,
        {
            "opening_range_bars": 4,
            "breakout_buffer_atr": 0.1,
            "atr_window": 3,
            "time_exit_bars": 4,
        },
    )

    assert int(signals["entry"].sum()) >= 1


def test_generate_purged_walk_forward_windows_respects_gaps() -> None:
    index = pd.date_range("2024-01-02 09:30", periods=80, freq="1h", tz="America/New_York")

    windows = generate_purged_walk_forward_windows(
        index=index,
        train_bars=20,
        test_bars=10,
        step_bars=10,
        purge_bars=2,
        embargo_bars=3,
    )

    assert windows
    first_window = windows[0]
    assert first_window["train_end_idx"] < first_window["test_start_idx"]
    assert first_window["test_start_idx"] - first_window["train_end_idx"] >= 4


def test_frequency_guardrails_fail_low_trade_candidate() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["guardrails"]["frequency"] = {
        "H1": {
            "min_total_trades": 300,
            "min_trades_per_day": 0.5,
            "min_oos_trades_per_split": 30,
        }
    }
    metrics = {
        "trade_count": 25.0,
        "max_drawdown": -0.05,
    }

    passes, notes = evaluate_guardrails(
        out_of_sample_metrics=metrics,
        split_trade_counts=[4, 6, 5],
        timeframe=Timeframe.H1,
        config=config,
        in_sample_score=0.6,
        out_of_sample_score=0.4,
        active_days=90,
        baseline_margin=0.05,
    )

    assert passes is False
    assert notes["frequency_ok"] is False
