from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, and drop unusable OHLC rows."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing OHLCV columns: %s" % ", ".join(missing))

    cleaned = frame.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    if getattr(cleaned.index, "tz", None) is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    cleaned["Volume"] = cleaned["Volume"].fillna(0.0)
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
    return cleaned


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.clip(lower=0.0, upper=100.0)


def bollinger_bands(close: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower}, index=close.index)


def rate_of_change(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(periods=period)


def annualized_volatility(close: pd.Series, window: int) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
