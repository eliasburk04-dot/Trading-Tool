from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def build_price_frame(
    start: str = "2022-01-03",
    periods: int = 800,
    seed: int = 7,
    drift: float = 0.25,
    volatility: float = 1.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq="B")
    base_noise = rng.normal(loc=drift, scale=volatility, size=periods)
    seasonal = 2.5 * np.sin(np.linspace(0, 18, periods))
    close = 100 + np.cumsum(base_noise + seasonal / 15.0)
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0, 0.4, periods)
    high = np.maximum(open_, close) + rng.uniform(0.2, 1.1, periods)
    low = np.minimum(open_, close) - rng.uniform(0.2, 1.1, periods)
    volume = rng.integers(10_000, 50_000, size=periods)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def build_intraday_frame(
    start: str = "2024-01-02",
    days: int = 5,
    timeframe: str = "1h",
    seed: int = 13,
    timezone: str = "America/New_York",
    base_price: float = 100.0,
) -> pd.DataFrame:
    freq_map = {
        "1h": "1h",
        "15m": "15min",
        "5m": "5min",
    }
    if timeframe not in freq_map:
        raise ValueError("Unsupported timeframe for fixture: %s" % timeframe)

    business_days = pd.bdate_range(start=start, periods=days, tz=timezone)
    index_parts: list[pd.DatetimeIndex] = []
    for business_day in business_days:
        day_start = business_day.normalize()
        session_open = day_start + pd.Timedelta(hours=9, minutes=30)
        session_close = day_start + pd.Timedelta(hours=16)
        index_parts.append(
            pd.date_range(
                start=session_open,
                end=session_close,
                freq=freq_map[timeframe],
                inclusive="left",
                tz=timezone,
            )
        )

    index = index_parts[0]
    for index_part in index_parts[1:]:
        index = index.append(index_part)

    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.03, scale=0.35, size=len(index))
    seasonal = 0.6 * np.sin(np.linspace(0, 12, len(index)))
    close = base_price + np.cumsum(increments + seasonal / 12.0)
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0, 0.12, len(index))
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.45, len(index))
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.45, len(index))
    volume = rng.integers(1_000, 7_000, size=len(index))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    return build_price_frame()


@pytest.fixture
def sample_intraday_prices() -> pd.DataFrame:
    return build_intraday_frame()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
