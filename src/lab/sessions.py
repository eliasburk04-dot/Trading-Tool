from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from src.lab.indicators import sanitize_ohlcv
from src.lab.timeframes import Timeframe


@dataclass(frozen=True)
class SessionRules:
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    trade_start: str = "09:45"
    trade_end: str = "15:30"
    flatten_time: str = "15:55"
    allow_overnight: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_market_bars(frame: pd.DataFrame, timeframe: Timeframe, rules: SessionRules) -> pd.DataFrame:
    cleaned = sanitize_ohlcv(frame)
    timezone = ZoneInfo(rules.timezone)
    if cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize(timezone)
    else:
        cleaned.index = cleaned.index.tz_convert(timezone)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned


def filter_session_bars(frame: pd.DataFrame, rules: SessionRules) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.between_time(rules.session_open, rules.session_close, inclusive="left")


def assess_bar_quality(frame: pd.DataFrame, timeframe: Timeframe, rules: SessionRules) -> Dict[str, Any]:
    normalized = filter_session_bars(normalize_market_bars(frame, timeframe, rules), rules)
    expected_diff = pd.Timedelta(timeframe.pandas_frequency)
    gap_count = 0
    partial_sessions = 0
    duplicate_count = int(normalized.index.duplicated().sum())

    if len(normalized.index) > 1:
        diffs = normalized.index.to_series().diff().dropna()
        intraday_gaps = diffs[(diffs > expected_diff) & (diffs < pd.Timedelta(days=1))]
        gap_count = int(len(intraday_gaps))

    if timeframe != Timeframe.D1:
        daily_counts = normalized.groupby(normalized.index.normalize()).size()
        expected = timeframe.bars_per_day
        partial_sessions = int((daily_counts < expected).sum())

    warnings = []
    if gap_count:
        warnings.append("Detected %d intraday gaps" % gap_count)
    if partial_sessions:
        warnings.append("Detected %d partial sessions" % partial_sessions)
    if duplicate_count:
        warnings.append("Detected %d duplicate bars" % duplicate_count)

    coverage = {
        "gap_count": gap_count,
        "partial_session_count": partial_sessions,
        "duplicate_count": duplicate_count,
        "warning_count": len(warnings),
        "warnings": warnings,
        "start": normalized.index.min().isoformat() if not normalized.empty else None,
        "end": normalized.index.max().isoformat() if not normalized.empty else None,
        "bar_count": int(len(normalized)),
    }
    return coverage


def is_end_of_day_bar(timestamp: pd.Timestamp, next_timestamp: pd.Timestamp | None, rules: SessionRules) -> bool:
    current = timestamp.tz_convert(rules.timezone) if timestamp.tzinfo is not None else timestamp.tz_localize(rules.timezone)
    flatten_time = pd.Timestamp(current.date()).tz_localize(rules.timezone) + pd.Timedelta(rules.flatten_time + ":00")
    if current >= flatten_time:
        return True
    if next_timestamp is None:
        return True
    next_value = next_timestamp.tz_convert(rules.timezone) if next_timestamp.tzinfo is not None else next_timestamp.tz_localize(rules.timezone)
    return current.date() != next_value.date()
