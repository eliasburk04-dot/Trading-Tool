from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List


class Timeframe(str, Enum):
    D1 = "D1"
    H1 = "H1"
    M15 = "M15"
    M5 = "M5"

    @classmethod
    def from_value(cls, value: str | "Timeframe") -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        normalized = value.strip().upper()
        aliases = {
            "1D": cls.D1,
            "D1": cls.D1,
            "1H": cls.H1,
            "H1": cls.H1,
            "15M": cls.M15,
            "M15": cls.M15,
            "5M": cls.M5,
            "M5": cls.M5,
        }
        if normalized not in aliases:
            raise ValueError("Unsupported timeframe: %s" % value)
        return aliases[normalized]

    @property
    def yahoo_interval(self) -> str:
        mapping = {
            Timeframe.D1: "1d",
            Timeframe.H1: "1h",
            Timeframe.M15: "15m",
            Timeframe.M5: "5m",
        }
        return mapping[self]

    @property
    def pandas_frequency(self) -> str:
        mapping = {
            Timeframe.D1: "1D",
            Timeframe.H1: "1h",
            Timeframe.M15: "15min",
            Timeframe.M5: "5min",
        }
        return mapping[self]

    @property
    def max_history_days(self) -> int | None:
        mapping = {
            Timeframe.D1: None,
            Timeframe.H1: 730,
            Timeframe.M15: 60,
            Timeframe.M5: 60,
        }
        return mapping[self]

    @property
    def bars_per_day(self) -> int:
        mapping = {
            Timeframe.D1: 1,
            Timeframe.H1: 7,
            Timeframe.M15: 26,
            Timeframe.M5: 78,
        }
        return mapping[self]


def parse_timeframes(raw_value: str | Iterable[str] | None) -> List[Timeframe]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        pieces = [piece for piece in raw_value.split(",") if piece.strip()]
    else:
        pieces = list(raw_value)
    return [Timeframe.from_value(piece) for piece in pieces]


def default_frequency_guardrails() -> Dict[str, Dict[str, float]]:
    return {
        Timeframe.D1.value: {
            "min_total_trades": 150,
            "min_trades_per_day": 0.15,
            "min_oos_trades_per_split": 30,
        },
        Timeframe.H1.value: {
            "min_total_trades": 300,
            "min_trades_per_day": 0.5,
            "min_oos_trades_per_split": 30,
        },
        Timeframe.M15.value: {
            "min_total_trades": 600,
            "min_trades_per_day": 1.0,
            "min_oos_trades_per_split": 30,
        },
        Timeframe.M5.value: {
            "min_total_trades": 600,
            "min_trades_per_day": 1.0,
            "min_oos_trades_per_split": 30,
        },
    }
