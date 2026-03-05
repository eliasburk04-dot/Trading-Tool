from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from src.lab.data_layer import BarStore, load_market_data
from src.lab.timeframes import Timeframe


def cache_file_name(symbol: str, interval: str, start_date: str, end_date: str) -> str:
    timeframe = Timeframe.from_value(interval)
    return BarStore(Path(".")).cache_path(symbol, timeframe, start_date, end_date).name


def load_symbol_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
    store = BarStore(cache_dir)
    cached = store.read(symbol, interval, start_date, end_date)
    if cached is None:
        raise FileNotFoundError("No cached data for %s %s" % (symbol, interval))
    return cached


__all__ = ["cache_file_name", "load_market_data", "load_symbol_data"]
