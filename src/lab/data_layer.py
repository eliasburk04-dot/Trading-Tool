from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import yfinance as yf

from src.lab.sessions import SessionRules, assess_bar_quality, filter_session_bars, normalize_market_bars
from src.lab.timeframes import Timeframe


class BarSource:
    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: str,
        end_date: str,
        session_rules: SessionRules,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        raise NotImplementedError


class YahooBarSource(BarSource):
    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: str,
        end_date: str,
        session_rules: SessionRules,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        requested_start = date.fromisoformat(start_date)
        requested_end = date.fromisoformat(end_date)
        coverage_warning = None
        effective_start = requested_start
        max_history_days = timeframe.max_history_days
        if max_history_days is not None and (requested_end - requested_start).days > max_history_days:
            effective_start = requested_end - timedelta(days=max_history_days)
            coverage_warning = (
                "Requested %s to %s for %s; Yahoo best-effort coverage limited to %s starting %s"
                % (start_date, end_date, timeframe.value, max_history_days, effective_start.isoformat())
            )

        ticker = yf.Ticker(symbol)
        frame = ticker.history(
            start=effective_start.isoformat(),
            end=end_date,
            interval=timeframe.yahoo_interval,
            auto_adjust=True,
            actions=False,
        )
        if frame is None or frame.empty:
            raise ValueError("No data returned for %s" % symbol)
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        normalized = filter_session_bars(normalize_market_bars(frame, timeframe, session_rules), session_rules)
        quality = assess_bar_quality(normalized, timeframe, session_rules)
        quality["coverage_warning"] = coverage_warning
        quality["requested_start"] = start_date
        quality["requested_end"] = end_date
        quality["effective_start"] = effective_start.isoformat()
        return normalized, quality


@dataclass
class BarStore:
    root: Path

    def cache_path(self, symbol: str, timeframe: str | Timeframe, start_date: str, end_date: str) -> Path:
        timeframe_value = Timeframe.from_value(timeframe).value
        safe_symbol = symbol.replace("^", "_").replace("=", "_")
        return self.root / safe_symbol / timeframe_value / ("%s_%s.parquet" % (start_date, end_date))

    def read(self, symbol: str, timeframe: str | Timeframe, start_date: str, end_date: str) -> pd.DataFrame | None:
        path = self.cache_path(symbol, timeframe, start_date, end_date)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def write(self, symbol: str, timeframe: str | Timeframe, start_date: str, end_date: str, frame: pd.DataFrame) -> Path:
        path = self.cache_path(symbol, timeframe, start_date, end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return path


def load_market_data(config: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    research = config["research"]
    session_config = config["sessions"]["default"]
    session_rules = SessionRules(**session_config)
    timeframes = [Timeframe.from_value(value) for value in research["timeframes"]]
    start_date = str(research["start_date"])
    end_date = str(research["end_date"])
    store = BarStore(Path(str(config["data"]["cache_dir"])))
    source = YahooBarSource()

    market_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for instrument, instrument_config in config["data"]["instruments"].items():
        symbols = [instrument_config["primary"]] + list(instrument_config.get("alternates", []))
        market_data[instrument] = {}
        for timeframe in timeframes:
            used_symbol = None
            loaded_frame = None
            quality = None
            for symbol in symbols:
                cached = store.read(symbol, timeframe, start_date, end_date)
                if cached is not None:
                    loaded_frame = filter_session_bars(normalize_market_bars(cached, timeframe, session_rules), session_rules)
                    quality = assess_bar_quality(loaded_frame, timeframe, session_rules)
                    used_symbol = symbol
                    break
                try:
                    loaded_frame, quality = source.fetch_bars(symbol, timeframe, start_date, end_date, session_rules)
                    store.write(symbol, timeframe, start_date, end_date, loaded_frame)
                    used_symbol = symbol
                    break
                except Exception:
                    continue
            if loaded_frame is None or used_symbol is None or quality is None:
                raise RuntimeError("Unable to load data for %s %s" % (instrument, timeframe.value))
            market_data[instrument][timeframe.value] = {
                "symbol": used_symbol,
                "prices": loaded_frame,
                "coverage": quality,
                "session_rules": session_rules.to_dict(),
            }
    return market_data
