"""
Data-fetching module.
 • Downloads OHLCV via *yfinance* with symbol fallbacks.
 • Caches to Parquet for reproducibility.
 • Provides a resampler for higher time-frames.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ── helpers ───────────────────────────────────────────────────────────────────

_RESAMPLE_MAP = {
    "1m": "1min", "2m": "2min", "5m": "5min",
    "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h",
    "1d": "1D",
}


def _safe_cache_path(cache_dir: str, symbol: str, interval: str, days: int) -> str:
    clean = symbol.replace("=", "_").replace("^", "_")
    return os.path.join(cache_dir, f"{clean}_{interval}_{days}d.parquet")


# ── public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Return OHLCV DataFrame from yfinance, or *None* on failure."""
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(interval=interval, start=start, end=end, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.dropna(inplace=True)
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] {symbol} @ {interval}: {exc}")
        return None


def get_asset_data(
    asset_name: str,
    symbols: list[str],
    interval: str,
    lookback_days: int,
    cache_dir: str = "data",
) -> tuple[pd.DataFrame | None, str]:
    """
    Try each *symbol* in order until one returns ≥ 30 bars.
    Returns ``(DataFrame, used_symbol)`` or ``(None, "")``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_days)
    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")

    for sym in symbols:
        cache_fp = _safe_cache_path(cache_dir, sym, interval, lookback_days)

        # try cache
        if os.path.exists(cache_fp):
            try:
                df = pd.read_parquet(cache_fp)
                if len(df) >= 30:
                    print(f"  [CACHE] {asset_name}: {sym} @ {interval}  ({len(df)} bars)")
                    return df, sym
            except Exception:
                pass

        # fetch live
        print(f"  [FETCH] {asset_name}: trying {sym} @ {interval} …")
        df = fetch_ohlcv(sym, interval, start_s, end_s)
        if df is not None and len(df) >= 30:
            try:
                df.to_parquet(cache_fp)
            except Exception:
                pass
            print(f"  [OK]    {asset_name}: {sym} @ {interval}  ({len(df)} bars)")
            return df, sym

    print(f"  [FAIL]  {asset_name}: no data for {symbols} @ {interval}")
    return None, ""


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample to a coarser bar size.  Drops incomplete bars."""
    rule = _RESAMPLE_MAP.get(target_tf, target_tf)
    out = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "Close"])
    return out
