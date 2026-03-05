"""
Configuration and default parameters for the One-Candle SMC Strategy backtest.
All parameters are centralised here for easy tuning.
"""
from __future__ import annotations
import os

# ── Asset Definitions ─────────────────────────────────────────────────────────
# Each asset has a list of candidate symbols (tried in order) and a tick size.
ASSETS = {
    "Gold": {
        "symbols": ["GC=F", "GLD"],
        "tick_size": 0.10,
    },
    "Nasdaq": {
        "symbols": ["NQ=F", "QQQ"],
        "tick_size": 0.25,
    },
    "SP500": {
        "symbols": ["ES=F", "SPY"],
        "tick_size": 0.25,
    },
}

# ── Timeframe Pairs ───────────────────────────────────────────────────────────
# (setup_tf, entry_tf, max_lookback_days, human-readable label)
# yfinance limits:  1h → 730 d,  15m/5m → 60 d,  1d → unlimited
TF_CONFIGS: list[tuple[str, str, int, str]] = [
    ("1h",  "1h",  729,  "1h / 1h  (≈2 yr)"),
    ("4h",  "1h",  729,  "4h / 1h  (≈2 yr)"),
    ("1d",  "1d",  1095, "1d / 1d  (3 yr)"),
    ("15m", "5m",  59,   "15m / 5m (60 d)"),
    ("5m",  "5m",  59,   "5m / 5m  (60 d)"),
]

# ── Default Strategy Parameters ──────────────────────────────────────────────
DEFAULT_PARAMS: dict = {
    # Structure detection
    "swing_lookback":      10,       # N – rolling window for swing high/low
    "sweep_lookback":      10,       # M – look-back for liquidity sweep check
    "use_sweep_filter":    False,    # toggle liquidity-sweep pre-filter

    # Entry
    "entry_validity":      5,        # K – max bars to wait for limit fill
    "entry_retrace_pct":   0.5,      # 50 % retrace of reference candle

    # Risk management
    "rr":                  3.0,      # reward / risk ratio (used for fixed-RR TP)
    "sl_buffer_atr":       0.1,      # SL buffer expressed as fraction of ATR
    "atr_period":          14,
    "tp_mode":             "fixed_rr",  # "fixed_rr" | "liquidity_target"

    # Trade management
    "use_breakeven":       False,
    "breakeven_rr":        1.0,      # move SL to entry when unrealised R = this
    "use_trailing":        False,
    "trailing_atr_mult":   2.0,
    "max_positions":       1,        # max concurrent positions per asset

    # Costs
    "slippage_pct":        0.01,     # one-way slippage in %
    "fee_pct":             0.02,     # round-trip fee in %
}

# ── Parameter-Sweep Grid ─────────────────────────────────────────────────────
SWEEP_GRID: dict[str, list] = {
    "swing_lookback":   [5, 10, 15, 20],
    "sweep_lookback":   [5, 10, 15],
    "entry_validity":   [3, 5, 10],
    "rr":               [2.0, 3.0, 4.0],
}

# ── Walk-Forward Split ────────────────────────────────────────────────────────
WF_TRAIN_FRAC = 0.67          # first 67 % of data for training
WF_OPTIMISE_METRIC = "sharpe" # metric used to pick best params in training

# ── Output Directories ────────────────────────────────────────────────────────
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "output")
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Timeframe → seconds mapping ──────────────────────────────────────────────
TF_SECONDS: dict[str, int] = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}
