"""
Signal-generation module for the One-Candle SMC Strategy.

Terminology (short recap)
 • *Swing High / Low*  – rolling max(High) / min(Low) over the last N bars
   (shifted by 1 so the current bar is NOT included → no look-ahead).
 • *MSS* (Market-Structure Shift)
   – Bullish: close > swing_high   – Bearish: close < swing_low
 • *Liquidity Sweep* (optional filter)
   – Bullish: before the MSS bar, a recent low dipped below the swing_low
   – Bearish: before the MSS bar, a recent high poked above the swing_high
 • *Reference Candle* = the MSS bar itself (the "one candle").
 • *Entry* = 50 % retrace of the reference candle range.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    pc = df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ── Signal detection (vectorised on setup-TF) ────────────────────────────────

def detect_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Return a DataFrame (same index as *df*) with per-bar signal info.

    Columns produced
    ----------------
    signal         1 = bullish, −1 = bearish, 0 = none
    entry_price    limit order price (50 % retrace)
    sl_long / sl_short   stop-loss for each direction
    tp_long / tp_short   take-profit (fixed RR)
    tp_liq_long / tp_liq_short   liq-target TP (wider swing)
    ref_high / ref_low / ref_open / ref_close
    atr
    """
    N = int(params["swing_lookback"])

    # ── swing levels (look-back only, current bar excluded) ────────────────
    swing_high = df["High"].rolling(N, min_periods=N).max().shift(1)
    swing_low  = df["Low"].rolling(N, min_periods=N).min().shift(1)

    # ── MSS ────────────────────────────────────────────────────────────────
    bull_mss = df["Close"] > swing_high
    bear_mss = df["Close"] < swing_low

    # ── optional liquidity-sweep pre-condition ─────────────────────────────
    if params.get("use_sweep_filter", False):
        M = int(params["sweep_lookback"])
        # bullish: a recent low swept below swing_low
        recent_min_low  = df["Low"].rolling(M, min_periods=1).min().shift(1)
        bull_sweep = recent_min_low < swing_low
        # bearish: a recent high swept above swing_high
        recent_max_high = df["High"].rolling(M, min_periods=1).max().shift(1)
        bear_sweep = recent_max_high > swing_high
        bull_mss = bull_mss & bull_sweep
        bear_mss = bear_mss & bear_sweep

    # resolve rare double-fire by candle direction
    both = bull_mss & bear_mss
    candle_up = df["Close"] >= df["Open"]
    bull_mss = bull_mss & (~both | candle_up)
    bear_mss = bear_mss & (~both | ~candle_up)

    # ── build output frame ─────────────────────────────────────────────────
    sig = pd.DataFrame(index=df.index)
    sig["signal"] = 0
    sig.loc[bull_mss, "signal"] =  1
    sig.loc[bear_mss, "signal"] = -1

    # reference candle
    sig["ref_high"]  = df["High"]
    sig["ref_low"]   = df["Low"]
    sig["ref_open"]  = df["Open"]
    sig["ref_close"] = df["Close"]

    # entry price (50 % retrace)
    retrace = params.get("entry_retrace_pct", 0.5)
    sig["entry_price"] = df["Low"] + retrace * (df["High"] - df["Low"])

    # ATR & buffer
    atr = compute_atr(df, params.get("atr_period", 14))
    sig["atr"] = atr
    buf = params.get("sl_buffer_atr", 0.0) * atr

    # stop-loss
    sig["sl_long"]  = df["Low"]  - buf          # for bullish trades
    sig["sl_short"] = df["High"] + buf          # for bearish trades

    # take-profit – fixed RR
    rr = params.get("rr", 3.0)
    risk_long  = sig["entry_price"] - sig["sl_long"]
    risk_short = sig["sl_short"]    - sig["entry_price"]
    sig["tp_long"]  = sig["entry_price"] + rr * risk_long
    sig["tp_short"] = sig["entry_price"] - rr * risk_short

    # take-profit – liquidity-target (wider swing level, no look-ahead)
    wide = int(max(3 * N, 30))
    prev_swing_high = df["High"].shift(N).rolling(wide, min_periods=N).max()
    prev_swing_low  = df["Low"].shift(N).rolling(wide, min_periods=N).min()
    sig["tp_liq_long"]  = prev_swing_high
    sig["tp_liq_short"] = prev_swing_low

    # swing levels (exposed for plots / diagnostics)
    sig["swing_high"] = swing_high
    sig["swing_low"]  = swing_low

    return sig


# ── Convert to list of signal dicts for the backtester ────────────────────────

def prepare_signals_list(
    sig_df: pd.DataFrame,
    params: dict,
    tf_seconds: int,
) -> list[dict]:
    """
    Filter valid signals and produce a sorted list of dicts consumed by
    ``run_backtest``.  Each dict carries everything needed to place /
    manage one order.
    """
    tp_mode = params.get("tp_mode", "fixed_rr")
    rr      = params.get("rr", 3.0)
    mask    = sig_df["signal"] != 0
    rows    = sig_df.loc[mask]
    out: list[dict] = []

    for idx, r in rows.iterrows():
        d = int(r["signal"])
        entry = r["entry_price"]

        if d == 1:
            sl = r["sl_long"]
            tp_rr  = r["tp_long"]
            tp_liq = r["tp_liq_long"]
        else:
            sl = r["sl_short"]
            tp_rr  = r["tp_short"]
            tp_liq = r["tp_liq_short"]

        # choose TP mode, with fallback to fixed-RR
        if tp_mode == "liquidity_target" and np.isfinite(tp_liq):
            valid_liq = (d == 1 and tp_liq > entry) or (d == -1 and tp_liq < entry)
            tp = tp_liq if valid_liq else tp_rr
        else:
            tp = tp_rr

        # sanity: SL and TP must be on correct sides
        if d == 1 and (tp <= entry or sl >= entry):
            continue
        if d == -1 and (tp >= entry or sl <= entry):
            continue
        if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp)):
            continue

        out.append({
            "signal_time":  idx,
            "entry_after":  idx + pd.Timedelta(seconds=tf_seconds),
            "direction":    d,
            "entry_price":  entry,
            "sl":           sl,
            "tp":           tp,
            "ref_high":     r["ref_high"],
            "ref_low":      r["ref_low"],
            "atr":          r["atr"],
        })

    return out
