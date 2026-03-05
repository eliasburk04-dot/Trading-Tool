#!/usr/bin/env python3
"""
Deep analysis of Gold trades to understand win/loss patterns and design
a high-win-rate strategy variant.
"""
from __future__ import annotations
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import ASSETS, DEFAULT_PARAMS, OUTPUT_DIR, DATA_CACHE_DIR, TF_SECONDS
from src.data import get_asset_data, resample_ohlcv
from src.strategy import detect_signals, prepare_signals_list, compute_atr
from src.backtest import run_backtest
from src.metrics import compute_metrics, numeric_metrics
from copy import deepcopy

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

# ── Load Gold data ────────────────────────────────────────────────────────────
print("Loading Gold 1h data...")
asset_cfg = ASSETS["Gold"]
df, sym = get_asset_data("Gold", asset_cfg["symbols"], "1h", 729, DATA_CACHE_DIR)
print(f"  {sym}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

# ── Run baseline strategy and capture all trade details ───────────────────────
params = deepcopy(DEFAULT_PARAMS)
sig_df = detect_signals(df, params)
signals = prepare_signals_list(sig_df, params, TF_SECONDS["1h"])
trades = run_backtest(df, signals, params)

print(f"\n=== BASELINE: {len(trades)} trades, WR={trades['pnl_pct'].gt(0).mean():.1%} ===")

# ── Analysis 1: How far does price retrace into the reference candle? ─────────
print("\n" + "=" * 70)
print("  ANALYSIS 1: Retrace depth into reference candle")
print("=" * 70)
# For each signal, track what actually happens
sig_rows = sig_df[sig_df["signal"] != 0].copy()
print(f"Total signals (before backtest filtering): {len(sig_rows)}")

# ── Analysis 2: Win/Loss by candle characteristics ────────────────────────────
print("\n" + "=" * 70)
print("  ANALYSIS 2: Trade outcomes by candle properties")
print("=" * 70)

if not trades.empty:
    trades["won"] = trades["pnl_pct"] > 0
    trades["candle_range"] = trades["ref_high"] - trades["ref_low"]
    trades["hold_hours"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    
    # Compute ATR at signal time for each trade
    atr = compute_atr(df, 14)
    
    # Match each trade to its ATR
    atr_at_signal = []
    for _, t in trades.iterrows():
        sig_t = t["signal_time"]
        if sig_t in atr.index:
            atr_at_signal.append(atr.loc[sig_t])
        else:
            # find nearest
            idx = atr.index.get_indexer([sig_t], method="nearest")[0]
            atr_at_signal.append(atr.iloc[idx])
    trades["atr_at_signal"] = atr_at_signal
    trades["candle_range_atr"] = trades["candle_range"] / trades["atr_at_signal"]
    
    # Body ratio (how much of the candle is body vs wick)
    body = (trades["ref_high"] - trades["ref_low"])  # total range
    # We'd need ref_open/ref_close for body, but we don't have it in trades
    # Let's approximate from the signal data
    
    print(f"\nAll trades: {len(trades)}")
    print(f"  Winners: {trades['won'].sum()} ({trades['won'].mean():.1%})")
    print(f"  Losers:  {(~trades['won']).sum()} ({(~trades['won']).mean():.1%})")
    
    print(f"\n--- By exit type ---")
    for et, g in trades.groupby("exit_type"):
        print(f"  {et}: {len(g)} trades, WR={g['won'].mean():.1%}, avg_pnl={g['pnl_pct'].mean():.4f}")
    
    print(f"\n--- By direction ---")
    for d, g in trades.groupby("direction"):
        label = "LONG" if d == 1 else "SHORT"
        print(f"  {label}: {len(g)} trades, WR={g['won'].mean():.1%}, avg_pnl={g['pnl_pct'].mean():.4f}")
    
    print(f"\n--- Candle range (ATR multiples) vs win rate ---")
    trades["range_bucket"] = pd.cut(trades["candle_range_atr"], bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
    for bucket, g in trades.groupby("range_bucket", observed=True):
        if len(g) >= 5:
            print(f"  {bucket}: n={len(g)}, WR={g['won'].mean():.1%}, avg_pnl={g['pnl_pct'].mean():.4f}")
    
    print(f"\n--- Hold time distribution ---")
    trades["hold_bucket"] = pd.cut(trades["hold_hours"], bins=[0, 1, 3, 6, 12, 24, 48, 200])
    for bucket, g in trades.groupby("hold_bucket", observed=True):
        if len(g) >= 5:
            print(f"  {bucket}: n={len(g)}, WR={g['won'].mean():.1%}, avg_pnl={g['pnl_pct'].mean():.4f}")
    
    # ── Analysis 3: Hour-of-day effect ────────────────────────────────────
    print(f"\n--- Entry hour vs win rate ---")
    trades["entry_hour"] = trades["entry_time"].dt.hour
    hourly = trades.groupby("entry_hour").agg(
        n=("won", "count"),
        wr=("won", "mean"),
        avg_pnl=("pnl_pct", "mean")
    )
    for h, row in hourly.iterrows():
        if row["n"] >= 5:
            bar = "#" * int(row["wr"] * 40)
            print(f"  {h:2d}h: n={row['n']:3.0f}, WR={row['wr']:.1%} {bar}  pnl={row['avg_pnl']:.4f}")

    # ── Analysis 4: Trend filter effect ───────────────────────────────────
    print(f"\n" + "=" * 70)
    print("  ANALYSIS 3: Effect of trend filters")
    print("=" * 70)

    for ma_period in [20, 50, 100, 200]:
        ma = df["Close"].rolling(ma_period, min_periods=ma_period).mean()
        # For each trade, check if close > MA at signal time (uptrend)
        trend_at_signal = []
        for _, t in trades.iterrows():
            sig_t = t["signal_time"]
            if sig_t in ma.index:
                val = ma.loc[sig_t]
                close_val = df.loc[sig_t, "Close"] if sig_t in df.index else np.nan
                if np.isfinite(val) and np.isfinite(close_val):
                    trend_at_signal.append(1 if close_val > val else -1)
                else:
                    trend_at_signal.append(0)
            else:
                trend_at_signal.append(0)
        trades[f"trend_ma{ma_period}"] = trend_at_signal
        
        # Only take trades in trend direction
        with_trend = trades[
            ((trades["direction"] == 1) & (trades[f"trend_ma{ma_period}"] == 1)) |
            ((trades["direction"] == -1) & (trades[f"trend_ma{ma_period}"] == -1))
        ]
        against_trend = trades[
            ((trades["direction"] == 1) & (trades[f"trend_ma{ma_period}"] == -1)) |
            ((trades["direction"] == -1) & (trades[f"trend_ma{ma_period}"] == 1))
        ]
        print(f"\n  MA({ma_period}) trend filter:")
        print(f"    WITH trend:    n={len(with_trend):3d}, WR={with_trend['won'].mean():.1%}, avg_pnl={with_trend['pnl_pct'].mean():.4f}" if len(with_trend) > 0 else "    WITH trend: 0 trades")
        print(f"    AGAINST trend: n={len(against_trend):3d}, WR={against_trend['won'].mean():.1%}, avg_pnl={against_trend['pnl_pct'].mean():.4f}" if len(against_trend) > 0 else "    AGAINST: 0 trades")

    # ── Analysis 5: RR sweep (key to win rate) ────────────────────────────
    print(f"\n" + "=" * 70)
    print("  ANALYSIS 4: RR sweep (lower RR = higher win rate)")
    print("=" * 70)
    
    for rr_test in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]:
        p = deepcopy(DEFAULT_PARAMS)
        p["rr"] = rr_test
        signals_t = prepare_signals_list(sig_df, p, TF_SECONDS["1h"])
        trades_t = run_backtest(df, signals_t, p)
        if not trades_t.empty:
            wr = trades_t["pnl_pct"].gt(0).mean()
            total_ret = (1 + trades_t["pnl_pct"]).prod() - 1
            avg_pnl = trades_t["pnl_pct"].mean()
            n = len(trades_t)
            pf_num = trades_t[trades_t["pnl_pct"] > 0]["pnl_pct"].sum()
            pf_den = abs(trades_t[trades_t["pnl_pct"] <= 0]["pnl_pct"].sum())
            pf = pf_num / pf_den if pf_den > 0 else 999
            print(f"  RR={rr_test:.2f}: n={n:3d}, WR={wr:.1%}, PF={pf:.2f}, avg_pnl={avg_pnl:.5f}, total_ret={total_ret:+.1%}")

    # ── Analysis 6: Combined filters (trend + RR + candle quality) ────────
    print(f"\n" + "=" * 70)
    print("  ANALYSIS 5: Combined filter tests")
    print("=" * 70)
    
    # Test: trend filter + low RR + breakeven
    for rr_test in [0.75, 1.0, 1.25, 1.5]:
        for ma_p in [50, 100]:
            for use_be in [False, True]:
                p = deepcopy(DEFAULT_PARAMS)
                p["rr"] = rr_test
                p["use_breakeven"] = use_be
                p["breakeven_rr"] = 1.0
                # We need to add trend filter to signal generation
                # For now, simulate by filtering signals manually
                
                sig_df2 = detect_signals(df, p)
                ma = df["Close"].rolling(ma_p, min_periods=ma_p).mean()
                
                # Zero out signals against the trend
                uptrend = df["Close"] > ma
                downtrend = df["Close"] < ma
                bull_against = (sig_df2["signal"] == 1) & downtrend
                bear_against = (sig_df2["signal"] == -1) & uptrend
                sig_df2.loc[bull_against, "signal"] = 0
                sig_df2.loc[bear_against, "signal"] = 0
                
                signals_t = prepare_signals_list(sig_df2, p, TF_SECONDS["1h"])
                trades_t = run_backtest(df, signals_t, p)
                if not trades_t.empty and len(trades_t) >= 10:
                    wr = trades_t["pnl_pct"].gt(0).mean()
                    total_ret = (1 + trades_t["pnl_pct"]).prod() - 1
                    pf_num = trades_t[trades_t["pnl_pct"] > 0]["pnl_pct"].sum()
                    pf_den = abs(trades_t[trades_t["pnl_pct"] <= 0]["pnl_pct"].sum())
                    pf = pf_num / pf_den if pf_den > 0 else 999
                    n = len(trades_t)
                    be_tag = "+BE" if use_be else "   "
                    print(f"  RR={rr_test:.2f} MA({ma_p:3d}) {be_tag}: n={n:3d}, WR={wr:.1%}, PF={pf:.2f}, ret={total_ret:+.1%}")

    # ── Analysis 7: Multi-candle confirmation ─────────────────────────────
    print(f"\n" + "=" * 70)
    print("  ANALYSIS 6: Swing lookback impact on Gold")
    print("=" * 70)
    
    for swing in [5, 10, 15, 20, 30, 40, 50]:
        for rr_test in [1.0, 1.5, 2.0]:
            p = deepcopy(DEFAULT_PARAMS)
            p["swing_lookback"] = swing
            p["rr"] = rr_test
            signals_t = prepare_signals_list(detect_signals(df, p), p, TF_SECONDS["1h"])
            trades_t = run_backtest(df, signals_t, p)
            if not trades_t.empty and len(trades_t) >= 10:
                wr = trades_t["pnl_pct"].gt(0).mean()
                total_ret = (1 + trades_t["pnl_pct"]).prod() - 1
                n = len(trades_t)
                pf_num = trades_t[trades_t["pnl_pct"] > 0]["pnl_pct"].sum()
                pf_den = abs(trades_t[trades_t["pnl_pct"] <= 0]["pnl_pct"].sum())
                pf = pf_num / pf_den if pf_den > 0 else 999
                print(f"  N={swing:2d} RR={rr_test:.1f}: n={n:3d}, WR={wr:.1%}, PF={pf:.2f}, ret={total_ret:+.1%}")

    # ── Analysis 8: Higher timeframes ─────────────────────────────────────
    print(f"\n" + "=" * 70)
    print("  ANALYSIS 7: Higher setup timeframes")
    print("=" * 70)
    
    for setup_tf in ["4h", "1d"]:
        df_setup = resample_ohlcv(df, setup_tf)
        if len(df_setup) < 60:
            print(f"  {setup_tf}: not enough data")
            continue
        for rr_test in [0.75, 1.0, 1.5, 2.0]:
            for swing in [5, 10, 15, 20]:
                p = deepcopy(DEFAULT_PARAMS)
                p["rr"] = rr_test
                p["swing_lookback"] = swing
                sig_df_htf = detect_signals(df_setup, p)
                signals_htf = prepare_signals_list(sig_df_htf, p, TF_SECONDS.get(setup_tf, 3600))
                trades_htf = run_backtest(df, signals_htf, p)
                if not trades_htf.empty and len(trades_htf) >= 5:
                    wr = trades_htf["pnl_pct"].gt(0).mean()
                    total_ret = (1 + trades_htf["pnl_pct"]).prod() - 1
                    n = len(trades_htf)
                    pf_num = trades_htf[trades_htf["pnl_pct"] > 0]["pnl_pct"].sum()
                    pf_den = abs(trades_htf[trades_htf["pnl_pct"] <= 0]["pnl_pct"].sum())
                    pf = pf_num / pf_den if pf_den > 0 else 999
                    print(f"  {setup_tf} N={swing:2d} RR={rr_test:.2f}: n={n:3d}, WR={wr:.1%}, PF={pf:.2f}, ret={total_ret:+.1%}")

print("\n\nAnalysis complete.")
