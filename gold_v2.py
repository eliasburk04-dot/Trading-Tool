#!/usr/bin/env python3
"""
gold_v2.py - Enhanced One-Candle SMC Strategy for Gold (XAUUSD/GC=F)
=====================================================================
Goal: Push win rate to 70%+ while maintaining positive expectancy.

Key improvements over v1:
  1. Lower RR (0.5 - 1.5) → mechanically higher win rate
  2. Wider SL buffer → fewer premature stop-outs
  3. EMA trend filter → only trade with-trend
  4. Impulse candle body filter → only strong candles
  5. Session filter → only trade during active hours
  6. Higher setup TF option (4h) → more meaningful MSS signals
  7. Break-even at 0.5R → protect winners
  8. ATR-based SL (dynamic) instead of fixed candle low/high
  9. Cooldown between signals → avoid overtrading in chop
"""
from __future__ import annotations
import sys, os, time, itertools, warnings
from copy import deepcopy

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

# ── Project imports ───────────────────────────────────────────────────────────
from config import OUTPUT_DIR, DATA_CACHE_DIR, ASSETS
from src.data import get_asset_data, resample_ohlcv

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  V2 STRATEGY PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
V2_DEFAULT = {
    # Structure detection
    "swing_lookback":      10,
    "setup_tf":            "1h",      # setup timeframe
    "entry_tf":            "1h",      # entry/execution timeframe

    # Entry
    "entry_retrace_pct":   0.5,       # 50% retrace of reference candle
    "entry_validity":      5,         # bars to wait for fill

    # Risk management
    "rr":                  1.0,       # LOWER RR = higher win rate
    "sl_mode":             "candle_atr",  # "candle" | "atr" | "candle_atr"
    "sl_buffer_atr":       0.3,       # wider buffer (was 0.1)
    "sl_atr_mult":         1.5,       # for ATR-only SL mode
    "atr_period":          14,

    # Filters (THE KEY TO HIGH WIN RATE)
    "use_trend_filter":    True,
    "trend_ema_period":    50,        # EMA period for trend
    "trend_ema_fast":      20,        # fast EMA (optional dual-EMA filter)
    "use_dual_ema":        False,     # require fast > slow for longs

    "use_body_filter":     True,
    "min_body_pct":        0.5,       # body must be >= 50% of candle range

    "use_session_filter":  False,     # restrict to active session hours
    "session_hours":       (6, 20),   # UTC hours (London open to NY close)

    "use_momentum_filter": True,
    "momentum_period":     5,         # RSI-like momentum lookback
    "momentum_threshold":  0.6,       # % of recent bars must be in direction

    "use_atr_filter":      True,
    "min_atr_pct":         0.3,       # candle ATR must be > this pctile of recent ATRs

    "cooldown_bars":       3,         # min bars between signals

    # Trade Management
    "use_breakeven":       True,
    "breakeven_rr":        0.5,       # move to BE at this R multiple
    "use_trailing":        False,
    "trailing_atr_mult":   2.0,

    # Costs
    "slippage_pct":        0.01,
    "fee_pct":             0.02,
}


# ═════════════════════════════════════════════════════════════════════════════
#  ATR COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ═════════════════════════════════════════════════════════════════════════════
#  V2 SIGNAL DETECTION (with all filters)
# ═════════════════════════════════════════════════════════════════════════════
def detect_signals_v2(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Enhanced signal detection with trend, body, session, momentum filters."""
    N = int(params["swing_lookback"])
    rr = float(params["rr"])

    # ── Swing levels ──────────────────────────────────────────────────────
    swing_high = df["High"].rolling(N, min_periods=N).max().shift(1)
    swing_low  = df["Low"].rolling(N, min_periods=N).min().shift(1)

    # ── MSS ───────────────────────────────────────────────────────────────
    bull_mss = df["Close"] > swing_high
    bear_mss = df["Close"] < swing_low

    # resolve double-fire
    both = bull_mss & bear_mss
    candle_up = df["Close"] >= df["Open"]
    bull_mss = bull_mss & (~both | candle_up)
    bear_mss = bear_mss & (~both | ~candle_up)

    # ── FILTER 1: Trend (EMA) ────────────────────────────────────────────
    if params.get("use_trend_filter", False):
        ema_period = int(params.get("trend_ema_period", 50))
        ema = df["Close"].ewm(span=ema_period, adjust=False).mean()

        if params.get("use_dual_ema", False):
            ema_fast = df["Close"].ewm(span=int(params.get("trend_ema_fast", 20)), adjust=False).mean()
            uptrend = (df["Close"] > ema) & (ema_fast > ema)
            downtrend = (df["Close"] < ema) & (ema_fast < ema)
        else:
            uptrend = df["Close"] > ema
            downtrend = df["Close"] < ema

        bull_mss = bull_mss & uptrend
        bear_mss = bear_mss & downtrend

    # ── FILTER 2: Body quality ────────────────────────────────────────────
    if params.get("use_body_filter", False):
        min_body = params.get("min_body_pct", 0.5)
        candle_range = df["High"] - df["Low"]
        body_size = (df["Close"] - df["Open"]).abs()
        body_pct = body_size / candle_range.replace(0, np.nan)
        good_body = body_pct >= min_body

        # Also require body direction matches signal direction
        bull_body = df["Close"] > df["Open"]  # green candle for bull
        bear_body = df["Close"] < df["Open"]  # red candle for bear

        bull_mss = bull_mss & good_body & bull_body
        bear_mss = bear_mss & good_body & bear_body

    # ── FILTER 3: Session hours ───────────────────────────────────────────
    if params.get("use_session_filter", False):
        h_start, h_end = params.get("session_hours", (6, 20))
        in_session = df.index.hour.to_series(index=df.index)
        if h_start < h_end:
            in_session = (in_session >= h_start) & (in_session < h_end)
        else:
            in_session = (in_session >= h_start) | (in_session < h_end)
        bull_mss = bull_mss & in_session
        bear_mss = bear_mss & in_session

    # ── FILTER 4: Momentum confirmation ───────────────────────────────────
    if params.get("use_momentum_filter", False):
        mom_period = int(params.get("momentum_period", 5))
        threshold = params.get("momentum_threshold", 0.6)

        bull_bars = (df["Close"] > df["Open"]).rolling(mom_period).sum() / mom_period
        bear_bars = (df["Close"] < df["Open"]).rolling(mom_period).sum() / mom_period

        bull_mss = bull_mss & (bull_bars >= threshold)
        bear_mss = bear_mss & (bear_bars >= threshold)

    # ── FILTER 5: ATR magnitude ───────────────────────────────────────────
    atr = compute_atr(df, int(params.get("atr_period", 14)))
    if params.get("use_atr_filter", False):
        min_atr_pct = params.get("min_atr_pct", 0.3)
        candle_range = df["High"] - df["Low"]
        atr_ratio = candle_range / atr
        bull_mss = bull_mss & (atr_ratio >= min_atr_pct)
        bear_mss = bear_mss & (atr_ratio >= min_atr_pct)

    # ── Build signal frame ────────────────────────────────────────────────
    sig = pd.DataFrame(index=df.index)
    sig["signal"] = 0
    sig.loc[bull_mss, "signal"] = 1
    sig.loc[bear_mss, "signal"] = -1

    # ── FILTER 6: Cooldown ────────────────────────────────────────────────
    cooldown = int(params.get("cooldown_bars", 0))
    if cooldown > 0:
        last_signal_idx = -cooldown - 1
        for i in range(len(sig)):
            if sig.iloc[i, sig.columns.get_loc("signal")] != 0:
                if i - last_signal_idx <= cooldown:
                    sig.iloc[i, sig.columns.get_loc("signal")] = 0
                else:
                    last_signal_idx = i

    # Reference candle data
    sig["ref_high"]  = df["High"]
    sig["ref_low"]   = df["Low"]
    sig["ref_open"]  = df["Open"]
    sig["ref_close"] = df["Close"]
    sig["atr"]       = atr

    # Entry price
    retrace = params.get("entry_retrace_pct", 0.5)
    sig["entry_price"] = df["Low"] + retrace * (df["High"] - df["Low"])

    # SL computation (enhanced)
    sl_mode = params.get("sl_mode", "candle_atr")
    buf = params.get("sl_buffer_atr", 0.3) * atr

    if sl_mode == "atr":
        sl_mult = params.get("sl_atr_mult", 1.5)
        sig["sl_long"]  = sig["entry_price"] - sl_mult * atr
        sig["sl_short"] = sig["entry_price"] + sl_mult * atr
    elif sl_mode == "candle_atr":
        sig["sl_long"]  = df["Low"] - buf
        sig["sl_short"] = df["High"] + buf
    else:  # "candle"
        sig["sl_long"]  = df["Low"]
        sig["sl_short"] = df["High"]

    # TP (fixed RR)
    risk_long  = sig["entry_price"] - sig["sl_long"]
    risk_short = sig["sl_short"] - sig["entry_price"]
    sig["tp_long"]  = sig["entry_price"] + rr * risk_long
    sig["tp_short"] = sig["entry_price"] - rr * risk_short

    # Liquidity targets
    wide = int(max(3 * N, 30))
    sig["tp_liq_long"]  = df["High"].shift(N).rolling(wide, min_periods=N).max()
    sig["tp_liq_short"] = df["Low"].shift(N).rolling(wide, min_periods=N).min()

    sig["swing_high"] = swing_high
    sig["swing_low"]  = swing_low

    return sig


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL LIST BUILDER
# ═════════════════════════════════════════════════════════════════════════════
def prepare_signals_v2(sig_df: pd.DataFrame, params: dict, tf_seconds: int) -> list[dict]:
    tp_mode = params.get("tp_mode", "fixed_rr")
    mask = sig_df["signal"] != 0
    rows = sig_df.loc[mask]
    out = []
    for idx, r in rows.iterrows():
        d = int(r["signal"])
        entry = r["entry_price"]
        sl = r["sl_long"] if d == 1 else r["sl_short"]
        tp = r["tp_long"] if d == 1 else r["tp_short"]

        if tp_mode == "liquidity_target":
            tp_liq = r["tp_liq_long"] if d == 1 else r["tp_liq_short"]
            if np.isfinite(tp_liq):
                valid = (d == 1 and tp_liq > entry) or (d == -1 and tp_liq < entry)
                if valid:
                    tp = tp_liq

        if d == 1 and (tp <= entry or sl >= entry): continue
        if d == -1 and (tp >= entry or sl <= entry): continue
        if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp)): continue

        out.append({
            "signal_time": idx,
            "entry_after": idx + pd.Timedelta(seconds=tf_seconds),
            "direction": d, "entry_price": entry,
            "sl": sl, "tp": tp,
            "ref_high": r["ref_high"], "ref_low": r["ref_low"],
            "atr": r["atr"],
        })
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTESTER (reused from src with minor tweaks)
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest_v2(entry_data: pd.DataFrame, signals_list: list[dict], params: dict) -> pd.DataFrame:
    K = int(params.get("entry_validity", 5))
    use_be = params.get("use_breakeven", False)
    be_rr = float(params.get("breakeven_rr", 0.5))
    use_trail = params.get("use_trailing", False)
    trail_mult = float(params.get("trailing_atr_mult", 2.0))
    slip_frac = float(params.get("slippage_pct", 0.0)) / 100.0
    fee_frac = float(params.get("fee_pct", 0.0)) / 100.0

    trades = []
    pending = None
    position = None
    sig_ptr = 0
    n_sig = len(signals_list)

    times  = entry_data.index.values
    opens  = entry_data["Open"].values.astype(np.float64)
    highs  = entry_data["High"].values.astype(np.float64)
    lows   = entry_data["Low"].values.astype(np.float64)
    closes = entry_data["Close"].values.astype(np.float64)
    n_bars = len(entry_data)

    for i in range(n_bars):
        t = times[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # 1. Manage open position
        if position is not None:
            d = position["direction"]
            sl_cur = position["sl"]
            tp_cur = position["tp"]
            exit_price = None
            exit_type = None

            if d == 1:
                sl_hit = l <= sl_cur
                tp_hit = h >= tp_cur
                if sl_hit and tp_hit:
                    exit_price, exit_type = sl_cur, "sl"
                elif sl_hit:
                    exit_price = min(sl_cur, o) if o < sl_cur else sl_cur
                    exit_type = "sl"
                elif tp_hit:
                    exit_price = max(tp_cur, o) if o > tp_cur else tp_cur
                    exit_type = "tp"
            else:
                sl_hit = h >= sl_cur
                tp_hit = l <= tp_cur
                if sl_hit and tp_hit:
                    exit_price, exit_type = sl_cur, "sl"
                elif sl_hit:
                    exit_price = max(sl_cur, o) if o > sl_cur else sl_cur
                    exit_type = "sl"
                elif tp_hit:
                    exit_price = min(tp_cur, o) if o < tp_cur else tp_cur
                    exit_type = "tp"

            if exit_type is not None:
                if d == 1:
                    exit_price *= (1 - slip_frac)
                else:
                    exit_price *= (1 + slip_frac)
                pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] - fee_frac
                trades.append({
                    "signal_time": position["signal_time"],
                    "entry_time": position["entry_time"],
                    "exit_time": pd.Timestamp(t),
                    "direction": d,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "sl": position["orig_sl"],
                    "tp": position["orig_tp"],
                    "pnl_pct": pnl_pct,
                    "exit_type": exit_type,
                    "ref_high": position["ref_high"],
                    "ref_low": position["ref_low"],
                })
                position = None
            else:
                # Break-even
                if use_be:
                    ep = position["entry_price"]
                    risk0 = abs(ep - position["orig_sl"])
                    if d == 1:
                        if h >= ep + be_rr * risk0 and position["sl"] < ep:
                            position["sl"] = ep
                    else:
                        if l <= ep - be_rr * risk0 and position["sl"] > ep:
                            position["sl"] = ep
                # Trailing stop
                if use_trail:
                    atr_val = position["atr"]
                    if d == 1:
                        trail = h - trail_mult * atr_val
                        if trail > position["sl"]:
                            position["sl"] = trail
                    else:
                        trail = l + trail_mult * atr_val
                        if trail < position["sl"]:
                            position["sl"] = trail

        # 2. Check pending limit
        if pending is not None:
            if i > pending["expiry_idx"]:
                pending = None
            else:
                ep = pending["entry_price"]
                d = pending["direction"]
                filled = (d == 1 and l <= ep) or (d == -1 and h >= ep)
                if filled:
                    actual_entry = ep * (1 + slip_frac) if d == 1 else ep * (1 - slip_frac)
                    position = {
                        "direction": d, "entry_price": actual_entry,
                        "sl": pending["sl"], "tp": pending["tp"],
                        "orig_sl": pending["sl"], "orig_tp": pending["tp"],
                        "signal_time": pending["signal_time"],
                        "entry_time": pd.Timestamp(t),
                        "ref_high": pending["ref_high"],
                        "ref_low": pending["ref_low"],
                        "atr": pending["atr"],
                    }
                    pending = None

        # 3. Consume signals
        while sig_ptr < n_sig and np.datetime64(signals_list[sig_ptr]["entry_after"]) <= t:
            if pending is None and position is None:
                sig = signals_list[sig_ptr]
                pending = {
                    "direction": sig["direction"],
                    "entry_price": sig["entry_price"],
                    "sl": sig["sl"], "tp": sig["tp"],
                    "signal_time": sig["signal_time"],
                    "expiry_idx": i + K,
                    "ref_high": sig["ref_high"],
                    "ref_low": sig["ref_low"],
                    "atr": sig["atr"],
                }
            sig_ptr += 1

    # Close remaining position
    if position is not None:
        d = position["direction"]
        exit_price = closes[-1]
        pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] - fee_frac
        trades.append({
            "signal_time": position["signal_time"],
            "entry_time": position["entry_time"],
            "exit_time": pd.Timestamp(times[-1]),
            "direction": d, "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "sl": position["orig_sl"], "tp": position["orig_tp"],
            "pnl_pct": pnl_pct, "exit_type": "end",
            "ref_high": position["ref_high"], "ref_low": position["ref_low"],
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics_v2(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "wr": 0, "pf": 0, "sharpe": -99, "total_ret": 0,
                "max_dd": 0, "avg_pnl": 0, "expectancy": 0}
    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    wr = len(wins) / n
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0
    pf_den = abs(losses["pnl_pct"].sum()) if len(losses) else 1e-10
    pf = wins["pnl_pct"].sum() / pf_den if pf_den > 0 else 999

    eq = [10000]
    for p in trades["pnl_pct"].values:
        eq.append(eq[-1] * (1 + p))
    eq = np.array(eq)
    total_ret = eq[-1] / eq[0] - 1
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # daily sharpe
    times_idx = [trades["entry_time"].iloc[0]] + list(trades["exit_time"])
    eq_s = pd.Series(eq, index=times_idx)
    daily = eq_s.resample("B").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0

    return {
        "n": n, "wr": wr, "pf": pf, "sharpe": float(sharpe),
        "total_ret": total_ret, "max_dd": max_dd,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_pnl": trades["pnl_pct"].mean(),
        "expectancy": wr * avg_win + (1 - wr) * avg_loss,
    }


def print_metrics(label: str, m: dict):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:       {m['n']}")
    print(f"  Win Rate:     {m['wr']:.1%}")
    print(f"  Profit Factor:{m['pf']:.2f}")
    print(f"  Sharpe:       {m['sharpe']:.2f}")
    print(f"  Total Return: {m['total_ret']:+.1%}")
    print(f"  Max Drawdown: {m['max_dd']:.1%}")
    print(f"  Avg Win:      {m['avg_win']:.5f}")
    print(f"  Avg Loss:     {m['avg_loss']:.5f}")
    print(f"  Expectancy:   {m['expectancy']:.5f}")


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE: run one config
# ═════════════════════════════════════════════════════════════════════════════
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
              "1h": 3600, "4h": 14400, "1d": 86400}

def run_one(df_entry: pd.DataFrame, params: dict, setup_tf: str = "1h") -> tuple[pd.DataFrame, dict]:
    if setup_tf != params.get("entry_tf", "1h"):
        df_setup = resample_ohlcv(df_entry, setup_tf)
    else:
        df_setup = df_entry
    sig = detect_signals_v2(df_setup, params)
    signals = prepare_signals_v2(sig, params, TF_SECONDS.get(setup_tf, 3600))
    trades = run_backtest_v2(df_entry, signals, params)
    m = compute_metrics_v2(trades)
    return trades, m


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
WIN_CLR = "#26a69a"
LOSS_CLR = "#ef5350"

def plot_trades_v2(df, trades, title, filepath):
    if trades.empty: return
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(df.index, df["Close"], color="#455a64", lw=0.5, alpha=0.8, label="Close")
    for _, r in trades.iterrows():
        won = r["pnl_pct"] > 0
        clr = WIN_CLR if won else LOSS_CLR
        mk = "^" if r["direction"] == 1 else "v"
        if r["entry_time"] >= df.index[0]:
            ax.scatter(r["entry_time"], r["entry_price"], marker=mk, color=clr,
                       s=60, zorder=5, edgecolors="k", linewidths=0.3)
        if r["exit_time"] >= df.index[0]:
            ax.scatter(r["exit_time"], r["exit_price"], marker="x", color=clr,
                       s=40, zorder=5, linewidths=1.0)
        if r["entry_time"] >= df.index[0] and r["exit_time"] >= df.index[0]:
            ax.plot([r["entry_time"], r["exit_time"]], [r["sl"], r["sl"]],
                    color=LOSS_CLR, lw=0.3, ls="--", alpha=0.4)
            ax.plot([r["entry_time"], r["exit_time"]], [r["tp"], r["tp"]],
                    color=WIN_CLR, lw=0.3, ls="--", alpha=0.4)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

def plot_equity_v2(trades, title, filepath):
    if trades.empty: return
    eq = [10000]
    for p in trades["pnl_pct"].values:
        eq.append(eq[-1] * (1 + p))
    times = [trades["entry_time"].iloc[0]] + list(trades["exit_time"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(times, eq, color="#1565c0", lw=1.2)
    ax1.fill_between(times, 10000, eq, alpha=0.1, color="#1565c0")
    ax1.axhline(10000, color="grey", lw=0.5, ls="--")
    ax1.set_title(title, fontweight="bold")
    ax1.set_ylabel("Equity")

    peak = np.maximum.accumulate(eq)
    dd = (np.array(eq) - peak) / peak * 100
    ax2.fill_between(times, dd, 0, color=LOSS_CLR, alpha=0.35)
    ax2.plot(times, dd, color=LOSS_CLR, lw=0.7)
    ax2.set_ylabel("Drawdown %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

def plot_heatmap(results_df, row_param, col_param, value_col, title, filepath):
    if results_df.empty: return
    piv = results_df.pivot_table(index=row_param, columns=col_param, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, len(piv.columns)*1.3), max(4, len(piv.index)*0.8)))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v}" for v in piv.index])
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(title, fontweight="bold")
    for yi in range(piv.shape[0]):
        for xi in range(piv.shape[1]):
            v = piv.values[yi, xi]
            if np.isfinite(v):
                ax.text(xi, yi, f"{v:.1%}" if abs(v) < 10 else f"{v:.2f}",
                        ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 70)
    print("  GOLD V2 - Enhanced High-Win-Rate SMC Strategy")
    print("=" * 70)

    # Load Gold data
    asset_cfg = ASSETS["Gold"]
    df, sym = get_asset_data("Gold", asset_cfg["symbols"], "1h", 729, DATA_CACHE_DIR)
    if df is None:
        print("ERROR: Could not load Gold data")
        return
    print(f"  Data: {sym}, {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    out_dir = os.path.join(OUTPUT_DIR, "gold_v2")
    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Massive parameter sweep to find high-WR configurations
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 1: Parameter Sweep (finding high win-rate configs)")
    print("=" * 70)

    sweep_grid = {
        "rr":               [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "swing_lookback":   [5, 10, 15, 20, 30],
        "sl_buffer_atr":    [0.1, 0.3, 0.5, 0.8],
        "use_trend_filter": [True, False],
        "use_body_filter":  [True, False],
        "use_breakeven":    [True, False],
    }

    keys = sorted(sweep_grid.keys())
    combos = list(itertools.product(*(sweep_grid[k] for k in keys)))
    total = len(combos)
    print(f"  Total combinations: {total}")

    rows = []
    for idx, vals in enumerate(combos, 1):
        p = deepcopy(V2_DEFAULT)
        for k, v in zip(keys, vals):
            p[k] = v
        _, m = run_one(df, p, p.get("setup_tf", "1h"))
        row = {k: v for k, v in zip(keys, vals)}
        row.update(m)
        rows.append(row)
        if idx % 100 == 0 or idx == total:
            print(f"    sweep {idx}/{total} ...")

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(out_dir, "sweep_full.csv"), index=False)

    # ── Show best by win rate ─────────────────────────────────────────────
    viable = sweep_df[sweep_df["n"] >= 20].copy()
    print(f"\n  Viable configs (>= 20 trades): {len(viable)}")

    print("\n  --- TOP 20 by Win Rate ---")
    top_wr = viable.nlargest(20, "wr")
    for _, r in top_wr.iterrows():
        filters = []
        if r.get("use_trend_filter"): filters.append("TREND")
        if r.get("use_body_filter"): filters.append("BODY")
        if r.get("use_breakeven"): filters.append("BE")
        f_str = "+".join(filters) if filters else "none"
        print(f"    RR={r['rr']:.2f} N={int(r['swing_lookback']):2d} SLbuf={r['sl_buffer_atr']:.1f} "
              f"[{f_str:15s}] => WR={r['wr']:.1%} n={int(r['n']):3d} PF={r['pf']:.2f} "
              f"ret={r['total_ret']:+.1%} Sharpe={r['sharpe']:.2f}")

    print("\n  --- TOP 20 by Sharpe (WR >= 50%) ---")
    high_wr = viable[viable["wr"] >= 0.50]
    if len(high_wr) > 0:
        top_sharpe = high_wr.nlargest(20, "sharpe")
        for _, r in top_sharpe.iterrows():
            filters = []
            if r.get("use_trend_filter"): filters.append("TREND")
            if r.get("use_body_filter"): filters.append("BODY")
            if r.get("use_breakeven"): filters.append("BE")
            f_str = "+".join(filters) if filters else "none"
            print(f"    RR={r['rr']:.2f} N={int(r['swing_lookback']):2d} SLbuf={r['sl_buffer_atr']:.1f} "
                  f"[{f_str:15s}] => WR={r['wr']:.1%} n={int(r['n']):3d} PF={r['pf']:.2f} "
                  f"ret={r['total_ret']:+.1%} Sharpe={r['sharpe']:.2f}")
    else:
        print("    No configs with WR >= 50%")

    print("\n  --- TOP 20 by Profit Factor (WR >= 40%) ---")
    med_wr = viable[viable["wr"] >= 0.40]
    if len(med_wr) > 0:
        top_pf = med_wr.nlargest(20, "pf")
        for _, r in top_pf.iterrows():
            filters = []
            if r.get("use_trend_filter"): filters.append("TREND")
            if r.get("use_body_filter"): filters.append("BODY")
            if r.get("use_breakeven"): filters.append("BE")
            f_str = "+".join(filters) if filters else "none"
            print(f"    RR={r['rr']:.2f} N={int(r['swing_lookback']):2d} SLbuf={r['sl_buffer_atr']:.1f} "
                  f"[{f_str:15s}] => WR={r['wr']:.1%} n={int(r['n']):3d} PF={r['pf']:.2f} "
                  f"ret={r['total_ret']:+.1%} Sharpe={r['sharpe']:.2f}")

    # ── Heatmaps ──────────────────────────────────────────────────────────
    plot_heatmap(viable, "swing_lookback", "rr", "wr",
                 "Win Rate: Swing Lookback vs RR", os.path.join(out_dir, "heatmap_wr_swing_rr.png"))
    plot_heatmap(viable, "sl_buffer_atr", "rr", "wr",
                 "Win Rate: SL Buffer vs RR", os.path.join(out_dir, "heatmap_wr_sl_rr.png"))
    plot_heatmap(viable, "swing_lookback", "rr", "sharpe",
                 "Sharpe: Swing Lookback vs RR", os.path.join(out_dir, "heatmap_sharpe_swing_rr.png"))
    plot_heatmap(viable, "swing_lookback", "rr", "total_ret",
                 "Total Return: Swing Lookback vs RR", os.path.join(out_dir, "heatmap_ret_swing_rr.png"))
    # filter impact
    filter_impact = viable.groupby(["use_trend_filter", "use_body_filter", "use_breakeven"]).agg(
        avg_wr=("wr", "mean"), avg_pf=("pf", "mean"), avg_sharpe=("sharpe", "mean"),
        avg_ret=("total_ret", "mean"), count=("n", "count")
    ).reset_index()
    filter_impact.to_csv(os.path.join(out_dir, "filter_impact.csv"), index=False)
    print("\n  --- Filter Impact (averages) ---")
    print(filter_impact.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Extended sweep with more filters on best candidates
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  PHASE 2: Deep sweep on promising configs")
    print("=" * 70)

    deep_grid = {
        "rr":                [0.5, 0.6, 0.75, 0.8, 1.0],
        "swing_lookback":    [15, 20, 25, 30, 40],
        "sl_buffer_atr":     [0.3, 0.5, 0.7, 1.0],
        "trend_ema_period":  [20, 50, 100],
        "min_body_pct":      [0.3, 0.5, 0.6, 0.7],
        "cooldown_bars":     [0, 3, 5],
        "breakeven_rr":      [0.3, 0.5, 0.7],
    }

    # Fixed filters ON for deep sweep
    deep_base = deepcopy(V2_DEFAULT)
    deep_base["use_trend_filter"] = True
    deep_base["use_body_filter"] = True
    deep_base["use_breakeven"] = True
    deep_base["use_momentum_filter"] = False  # try off initially
    deep_base["use_atr_filter"] = False

    keys2 = sorted(deep_grid.keys())
    combos2 = list(itertools.product(*(deep_grid[k] for k in keys2)))
    total2 = len(combos2)
    print(f"  Total deep combinations: {total2}")

    rows2 = []
    for idx, vals in enumerate(combos2, 1):
        p = deepcopy(deep_base)
        for k, v in zip(keys2, vals):
            p[k] = v
        _, m = run_one(df, p)
        row = {k: v for k, v in zip(keys2, vals)}
        row.update(m)
        rows2.append(row)
        if idx % 200 == 0 or idx == total2:
            print(f"    deep sweep {idx}/{total2} ...")

    deep_df = pd.DataFrame(rows2)
    deep_df.to_csv(os.path.join(out_dir, "sweep_deep.csv"), index=False)

    viable2 = deep_df[deep_df["n"] >= 15].copy()

    print("\n  --- TOP 30 by Win Rate (deep sweep) ---")
    top30 = viable2.nlargest(30, "wr")
    for _, r in top30.iterrows():
        print(f"    RR={r['rr']:.2f} N={int(r['swing_lookback']):2d} SL={r['sl_buffer_atr']:.1f} "
              f"EMA={int(r['trend_ema_period']):3d} body={r['min_body_pct']:.1f} "
              f"cd={int(r['cooldown_bars'])} BE_rr={r['breakeven_rr']:.1f} "
              f"=> WR={r['wr']:.1%} n={int(r['n']):3d} PF={r['pf']:.2f} "
              f"ret={r['total_ret']:+.1%} Sharpe={r['sharpe']:.2f}")

    print("\n  --- TOP 30 by Profit Factor (deep, WR >= 50%) ---")
    hw2 = viable2[viable2["wr"] >= 0.50]
    if len(hw2) > 0:
        for _, r in hw2.nlargest(30, "pf").iterrows():
            print(f"    RR={r['rr']:.2f} N={int(r['swing_lookback']):2d} SL={r['sl_buffer_atr']:.1f} "
                  f"EMA={int(r['trend_ema_period']):3d} body={r['min_body_pct']:.1f} "
                  f"cd={int(r['cooldown_bars'])} BE_rr={r['breakeven_rr']:.1f} "
                  f"=> WR={r['wr']:.1%} n={int(r['n']):3d} PF={r['pf']:.2f} "
                  f"ret={r['total_ret']:+.1%} Sharpe={r['sharpe']:.2f}")

    # Heatmaps for deep sweep
    plot_heatmap(viable2, "swing_lookback", "rr", "wr",
                 "Deep: Win Rate (Swing vs RR)", os.path.join(out_dir, "deep_heatmap_wr.png"))
    plot_heatmap(viable2, "sl_buffer_atr", "rr", "wr",
                 "Deep: Win Rate (SL Buffer vs RR)", os.path.join(out_dir, "deep_heatmap_wr_sl.png"))
    plot_heatmap(viable2, "trend_ema_period", "rr", "wr",
                 "Deep: Win Rate (EMA vs RR)", os.path.join(out_dir, "deep_heatmap_wr_ema.png"))

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: Pick best config and generate full results
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  PHASE 3: Best Configuration - Full Analysis")
    print("=" * 70)

    # Pick best by composite score: prioritize WR >= 60%, then Sharpe
    candidates = viable2[viable2["wr"] >= 0.55].copy() if len(viable2[viable2["wr"] >= 0.55]) >= 3 else viable2.nlargest(10, "wr")
    if len(candidates) > 0:
        best_row = candidates.nlargest(1, "pf").iloc[0]
    else:
        best_row = viable2.nlargest(1, "wr").iloc[0]

    best_params = deepcopy(deep_base)
    for k in keys2:
        best_params[k] = best_row[k]

    print(f"\n  BEST CONFIG:")
    for k in keys2:
        print(f"    {k}: {best_params[k]}")

    trades_best, m_best = run_one(df, best_params)
    print_metrics("BEST CONFIG - Gold", m_best)

    if not trades_best.empty:
        trades_best.to_csv(os.path.join(out_dir, "best_trades.csv"), index=False)
        plot_trades_v2(df, trades_best, f"Gold V2 - Best Config (WR={m_best['wr']:.0%})",
                       os.path.join(out_dir, "best_trades_chart.png"))
        plot_equity_v2(trades_best, f"Gold V2 - Equity + Drawdown (WR={m_best['wr']:.0%})",
                       os.path.join(out_dir, "best_equity.png"))

    # ── Walk-forward on best ──────────────────────────────────────────────
    print("\n  Walk-Forward Validation...")
    n_bars = len(df)
    split = int(n_bars * 0.67)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    trades_train, m_train = run_one(df_train, best_params)
    trades_test, m_test = run_one(df_test, best_params)
    print_metrics("Walk-Forward TRAIN", m_train)
    print_metrics("Walk-Forward TEST", m_test)

    if not trades_test.empty:
        plot_trades_v2(df_test, trades_test, f"Gold V2 - Walk-Forward TEST (WR={m_test['wr']:.0%})",
                       os.path.join(out_dir, "wf_test_trades.png"))
        plot_equity_v2(trades_test, f"Gold V2 - WF TEST Equity (WR={m_test['wr']:.0%})",
                       os.path.join(out_dir, "wf_test_equity.png"))

    # ── Also test with 4h setup TF ───────────────────────────────────────
    print("\n\n  --- 4h Setup TF Results ---")
    params_4h = deepcopy(best_params)
    params_4h["setup_tf"] = "4h"
    trades_4h, m_4h = run_one(df, params_4h, setup_tf="4h")
    print_metrics("4h Setup TF", m_4h)

    if not trades_4h.empty:
        plot_trades_v2(df, trades_4h, f"Gold V2 - 4h Setup (WR={m_4h['wr']:.0%})",
                       os.path.join(out_dir, "4h_trades_chart.png"))
        plot_equity_v2(trades_4h, f"Gold V2 - 4h Setup Equity (WR={m_4h['wr']:.0%})",
                       os.path.join(out_dir, "4h_equity.png"))

    # ── Also try daily setup ──────────────────────────────────────────────
    print("\n  --- Daily Setup TF Results ---")
    params_1d = deepcopy(best_params)
    params_1d["setup_tf"] = "1d"
    trades_1d, m_1d = run_one(df, params_1d, setup_tf="1d")
    print_metrics("Daily Setup TF", m_1d)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4: Additional filter combos (momentum, session, dual EMA)
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  PHASE 4: Additional Filters on Best Config")
    print("=" * 70)

    variants = [
        ("+ Momentum filter", {"use_momentum_filter": True, "momentum_period": 5, "momentum_threshold": 0.6}),
        ("+ Momentum (loose)", {"use_momentum_filter": True, "momentum_period": 5, "momentum_threshold": 0.4}),
        ("+ Session filter", {"use_session_filter": True, "session_hours": (6, 20)}),
        ("+ ATR filter", {"use_atr_filter": True, "min_atr_pct": 0.5}),
        ("+ Dual EMA", {"use_dual_ema": True, "trend_ema_fast": 20}),
        ("+ All filters", {"use_momentum_filter": True, "momentum_threshold": 0.5,
                           "use_session_filter": True, "use_atr_filter": True, "min_atr_pct": 0.5}),
    ]

    for name, overrides in variants:
        p = deepcopy(best_params)
        p.update(overrides)
        trades_v, m_v = run_one(df, p)
        wr_str = f"{m_v['wr']:.1%}" if m_v['n'] > 0 else "N/A"
        print(f"  {name:25s}: n={m_v['n']:3d} WR={wr_str:6s} PF={m_v['pf']:.2f} "
              f"ret={m_v['total_ret']:+.1%} Sharpe={m_v['sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 70}")
    print(f"  DONE in {elapsed:.0f}s.  Output -> {os.path.abspath(out_dir)}")
    print(f"{'=' * 70}")
    print(f"\n  BEST CONFIGURATION SUMMARY:")
    print(f"    Win Rate:      {m_best['wr']:.1%}")
    print(f"    Profit Factor: {m_best['pf']:.2f}")
    print(f"    Total Return:  {m_best['total_ret']:+.1%}")
    print(f"    Sharpe Ratio:  {m_best['sharpe']:.2f}")
    print(f"    Max Drawdown:  {m_best['max_dd']:.1%}")
    print(f"    Trades:        {m_best['n']}")
    print(f"\n  Walk-forward test WR: {m_test['wr']:.1%} (n={m_test['n']})")


if __name__ == "__main__":
    main()
