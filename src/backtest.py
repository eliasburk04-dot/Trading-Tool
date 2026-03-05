"""
Event-driven backtester for the One-Candle SMC Strategy.

Design
------
* **No look-ahead bias** – the backtester iterates bar-by-bar on the
  *entry* time-frame and only acts on information available at the close
  of each bar.
* Supports single-TF (setup == entry) and dual-TF modes.
* Handles pending limit orders, SL / TP exits, optional break-even
  and ATR trailing stop.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_backtest(
    entry_data: pd.DataFrame,
    signals_list: list[dict],
    params: dict,
) -> pd.DataFrame:
    """
    Run the event-driven backtest.

    Parameters
    ----------
    entry_data : DataFrame
        OHLCV on the *entry* time-frame (the TF on which fills / SL / TP
        are evaluated).
    signals_list : list[dict]
        Sorted signal dicts produced by ``prepare_signals_list``.
    params : dict
        Strategy + cost parameters.

    Returns
    -------
    DataFrame with one row per closed trade.
    """
    K              = int(params["entry_validity"])
    use_be         = params.get("use_breakeven", False)
    be_rr          = params.get("breakeven_rr", 1.0)
    use_trail      = params.get("use_trailing", False)
    trail_mult     = params.get("trailing_atr_mult", 2.0)
    slip_frac      = params.get("slippage_pct", 0.0) / 100.0
    fee_frac       = params.get("fee_pct", 0.0) / 100.0

    trades: list[dict[str, Any]] = []
    pending: dict | None = None
    position: dict | None = None
    sig_ptr = 0
    n_sig = len(signals_list)

    # numpy arrays for speed
    times  = entry_data.index.values
    opens  = entry_data["Open"].values.astype(np.float64)
    highs  = entry_data["High"].values.astype(np.float64)
    lows   = entry_data["Low"].values.astype(np.float64)
    closes = entry_data["Close"].values.astype(np.float64)

    n_bars = len(entry_data)

    for i in range(n_bars):
        t = times[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # ── 1. manage open position ───────────────────────────────────────
        if position is not None:
            d      = position["direction"]
            sl_cur = position["sl"]
            tp_cur = position["tp"]

            exit_price: float | None = None
            exit_type: str | None = None

            if d == 1:  # long
                sl_hit = l <= sl_cur
                tp_hit = h >= tp_cur
                if sl_hit and tp_hit:
                    # conservative: SL wins
                    exit_price, exit_type = sl_cur, "sl"
                elif sl_hit:
                    exit_price, exit_type = min(sl_cur, o) if o < sl_cur else sl_cur, "sl"
                elif tp_hit:
                    exit_price, exit_type = max(tp_cur, o) if o > tp_cur else tp_cur, "tp"
            else:  # short
                sl_hit = h >= sl_cur
                tp_hit = l <= tp_cur
                if sl_hit and tp_hit:
                    exit_price, exit_type = sl_cur, "sl"
                elif sl_hit:
                    exit_price, exit_type = max(sl_cur, o) if o > sl_cur else sl_cur, "sl"
                elif tp_hit:
                    exit_price, exit_type = min(tp_cur, o) if o < tp_cur else tp_cur, "tp"

            if exit_type is not None:
                # apply slippage (adverse)
                if d == 1:
                    exit_price *= (1 - slip_frac)
                else:
                    exit_price *= (1 + slip_frac)
                pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] - fee_frac
                trades.append({
                    "signal_time":  position["signal_time"],
                    "entry_time":   position["entry_time"],
                    "exit_time":    pd.Timestamp(t),
                    "direction":    d,
                    "entry_price":  position["entry_price"],
                    "exit_price":   exit_price,
                    "sl":           position["orig_sl"],
                    "tp":           position["orig_tp"],
                    "pnl_pct":      pnl_pct,
                    "exit_type":    exit_type,
                    "ref_high":     position["ref_high"],
                    "ref_low":      position["ref_low"],
                })
                position = None
            else:
                # optional break-even
                if use_be:
                    ep = position["entry_price"]
                    risk0 = abs(ep - position["orig_sl"])
                    if d == 1:
                        if h >= ep + be_rr * risk0 and position["sl"] < ep:
                            position["sl"] = ep
                    else:
                        if l <= ep - be_rr * risk0 and position["sl"] > ep:
                            position["sl"] = ep
                # optional trailing stop
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

        # ── 2. check pending limit order ──────────────────────────────────
        if pending is not None:
            if i > pending["expiry_idx"]:
                pending = None
            else:
                ep = pending["entry_price"]
                d  = pending["direction"]
                filled = (d == 1 and l <= ep) or (d == -1 and h >= ep)
                if filled:
                    actual_entry = ep * (1 + slip_frac) if d == 1 else ep * (1 - slip_frac)
                    position = {
                        "direction":    d,
                        "entry_price":  actual_entry,
                        "sl":           pending["sl"],
                        "tp":           pending["tp"],
                        "orig_sl":      pending["sl"],
                        "orig_tp":      pending["tp"],
                        "signal_time":  pending["signal_time"],
                        "entry_time":   pd.Timestamp(t),
                        "ref_high":     pending["ref_high"],
                        "ref_low":      pending["ref_low"],
                        "atr":          pending["atr"],
                    }
                    pending = None

        # ── 3. consume new signals ────────────────────────────────────────
        while sig_ptr < n_sig and np.datetime64(signals_list[sig_ptr]["entry_after"]) <= t:
            if pending is None and position is None:
                sig = signals_list[sig_ptr]
                pending = {
                    "direction":    sig["direction"],
                    "entry_price":  sig["entry_price"],
                    "sl":           sig["sl"],
                    "tp":           sig["tp"],
                    "signal_time":  sig["signal_time"],
                    "expiry_idx":   i + K,
                    "ref_high":     sig["ref_high"],
                    "ref_low":      sig["ref_low"],
                    "atr":          sig["atr"],
                }
            sig_ptr += 1

    # close any remaining position at last close
    if position is not None:
        d = position["direction"]
        exit_price = closes[-1]
        pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] - fee_frac
        trades.append({
            "signal_time":  position["signal_time"],
            "entry_time":   position["entry_time"],
            "exit_time":    pd.Timestamp(times[-1]),
            "direction":    d,
            "entry_price":  position["entry_price"],
            "exit_price":   exit_price,
            "sl":           position["orig_sl"],
            "tp":           position["orig_tp"],
            "pnl_pct":      pnl_pct,
            "exit_type":    "end",
            "ref_high":     position["ref_high"],
            "ref_low":      position["ref_low"],
        })

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)
