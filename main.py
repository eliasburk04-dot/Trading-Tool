#!/usr/bin/env python3
"""
main.py – One-Candle SMC Strategy: data fetch → backtest → sweep → plots.

Usage
-----
    pip install -r requirements.txt
    python main.py                     # run with default params
    python main.py --no-sweep          # skip the parameter sweep (faster)
    python main.py --rr 4 --swing 15   # override individual defaults

All tuneable parameters live in  config.py  (ASSETS, TF_CONFIGS,
DEFAULT_PARAMS, SWEEP_GRID).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from copy import deepcopy
from textwrap import dedent

# Force UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
from config import (
    ASSETS, TF_CONFIGS, DEFAULT_PARAMS, SWEEP_GRID,
    WF_TRAIN_FRAC, WF_OPTIMISE_METRIC,
    OUTPUT_DIR, DATA_CACHE_DIR, TF_SECONDS,
)
from src.data      import get_asset_data, resample_ohlcv
from src.strategy  import detect_signals, prepare_signals_list
from src.backtest  import run_backtest
from src.metrics   import compute_metrics, numeric_metrics, build_equity_curve
from src.plots     import (
    plot_trade_chart, plot_equity, plot_drawdown,
    plot_sweep_heatmap, plot_summary_bars,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _tf_secs(tf: str) -> int:
    return TF_SECONDS.get(tf, 3600)


def _run_single(
    setup_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    params: dict,
    setup_tf: str,
) -> pd.DataFrame:
    """Detect signals on *setup_df*, backtest on *entry_df*."""
    sig_df  = detect_signals(setup_df, params)
    signals = prepare_signals_list(sig_df, params, _tf_secs(setup_tf))
    trades  = run_backtest(entry_df, signals, params)
    return trades


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for k, v in m.items():
        print(f"  {k:<18s} {v}")


# ═════════════════════════════════════════════════════════════════════════════
#  PARAMETER SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def _parameter_sweep(
    setup_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    base_params: dict,
    setup_tf: str,
    grid: dict[str, list],
) -> pd.DataFrame:
    """
    Iterate over all grid combinations, run backtests, return a DataFrame
    with one row per combination and its numeric metrics.
    """
    keys  = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows: list[dict] = []
    total = len(combos)

    for idx, vals in enumerate(combos, 1):
        p = deepcopy(base_params)
        for k, v in zip(keys, vals):
            p[k] = v
        trades = _run_single(setup_df, entry_df, p, setup_tf)
        nm = numeric_metrics(trades)
        row = {k: v for k, v in zip(keys, vals)}
        row.update(nm)
        rows.append(row)
        if idx % 20 == 0 or idx == total:
            print(f"    sweep {idx}/{total} …")

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ═════════════════════════════════════════════════════════════════════════════

def _walk_forward(
    setup_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    base_params: dict,
    setup_tf: str,
    grid: dict[str, list],
    train_frac: float,
    opt_metric: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Train / test split → optimise on train → evaluate on test.

    Returns (best_params, train_trades, test_trades).
    """
    n = len(entry_df)
    split = int(n * train_frac)
    train_entry = entry_df.iloc[:split]
    test_entry  = entry_df.iloc[split:]

    # same split for setup (align by time)
    cutoff = train_entry.index[-1]
    train_setup = setup_df[setup_df.index <= cutoff]
    test_setup  = setup_df[setup_df.index > cutoff]

    if len(train_setup) < 50 or len(test_setup) < 20:
        print("    [WARN] not enough data for walk-forward")
        return base_params, pd.DataFrame(), pd.DataFrame()

    print("    WF train …")
    sweep = _parameter_sweep(train_setup, train_entry, base_params, setup_tf, grid)
    if sweep.empty:
        return base_params, pd.DataFrame(), pd.DataFrame()

    best_row = sweep.sort_values(opt_metric, ascending=False).iloc[0]
    best_p = deepcopy(base_params)
    for k in grid:
        best_p[k] = best_row[k]

    train_trades = _run_single(train_setup, train_entry, best_p, setup_tf)
    test_trades  = _run_single(test_setup,  test_entry,  best_p, setup_tf)
    return best_p, train_trades, test_trades


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="One-Candle SMC Strategy Backtest")
    ap.add_argument("--no-sweep", action="store_true", help="Skip parameter sweep")
    ap.add_argument("--no-wf",    action="store_true", help="Skip walk-forward")
    ap.add_argument("--rr",    type=float, default=None)
    ap.add_argument("--swing", type=int,   default=None)
    ap.add_argument("--sweep-lookback", type=int, default=None)
    ap.add_argument("--entry-validity", type=int, default=None)
    ap.add_argument("--sweep-filter", action="store_true", default=False)
    args = ap.parse_args()

    params = deepcopy(DEFAULT_PARAMS)
    if args.rr is not None:            params["rr"] = args.rr
    if args.swing is not None:         params["swing_lookback"] = args.swing
    if args.sweep_lookback is not None: params["sweep_lookback"] = args.sweep_lookback
    if args.entry_validity is not None: params["entry_validity"] = args.entry_validity
    if args.sweep_filter:              params["use_sweep_filter"] = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("  ONE-CANDLE SMC STRATEGY BACKTEST")
    print("=" * 70)
    print(f"  Parameters: {params}\n")

    # ── collect per-asset + per-TF results ────────────────────────────────
    all_results: list[dict] = []
    all_trades_map: dict[str, pd.DataFrame] = {}
    all_data_map: dict[str, pd.DataFrame] = {}
    sweep_results: list[pd.DataFrame] = []
    wf_results: list[dict] = []

    for asset_name, asset_cfg in ASSETS.items():
        print(f"\n{'=' * 70}")
        print(f"  ASSET: {asset_name}")
        print(f"{'=' * 70}")

        for setup_tf, entry_tf, lookback, tf_label in TF_CONFIGS:
            # determine which base interval to fetch
            base_interval = entry_tf  # always fetch the finer TF
            df_entry, sym = get_asset_data(
                asset_name, asset_cfg["symbols"], base_interval,
                lookback, DATA_CACHE_DIR,
            )
            if df_entry is None:
                continue

            # resample for setup TF if different
            if setup_tf != entry_tf:
                df_setup = resample_ohlcv(df_entry, setup_tf)
            else:
                df_setup = df_entry.copy()

            if len(df_setup) < 60:
                print(f"    [SKIP] {tf_label}: only {len(df_setup)} setup bars")
                continue

            # ── default backtest ──────────────────────────────────────────
            print(f"\n  ▸ TF config: {tf_label}  ({sym})")
            trades = _run_single(df_setup, df_entry, params, setup_tf)
            key = f"{asset_name}|{tf_label}"
            all_trades_map[key] = trades
            all_data_map[key]   = df_entry

            m = compute_metrics(trades)
            _print_metrics(f"{asset_name}  {tf_label}", m)

            result_row = {"Asset": asset_name, "TF": tf_label, "Symbol": sym}
            result_row.update(m)
            all_results.append(result_row)

            # ── plots ─────────────────────────────────────────────────────
            plot_trade_chart(df_entry, trades, asset_name, tf_label, OUTPUT_DIR)
            plot_equity(trades, asset_name, tf_label, OUTPUT_DIR)
            plot_drawdown(trades, asset_name, tf_label, OUTPUT_DIR)

            # ── parameter sweep (only for first TF config with enough data) ──
            if not args.no_sweep and len(df_setup) >= 200:
                print(f"\n    Parameter sweep for {asset_name} / {tf_label} …")
                sw = _parameter_sweep(df_setup, df_entry, params, setup_tf, SWEEP_GRID)
                sw["asset"] = asset_name
                sw["tf"]    = tf_label
                sweep_results.append(sw)
                # save per-asset sweep
                sw.to_csv(os.path.join(OUTPUT_DIR, f"{asset_name}_sweep_{tf_label.replace(' ','_').replace('/','_')}.csv"), index=False)

            # ── walk-forward ──────────────────────────────────────────────
            if not args.no_wf and len(df_setup) >= 200:
                print(f"\n    Walk-forward for {asset_name} / {tf_label} …")
                best_p, wf_train, wf_test = _walk_forward(
                    df_setup, df_entry, params, setup_tf, SWEEP_GRID,
                    WF_TRAIN_FRAC, WF_OPTIMISE_METRIC,
                )
                m_train = compute_metrics(wf_train)
                m_test  = compute_metrics(wf_test)
                _print_metrics(f"WF TRAIN  {asset_name}  {tf_label}", m_train)
                _print_metrics(f"WF TEST   {asset_name}  {tf_label}", m_test)
                wf_results.append({
                    "Asset": asset_name, "TF": tf_label,
                    "Best Params": {k: best_p[k] for k in SWEEP_GRID},
                    "Train Sharpe": m_train.get("Sharpe", "—"),
                    "Test Sharpe":  m_test.get("Sharpe", "—"),
                    "Train Trades": m_train.get("Total Trades", 0),
                    "Test Trades":  m_test.get("Total Trades", 0),
                    "Train Return": m_train.get("Total Return", "—"),
                    "Test Return":  m_test.get("Total Return", "—"),
                })

            # only run sweep & WF once per asset (on the first TF config
            # with enough data) to keep runtime reasonable
            break  # remove this line to run all TF combos

    # ── combined summary table ────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    if not summary_df.empty:
        print("\n\n" + "=" * 70)
        print("  SUMMARY TABLE")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    # ── sweep heatmap (aggregate) ─────────────────────────────────────────
    if sweep_results:
        all_sweep = pd.concat(sweep_results, ignore_index=True)
        all_sweep.to_csv(os.path.join(OUTPUT_DIR, "sweep_all.csv"), index=False)
        # heatmap: swing_lookback vs rr, coloured by sharpe
        if "swing_lookback" in all_sweep.columns and "rr" in all_sweep.columns:
            plot_sweep_heatmap(
                all_sweep, "swing_lookback", "rr", "sharpe",
                "Parameter Sweep – Sharpe (avg across assets)",
                OUTPUT_DIR, "sweep_heatmap_sharpe.png",
            )
            plot_sweep_heatmap(
                all_sweep, "swing_lookback", "rr", "total_return",
                "Parameter Sweep – Total Return (avg across assets)",
                OUTPUT_DIR, "sweep_heatmap_return.png",
            )
            plot_sweep_heatmap(
                all_sweep, "entry_validity", "rr", "sharpe",
                "Parameter Sweep – Entry Validity vs RR (Sharpe)",
                OUTPUT_DIR, "sweep_heatmap_ev_rr.png",
            )

    # ── walk-forward summary ──────────────────────────────────────────────
    if wf_results:
        wf_df = pd.DataFrame(wf_results)
        print("\n\n" + "=" * 70)
        print("  WALK-FORWARD RESULTS")
        print("=" * 70)
        for _, r in wf_df.iterrows():
            print(f"  {r['Asset']} {r['TF']}")
            print(f"    Best params : {r['Best Params']}")
            print(f"    Train:  Sharpe={r['Train Sharpe']}  Return={r['Train Return']}  Trades={r['Train Trades']}")
            print(f"    Test:   Sharpe={r['Test Sharpe']}  Return={r['Test Return']}  Trades={r['Test Trades']}")
        wf_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s.  Output → {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 70}")

    # ── interpretation ────────────────────────────────────────────────────
    print(dedent("""
    +----------------------------------------------------------------------+
    |  INTERPRETATION GUIDE                                                |
    +----------------------------------------------------------------------+
    |  * Sharpe > 1 and Profit Factor > 1.5 -> promising edge              |
    |  * Win Rate around 35-45% is normal for RR >= 3 strategies           |
    |  * Max Drawdown > 25% usually means the strategy needs filters       |
    |  * Walk-forward test Sharpe should be >= 50% of train Sharpe         |
    |  * If sweep heatmap shows a *gradient* (not random), the strategy    |
    |    has genuine sensitivity to that parameter -> good sign            |
    |                                                                      |
    |  TYPICAL FAILURE MODES                                               |
    |  * Range / chop markets -> many false MSS signals                    |
    |  * News spikes -> SL hunted before move completes                    |
    |  * Low-volatility periods -> tiny candles, bad RR                    |
    |  * Over-fitting to N / RR on one asset -> poor walk-forward          |
    +----------------------------------------------------------------------+
    """))


if __name__ == "__main__":
    main()
