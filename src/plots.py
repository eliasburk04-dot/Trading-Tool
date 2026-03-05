"""
Plotting module – all visualisation uses **matplotlib** only (no seaborn).

Generates:
 1. Trade chart – price line with entry / exit markers + SL / TP levels
 2. Equity curve
 3. Drawdown chart
 4. Parameter-sweep heatmap
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.metrics import build_equity_curve

# ── style defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        9,
})

WIN_CLR  = "#26a69a"
LOSS_CLR = "#ef5350"
LINE_CLR = "#455a64"


# ── 1. trade chart ────────────────────────────────────────────────────────────

def plot_trade_chart(
    price_df: pd.DataFrame,
    trades: pd.DataFrame,
    asset: str,
    tf_label: str,
    out_dir: str,
    max_bars: int = 3000,
) -> str | None:
    """Price line with arrows at entries, × at exits, thin horiz. lines for SL/TP."""
    if trades.empty:
        return None

    # trim for readability
    df = price_df.iloc[-max_bars:].copy()
    trd = trades.copy()

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(df.index, df["Close"], color=LINE_CLR, lw=0.6, alpha=0.85, label="Close")

    for _, r in trd.iterrows():
        won = r["pnl_pct"] > 0
        clr = WIN_CLR if won else LOSS_CLR
        mk  = "^" if r["direction"] == 1 else "v"
        # entry
        if r["entry_time"] >= df.index[0]:
            ax.scatter(r["entry_time"], r["entry_price"], marker=mk, color=clr,
                       s=48, zorder=5, edgecolors="k", linewidths=0.3)
        # exit
        if r["exit_time"] >= df.index[0]:
            ax.scatter(r["exit_time"], r["exit_price"], marker="x", color=clr,
                       s=36, zorder=5, linewidths=1.0)
        # thin SL / TP lines
        if r["entry_time"] >= df.index[0] and r["exit_time"] >= df.index[0]:
            ax.plot([r["entry_time"], r["exit_time"]], [r["sl"], r["sl"]],
                    color=LOSS_CLR, lw=0.4, ls="--", alpha=0.5)
            ax.plot([r["entry_time"], r["exit_time"]], [r["tp"], r["tp"]],
                    color=WIN_CLR, lw=0.4, ls="--", alpha=0.5)

    ax.set_title(f"{asset}  –  Trade Chart  ({tf_label})", fontweight="bold")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fp = os.path.join(out_dir, f"{asset}_trades_{tf_label.replace(' ', '_').replace('/', '_')}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── 2. equity curve ──────────────────────────────────────────────────────────

def plot_equity(
    trades: pd.DataFrame,
    asset: str,
    tf_label: str,
    out_dir: str,
    initial_capital: float = 10_000.0,
) -> str | None:
    if trades.empty:
        return None
    eq = build_equity_curve(trades, initial_capital)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eq.index, eq.values, color="#1565c0", lw=1.0)
    ax.fill_between(eq.index, initial_capital, eq.values, alpha=0.08, color="#1565c0")
    ax.axhline(initial_capital, color="grey", lw=0.5, ls="--")
    ax.set_title(f"{asset}  –  Equity Curve  ({tf_label})", fontweight="bold")
    ax.set_ylabel("Equity")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fp = os.path.join(out_dir, f"{asset}_equity_{tf_label.replace(' ', '_').replace('/', '_')}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── 3. drawdown chart ────────────────────────────────────────────────────────

def plot_drawdown(
    trades: pd.DataFrame,
    asset: str,
    tf_label: str,
    out_dir: str,
    initial_capital: float = 10_000.0,
) -> str | None:
    if trades.empty:
        return None
    eq = build_equity_curve(trades, initial_capital)
    peak = np.maximum.accumulate(eq.values)
    dd = (eq.values - peak) / peak * 100  # in percent
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(eq.index, dd, 0, color=LOSS_CLR, alpha=0.35)
    ax.plot(eq.index, dd, color=LOSS_CLR, lw=0.7)
    ax.set_title(f"{asset}  –  Drawdown  ({tf_label})", fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fp = os.path.join(out_dir, f"{asset}_drawdown_{tf_label.replace(' ', '_').replace('/', '_')}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── 4. parameter-sweep heatmap ───────────────────────────────────────────────

def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    row_param: str,
    col_param: str,
    value_col: str,
    title: str,
    out_dir: str,
    filename: str = "sweep_heatmap.png",
) -> str | None:
    """
    Pivot *sweep_df* on (row_param, col_param) → value_col and draw a
    colour-coded heatmap with text annotations.
    """
    if sweep_df.empty:
        return None
    piv = sweep_df.pivot_table(index=row_param, columns=col_param,
                                values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, len(piv.columns) * 1.2),
                                     max(4, len(piv.index) * 0.8)))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(v) for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(v) for v in piv.index])
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(title, fontweight="bold")
    # annotate cells
    for yi in range(piv.shape[0]):
        for xi in range(piv.shape[1]):
            val = piv.values[yi, xi]
            if np.isfinite(val):
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fp = os.path.join(out_dir, filename)
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── 5. combined summary bar chart ────────────────────────────────────────────

def plot_summary_bars(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    out_dir: str,
    filename: str = "summary_bar.png",
) -> str | None:
    if summary.empty or metric not in summary.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = pd.to_numeric(summary[metric].str.rstrip("%"), errors="coerce")
    colours = [WIN_CLR if v > 0 else LOSS_CLR for v in vals]
    ax.bar(summary.index, vals, color=colours, edgecolor="k", linewidth=0.3)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = os.path.join(out_dir, filename)
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp
