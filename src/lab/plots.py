from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.lab.metrics import compute_drawdown


def _prepare_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_equity_curve(equity_curve: pd.Series, title: str, output_path: Path) -> Path:
    _prepare_path(output_path)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(equity_curve.index, equity_curve.values, color="#1f77b4", linewidth=1.2)
    axis.set_title(title)
    axis.set_ylabel("Equity")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_drawdown_curve(equity_curve: pd.Series, title: str, output_path: Path) -> Path:
    _prepare_path(output_path)
    drawdown = compute_drawdown(equity_curve) * 100.0
    fig, axis = plt.subplots(figsize=(12, 3))
    axis.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.35)
    axis.set_title(title)
    axis.set_ylabel("Drawdown %")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_return_distribution(trades: pd.DataFrame, title: str, output_path: Path) -> Path:
    _prepare_path(output_path)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(trades["return_pct"], bins=20, color="#2ca02c", edgecolor="#1b1b1b", alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Trade Return")
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_trade_markers(prices: pd.DataFrame, trades: pd.DataFrame, title: str, output_path: Path) -> Path:
    _prepare_path(output_path)
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(prices.index, prices["Close"], color="#111827", linewidth=0.9, label="Close")
    if not trades.empty:
        axis.scatter(trades["entry_time"], trades["entry_price"], color="#16a34a", marker="^", s=40, label="Entry")
        axis.scatter(trades["exit_time"], trades["exit_price"], color="#dc2626", marker="x", s=36, label="Exit")
    axis.set_title(title)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
