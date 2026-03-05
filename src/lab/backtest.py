from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pandas as pd

from src.lab.risk import DailyRiskState, calculate_position_notional, compute_stop_levels, pre_trade_risk_check
from src.lab.sessions import SessionRules, is_end_of_day_bar
from src.lab.timeframes import Timeframe


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    position_fraction: float
    max_exposure: float
    transaction_cost_bps: float
    slippage_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_drawdown_circuit_breaker: float
    max_concurrent_trades: int
    cooldown_bars_after_loss: int
    timeframe: Timeframe = Timeframe.D1
    max_trades_per_day: int = 999_999
    daily_loss_limit_pct: float = 1.0
    max_consecutive_losses_per_day: int = 999_999
    flatten_end_of_day: bool = False
    allow_overnight: bool = True
    time_stop_bars: int = 0
    max_notional: float | None = None
    session_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timeframe"] = self.timeframe.value
        return payload


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metadata: Dict[str, Any]


def _apply_slippage(price: float, slippage_bps: float, is_buy: bool) -> float:
    multiplier = 1.0 + (slippage_bps / 10_000.0)
    return price * multiplier if is_buy else price / multiplier


def _reset_daily_risk_state(timestamp: pd.Timestamp, state: DailyRiskState) -> DailyRiskState:
    current_date = timestamp.date().isoformat()
    if state.session_date == current_date:
        return state
    return DailyRiskState(session_date=current_date)


def _exit_position(
    position: Dict[str, Any],
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    cash: float,
    config: BacktestConfig,
) -> Dict[str, Any]:
    exit_notional = position["quantity"] * exit_price
    exit_fee = exit_notional * config.transaction_cost_bps / 10_000.0
    gross_pnl = exit_notional - position["entry_notional"]
    net_pnl = gross_pnl - position["entry_fee"] - exit_fee
    cash = cash + exit_notional - exit_fee
    return {
        "cash": cash,
        "trade": {
            "instrument": position["instrument"],
            "signal_time": position["signal_time"],
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "quantity": position["quantity"],
            "entry_notional": position["entry_notional"],
            "exit_notional": exit_notional,
            "fees": position["entry_fee"] + exit_fee,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(position["entry_notional"], 1e-9),
            "bars_held": position["bars_held"],
            "timeframe": config.timeframe.value,
            "exit_reason": exit_reason,
        },
    }


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig,
    instrument: str,
) -> BacktestResult:
    required_signal_columns = {"entry", "exit"}
    if not required_signal_columns.issubset(set(signals.columns)):
        raise ValueError("signals must contain entry and exit columns")

    aligned_prices = prices.copy().sort_index()
    aligned_signals = signals.reindex(aligned_prices.index).copy()
    aligned_signals["entry"] = aligned_signals["entry"].fillna(False).astype(bool)
    aligned_signals["exit"] = aligned_signals["exit"].fillna(False).astype(bool)
    if "time_exit_bars" not in aligned_signals.columns:
        aligned_signals["time_exit_bars"] = 0
    aligned_signals["time_exit_bars"] = aligned_signals["time_exit_bars"].fillna(0).astype(int)
    rules = SessionRules(**config.session_rules) if config.session_rules else SessionRules()

    trades = []
    equity_history = [(aligned_prices.index[0], float(config.initial_capital))]
    cash = float(config.initial_capital)
    peak_equity = float(config.initial_capital)
    position: Optional[Dict[str, Any]] = None
    cooldown_until_bar = -1
    circuit_broken = False
    daily_risk = DailyRiskState(session_date=None)

    for bar_index in range(1, len(aligned_prices)):
        timestamp = aligned_prices.index[bar_index]
        next_timestamp = aligned_prices.index[bar_index + 1] if bar_index + 1 < len(aligned_prices) else None
        row = aligned_prices.iloc[bar_index]
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        previous_signal = aligned_signals.iloc[bar_index - 1]
        daily_risk = _reset_daily_risk_state(timestamp, daily_risk)

        if position is not None:
            position["bars_held"] += 1
            exit_price = None
            exit_reason = ""

            if bool(previous_signal["exit"]):
                exit_price = _apply_slippage(open_price, config.slippage_bps, is_buy=False)
                exit_reason = "signal_exit"
            else:
                stop_price = float(position["stop_price"])
                target_price = float(position["target_price"])
                if low_price <= stop_price and high_price >= target_price:
                    exit_price = _apply_slippage(min(stop_price, open_price), config.slippage_bps, is_buy=False)
                    exit_reason = "stop_loss"
                elif low_price <= stop_price:
                    exit_price = _apply_slippage(min(stop_price, open_price), config.slippage_bps, is_buy=False)
                    exit_reason = "stop_loss"
                elif high_price >= target_price:
                    exit_price = _apply_slippage(max(target_price, open_price), config.slippage_bps, is_buy=False)
                    exit_reason = "take_profit"
                elif int(position.get("time_exit_bars", config.time_stop_bars)) > 0 and position["bars_held"] >= int(position.get("time_exit_bars", config.time_stop_bars)):
                    exit_price = _apply_slippage(close_price, config.slippage_bps, is_buy=False)
                    exit_reason = "time_stop"
                elif config.flatten_end_of_day and not config.allow_overnight and config.timeframe != Timeframe.D1:
                    if is_end_of_day_bar(timestamp, next_timestamp, rules):
                        exit_price = _apply_slippage(close_price, config.slippage_bps, is_buy=False)
                        exit_reason = "end_of_day"
                elif config.trailing_stop_pct > 0:
                    trailing_floor = high_price * (1.0 - float(config.trailing_stop_pct))
                    position["stop_price"] = max(float(position["stop_price"]), trailing_floor)

            if exit_price is not None:
                exit_payload = _exit_position(position, timestamp, exit_price, exit_reason, cash, config)
                cash = float(exit_payload["cash"])
                trades.append(exit_payload["trade"])
                daily_risk.realized_pnl += float(exit_payload["trade"]["net_pnl"])
                daily_risk.trades_taken += 1
                if float(exit_payload["trade"]["net_pnl"]) < 0:
                    daily_risk.consecutive_losses += 1
                    cooldown_until_bar = bar_index + int(config.cooldown_bars_after_loss)
                else:
                    daily_risk.consecutive_losses = 0
                position = None

        current_equity = cash if position is None else cash + position["quantity"] * close_price
        peak_equity = max(peak_equity, current_equity)
        drawdown = (current_equity - peak_equity) / max(peak_equity, 1e-9)
        if drawdown <= -float(config.max_drawdown_circuit_breaker):
            circuit_broken = True

        if (
            position is None
            and not circuit_broken
            and bar_index > cooldown_until_bar
            and bool(previous_signal["entry"])
        ):
            risk_config = config.to_dict()
            if config.max_notional is not None:
                risk_config["max_notional"] = float(config.max_notional)
            else:
                risk_config["max_notional"] = calculate_position_notional(cash, risk_config)
            proposed_notional = calculate_position_notional(cash, risk_config)
            state = SimpleNamespace(risk=daily_risk)
            decision = pre_trade_risk_check(timestamp, state, risk_config, proposed_notional)
            if decision.allowed:
                entry_price = _apply_slippage(open_price, config.slippage_bps, is_buy=True)
                quantity = proposed_notional / max(entry_price, 1e-9)
                entry_fee = proposed_notional * float(config.transaction_cost_bps) / 10_000.0
                stop_price, take_profit = compute_stop_levels(entry_price, risk_config)
                time_exit_bars = int(previous_signal["time_exit_bars"]) if int(previous_signal["time_exit_bars"]) > 0 else int(config.time_stop_bars)
                cash = cash - proposed_notional - entry_fee
                position = {
                    "instrument": instrument,
                    "signal_time": aligned_prices.index[bar_index - 1],
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "entry_notional": proposed_notional,
                    "entry_fee": entry_fee,
                    "stop_price": stop_price,
                    "target_price": take_profit,
                    "bars_held": 0,
                    "time_exit_bars": time_exit_bars,
                }
                immediate_exit_price = None
                immediate_exit_reason = ""
                if low_price <= stop_price and high_price >= take_profit:
                    immediate_exit_price = _apply_slippage(min(stop_price, open_price), config.slippage_bps, is_buy=False)
                    immediate_exit_reason = "stop_loss"
                elif low_price <= stop_price:
                    immediate_exit_price = _apply_slippage(min(stop_price, open_price), config.slippage_bps, is_buy=False)
                    immediate_exit_reason = "stop_loss"
                elif high_price >= take_profit:
                    immediate_exit_price = _apply_slippage(max(take_profit, open_price), config.slippage_bps, is_buy=False)
                    immediate_exit_reason = "take_profit"

                if immediate_exit_price is not None:
                    exit_payload = _exit_position(position, timestamp, immediate_exit_price, immediate_exit_reason, cash, config)
                    cash = float(exit_payload["cash"])
                    trades.append(exit_payload["trade"])
                    daily_risk.realized_pnl += float(exit_payload["trade"]["net_pnl"])
                    daily_risk.trades_taken += 1
                    if float(exit_payload["trade"]["net_pnl"]) < 0:
                        daily_risk.consecutive_losses += 1
                        cooldown_until_bar = bar_index + int(config.cooldown_bars_after_loss)
                    else:
                        daily_risk.consecutive_losses = 0
                    position = None

        current_equity = cash if position is None else cash + position["quantity"] * close_price
        equity_history.append((timestamp, current_equity))

    if position is not None:
        final_time = aligned_prices.index[-1]
        final_price = _apply_slippage(float(aligned_prices["Close"].iloc[-1]), config.slippage_bps, is_buy=False)
        exit_reason = "end_of_data" if config.allow_overnight else "end_of_day"
        exit_payload = _exit_position(position, final_time, final_price, exit_reason, cash, config)
        cash = float(exit_payload["cash"])
        trades.append(exit_payload["trade"])
        equity_history[-1] = (final_time, cash)

    trades_frame = pd.DataFrame(trades)
    equity_curve = pd.Series(
        [value for _, value in equity_history],
        index=[timestamp for timestamp, _ in equity_history],
        name=instrument,
    )
    return BacktestResult(
        trades=trades_frame,
        equity_curve=equity_curve,
        metadata={
            "instrument": instrument,
            "timeframe": config.timeframe.value,
            "circuit_broken": circuit_broken,
            "final_equity": float(cash if position is None else equity_curve.iloc[-1]),
        },
    )
