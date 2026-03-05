from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
import json
import random

import pandas as pd

from src.lab.backtest import BacktestConfig, run_backtest
from src.lab.execution.broker import Order, PaperBrokerAdapter
from src.lab.execution.clock import LivePaperClock, ReplayClock
from src.lab.execution.state import ExecutionState
from src.lab.risk import DailyRiskState, calculate_position_notional
from src.lab.strategies import get_strategy
from src.lab.timeframes import Timeframe


@dataclass
class FillModel:
    spread_bps: float
    slippage_bps: float
    seed: int

    def fill_price(self, side: str, open_price: float, high_price: float, low_price: float, limit_price: float) -> float:
        rng = random.Random(self.seed)
        spread = open_price * (self.spread_bps / 10_000.0)
        slippage = open_price * (self.slippage_bps / 10_000.0) * (0.5 + rng.random())
        if side == "buy":
            candidate = max(open_price + spread + slippage, limit_price)
            return min(candidate, high_price)
        candidate = min(open_price - spread - slippage, limit_price)
        return max(candidate, low_price)


def _build_backtest_config(champion: Mapping[str, Any]) -> BacktestConfig:
    risk = dict(champion["risk"])
    risk["timeframe"] = Timeframe.from_value(champion["champion"]["timeframe"])
    risk["session_rules"] = champion["champion"]["session_rules"]
    return BacktestConfig(**risk)


class PaperExecutionEngine:
    def __init__(
        self,
        champion: Mapping[str, Any],
        report_dir: Path,
        mode: str,
        fill_model: FillModel,
        existing_state: ExecutionState | None = None,
    ) -> None:
        self.champion = champion
        self.report_dir = report_dir
        self.mode = mode
        self.fill_model = fill_model
        self.paper_dir = report_dir / "paper"
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.paper_dir / "events.jsonl"
        self.state_path = self.paper_dir / "state.json"
        self.broker = PaperBrokerAdapter()
        self.state = existing_state or ExecutionState(
            cash=float(champion["risk"]["initial_capital"]),
            realized_pnl=0.0,
            open_positions={},
            last_processed_at=None,
            risk=DailyRiskState(session_date=None),
        )

    def _log_event(self, payload: Dict[str, Any]) -> None:
        payload["paper_only"] = True
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def run(self, market_data: Mapping[str, Mapping[str, Any]], days: int) -> Dict[str, Path]:
        strategy = get_strategy(self.champion["champion"]["strategy_id"])
        params = self.champion["champion"]["parameters"]
        timeframe = self.champion["champion"]["timeframe"]
        backtest_config = _build_backtest_config(self.champion)

        for instrument, payload in market_data.items():
            if timeframe not in payload:
                continue
            prices = payload[timeframe]["prices"].iloc[-days * Timeframe.from_value(timeframe).bars_per_day :].copy()
            signals = strategy.generate_signals(prices, params)
            result = run_backtest(prices, signals, backtest_config, instrument=instrument)
            clock = ReplayClock(prices.index) if self.mode == "replay" else LivePaperClock(prices.index, max_iterations=1)
            for timestamp in clock:
                self.state.last_processed_at = timestamp.isoformat()
            for ordinal, (_, trade) in enumerate(result.trades.iterrows(), start=1):
                order_notional = calculate_position_notional(self.state.cash, self.champion["risk"])
                quantity = order_notional / max(float(trade["entry_price"]), 1e-9)
                entry_order = Order(
                    order_id="%s-entry-%d" % (instrument, ordinal),
                    instrument=instrument,
                    side="buy",
                    quantity=quantity,
                    submitted_at=pd.Timestamp(trade["entry_time"]).isoformat(),
                    limit_price=float(trade["entry_price"]),
                    reason="strategy_signal",
                )
                entry_fill_price = self.fill_model.fill_price(
                    side="buy",
                    open_price=float(trade["entry_price"]),
                    high_price=float(trade["entry_price"]),
                    low_price=float(trade["entry_price"]),
                    limit_price=float(trade["entry_price"]),
                )
                self.broker.place_order(entry_order, entry_fill_price, pd.Timestamp(trade["entry_time"]))

                exit_order = Order(
                    order_id="%s-exit-%d" % (instrument, ordinal),
                    instrument=instrument,
                    side="sell",
                    quantity=float(trade["quantity"]),
                    submitted_at=pd.Timestamp(trade["exit_time"]).isoformat(),
                    limit_price=float(trade["exit_price"]),
                    reason=str(trade["exit_reason"]),
                )
                exit_fill_price = self.fill_model.fill_price(
                    side="sell",
                    open_price=float(trade["exit_price"]),
                    high_price=float(trade["exit_price"]),
                    low_price=float(trade["exit_price"]),
                    limit_price=float(trade["exit_price"]),
                )
                self.broker.place_order(exit_order, exit_fill_price, pd.Timestamp(trade["exit_time"]))
                self.state.cash += float(trade["net_pnl"])
                self.state.realized_pnl += float(trade["net_pnl"])
                self._log_event(
                    {
                        "instrument": instrument,
                        "timeframe": timeframe,
                        "strategy_id": self.champion["champion"]["strategy_id"],
                        "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                        "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
                        "net_pnl": float(trade["net_pnl"]),
                        "mode": self.mode,
                    }
                )
                self.state.persist(self.state_path)

        orders_path = self.paper_dir / "orders.csv"
        fills_path = self.paper_dir / "fills.csv"
        pd.DataFrame(self.broker.orders).to_csv(orders_path, index=False)
        pd.DataFrame(self.broker.fills).to_csv(fills_path, index=False)
        summary_path = self.paper_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "paper_only": True,
                    "mode": self.mode,
                    "ending_equity": self.state.cash,
                    "fill_count": len(self.broker.fills),
                    "order_count": len(self.broker.orders),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.state.persist(self.state_path)
        return {
            "orders": orders_path,
            "fills": fills_path,
            "summary": summary_path,
            "state": self.state_path,
            "events": self.events_path,
        }
