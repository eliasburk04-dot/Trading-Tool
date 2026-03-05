from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import json

import pandas as pd

from src.lab.backtest import BacktestConfig, run_backtest
from src.lab.strategies import get_strategy


@dataclass(frozen=True)
class Order:
    order_id: str
    instrument: str
    side: str
    quantity: float
    submitted_at: str
    limit_price: float
    reason: str
    paper_only: bool = True


@dataclass(frozen=True)
class Fill:
    order_id: str
    instrument: str
    side: str
    quantity: float
    filled_at: str
    fill_price: float
    paper_only: bool = True


class BrokerAdapter:
    paper_only = True

    def submit_order(self, order: Order, fill_price: float, fill_time: pd.Timestamp) -> Fill:
        raise NotImplementedError


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.orders: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []

    def submit_order(self, order: Order, fill_price: float, fill_time: pd.Timestamp) -> Fill:
        self.orders.append(asdict(order))
        fill = Fill(
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=order.quantity,
            filled_at=fill_time.isoformat(),
            fill_price=float(fill_price),
        )
        self.fills.append(asdict(fill))
        return fill


def signal_to_order(
    signal_time: pd.Timestamp,
    execution_time: pd.Timestamp,
    instrument: str,
    side: str,
    price: float,
    equity: float,
    risk_config: Mapping[str, Any],
    ordinal: int,
) -> Order:
    max_notional = equity * min(float(risk_config["position_fraction"]), float(risk_config["max_exposure"]))
    quantity = max_notional / max(float(price), 1e-9)
    if quantity <= 0:
        raise ValueError("Order quantity must be positive")
    return Order(
        order_id="%s-%s-%s" % (instrument, execution_time.strftime("%Y%m%d"), ordinal),
        instrument=instrument,
        side=side,
        quantity=float(quantity),
        submitted_at=signal_time.isoformat(),
        limit_price=float(price),
        reason="strategy_signal",
    )


def _build_backtest_config(champion: Mapping[str, Any]) -> BacktestConfig:
    risk = champion["risk"]
    return BacktestConfig(
        initial_capital=float(risk["initial_capital"]),
        position_fraction=float(risk["position_fraction"]),
        max_exposure=float(risk["max_exposure"]),
        transaction_cost_bps=float(risk["transaction_cost_bps"]),
        slippage_bps=float(risk["slippage_bps"]),
        stop_loss_pct=float(risk["stop_loss_pct"]),
        take_profit_pct=float(risk["take_profit_pct"]),
        trailing_stop_pct=float(risk["trailing_stop_pct"]),
        max_drawdown_circuit_breaker=float(risk["max_drawdown_circuit_breaker"]),
        max_concurrent_trades=int(risk["max_concurrent_trades"]),
        cooldown_bars_after_loss=int(risk["cooldown_bars_after_loss"]),
    )


def run_paper_simulation(
    champion: Mapping[str, Any],
    market_data: Mapping[str, Mapping[str, Any]],
    replay_days: int,
    report_dir: Path,
) -> Dict[str, Path]:
    strategy = get_strategy(champion["champion"]["strategy_id"])
    params = champion["champion"]["parameters"]
    risk = champion["risk"]
    backtest_config = _build_backtest_config(champion)
    broker = PaperBrokerAdapter()

    current_equity = float(risk["initial_capital"])
    paper_dir = report_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    order_count = 0

    for instrument, payload in market_data.items():
        prices = payload["prices"].iloc[-replay_days:].copy()
        signals = strategy.generate_signals(prices, params)
        result = run_backtest(prices, signals, backtest_config, instrument=instrument)

        for _, trade in result.trades.iterrows():
            order_count += 1
            entry_order = signal_to_order(
                signal_time=pd.Timestamp(trade["signal_time"]),
                execution_time=pd.Timestamp(trade["entry_time"]),
                instrument=instrument,
                side="buy",
                price=float(trade["entry_price"]),
                equity=current_equity,
                risk_config=risk,
                ordinal=order_count,
            )
            broker.submit_order(entry_order, fill_price=float(trade["entry_price"]), fill_time=pd.Timestamp(trade["entry_time"]))

            order_count += 1
            exit_order = Order(
                order_id="%s-%s-%s" % (instrument, pd.Timestamp(trade["exit_time"]).strftime("%Y%m%d"), order_count),
                instrument=instrument,
                side="sell",
                quantity=float(trade["quantity"]),
                submitted_at=pd.Timestamp(trade["exit_time"]).isoformat(),
                limit_price=float(trade["exit_price"]),
                reason=str(trade["exit_reason"]),
            )
            broker.submit_order(exit_order, fill_price=float(trade["exit_price"]), fill_time=pd.Timestamp(trade["exit_time"]))
            current_equity += float(trade["net_pnl"])

    orders_path = paper_dir / "orders.csv"
    fills_path = paper_dir / "fills.csv"
    pd.DataFrame(broker.orders).to_csv(orders_path, index=False)
    pd.DataFrame(broker.fills).to_csv(fills_path, index=False)

    summary_path = paper_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "paper_only": True,
                "order_count": len(broker.orders),
                "fill_count": len(broker.fills),
                "ending_equity": current_equity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "orders": orders_path,
        "fills": fills_path,
        "summary": summary_path,
    }
