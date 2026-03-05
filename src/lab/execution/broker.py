from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import os

import pandas as pd


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

    def place_order(self, order: Order, fill_price: float, fill_time: pd.Timestamp) -> Fill:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_fills(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.orders: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []

    def place_order(self, order: Order, fill_price: float, fill_time: pd.Timestamp) -> Fill:
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

    def cancel_order(self, order_id: str) -> None:
        self.orders.append({"order_id": order_id, "status": "cancelled", "paper_only": True})

    def get_positions(self) -> List[Dict[str, Any]]:
        return []

    def get_fills(self) -> List[Dict[str, Any]]:
        return list(self.fills)


class LiveBrokerAdapter(BrokerAdapter):
    paper_only = False

    def __init__(self, config: Dict[str, Any]) -> None:
        enabled = bool(config.get("features", {}).get("enable_live_trading", False))
        allowlisted = os.getenv("LAB_ENABLE_LIVE_TRADING", "").lower() in {"1", "true", "yes"}
        if not enabled or not allowlisted:
            raise RuntimeError("Live trading is hard-disabled. Enable via config and LAB_ENABLE_LIVE_TRADING for future work.")

    def place_order(self, order: Order, fill_price: float, fill_time: pd.Timestamp) -> Fill:
        raise RuntimeError("Live broker integration is not implemented in this MVP")

    def cancel_order(self, order_id: str) -> None:
        raise RuntimeError("Live broker integration is not implemented in this MVP")

    def get_positions(self) -> List[Dict[str, Any]]:
        raise RuntimeError("Live broker integration is not implemented in this MVP")

    def get_fills(self) -> List[Dict[str, Any]]:
        raise RuntimeError("Live broker integration is not implemented in this MVP")
