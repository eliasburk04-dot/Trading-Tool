from __future__ import annotations

from typing import Mapping


def compute_stop_levels(entry_price: float, risk_config: Mapping[str, float]) -> tuple[float, float]:
    stop_price = entry_price * (1.0 - float(risk_config["stop_loss_pct"]))
    take_profit = entry_price * (1.0 + float(risk_config["take_profit_pct"]))
    return stop_price, take_profit
