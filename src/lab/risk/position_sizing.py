from __future__ import annotations

from typing import Mapping


def calculate_position_notional(equity: float, risk_config: Mapping[str, float]) -> float:
    max_fraction = min(float(risk_config["position_fraction"]), float(risk_config["max_exposure"]))
    notional = equity * max_fraction
    raw_max_notional = risk_config.get("max_notional", notional)
    max_notional = notional if raw_max_notional is None else float(raw_max_notional)
    return max(0.0, min(notional, max_notional))
