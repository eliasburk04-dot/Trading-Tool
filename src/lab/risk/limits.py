from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass
class DailyRiskState:
    session_date: str | None
    realized_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_taken: int = 0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


def pre_trade_risk_check(
    timestamp: pd.Timestamp,
    state,
    risk_config: Mapping[str, float],
    proposed_notional: float,
) -> RiskDecision:
    if proposed_notional <= 0:
        return RiskDecision(False, "invalid_notional")
    if proposed_notional > float(risk_config.get("max_notional", proposed_notional)):
        return RiskDecision(False, "max_notional")
    if state.risk.trades_taken >= int(risk_config.get("max_trades_per_day", 999999)):
        return RiskDecision(False, "max_trades_per_day")
    daily_loss_limit = float(risk_config.get("daily_loss_limit_pct", 0.0)) * float(risk_config["initial_capital"])
    if daily_loss_limit > 0 and state.risk.realized_pnl <= -daily_loss_limit:
        return RiskDecision(False, "daily_loss_limit")
    if state.risk.consecutive_losses >= int(risk_config.get("max_consecutive_losses_per_day", 999999)):
        return RiskDecision(False, "consecutive_losses")
    return RiskDecision(True, "allowed")
