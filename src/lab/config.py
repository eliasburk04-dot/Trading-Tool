from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import os
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from src.lab.timeframes import Timeframe, default_frequency_guardrails, parse_timeframes


DEFAULT_CONFIG: Dict[str, Any] = {
    "research": {
        "last_years": 3,
        "start_date": None,
        "end_date": None,
        "interval": "1d",
        "timeframes": ["D1", "H1"],
        "seed": 42,
        "report_dir": "reports",
    },
    "data": {
        "provider": "yahoo",
        "cache_dir": "data/cache",
        "instruments": {
            "Gold": {"primary": "GC=F", "alternates": ["GLD"]},
            "Nasdaq": {"primary": "^IXIC", "alternates": ["QQQ"]},
            "SP500": {"primary": "^GSPC", "alternates": ["SPY"]},
        },
    },
    "sessions": {
        "default": {
            "timezone": "America/New_York",
            "session_open": "09:30",
            "session_close": "16:00",
            "trade_start": "09:45",
            "trade_end": "15:30",
            "flatten_time": "15:55",
            "allow_overnight": False,
        }
    },
    "risk": {
        "initial_capital": 100_000.0,
        "position_fraction": 0.20,
        "max_exposure": 0.20,
        "max_notional": 25_000.0,
        "transaction_cost_bps": 2.0,
        "slippage_bps": 1.0,
        "spread_bps": 1.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "trailing_stop_pct": 0.01,
        "max_drawdown_circuit_breaker": 0.20,
        "max_concurrent_trades": 1,
        "cooldown_bars_after_loss": 3,
        "max_trades_per_day": 4,
        "daily_loss_limit_pct": 0.02,
        "max_consecutive_losses_per_day": 2,
        "flatten_end_of_day": True,
        "allow_overnight": False,
        "time_stop_bars": 8,
    },
    "walk_forward": {
        "train_months": 12,
        "test_months": 3,
        "step_months": 3,
        "purge_bars": 2,
        "embargo_bars": 2,
    },
    "optimization": {
        "parameter_budget_per_strategy": 8,
        "min_samples_before_stop": 8,
        "bootstrap_samples": 100,
        "slippage_jitter_bps": 2.0,
    },
    "guardrails": {
        "min_trade_count": 8,
        "max_drawdown": 0.25,
        "min_out_of_sample_score": 0.10,
        "min_oos_is_ratio": 0.50,
        "baseline_margin": 0.0,
        "frequency": default_frequency_guardrails(),
    },
    "scoring": {
        "weights": {
            "out_of_sample": 0.30,
            "profit_factor": 0.12,
            "expectancy": 0.12,
            "drawdown": 0.12,
            "stability": 0.10,
            "frequency": 0.10,
            "baseline_margin": 0.08,
            "parameter_robustness": 0.04,
            "complexity_penalty": 0.02,
        }
    },
    "strategies": {
        "enabled": [
            "ma_crossover",
            "donchian_breakout",
            "atr_trend_filter",
            "rsi_reversion",
            "bollinger_reversion",
            "roc_momentum",
            "atr_band_breakout",
            "vwap_mean_reversion",
            "opening_range_breakout",
            "trend_pullback",
            "squeeze_release",
            "intraday_rsi_reversion",
        ]
    },
    "paper": {
        "replay_days": 60,
    },
    "features": {
        "enable_live_trading": False,
    },
}

SAFE_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9=^._-]+$")


def deep_merge(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    current = config
    keys = list(path)
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_env_overrides(config: MutableMapping[str, Any], prefix: str = "LAB_") -> MutableMapping[str, Any]:
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        _set_nested(config, env_key[len(prefix):].lower().split("__"), yaml.safe_load(raw_value))
    return config


def resolve_research_window(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    research = config["research"]
    if research.get("start_date") and research.get("end_date"):
        return config
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * int(research.get("last_years", 3)))
    research["start_date"] = start_date.isoformat()
    research["end_date"] = end_date.isoformat()
    return config


def _resolve_timeframes(research: MutableMapping[str, Any]) -> None:
    raw_timeframes = research.get("timeframes")
    if not raw_timeframes:
        raw_timeframes = [Timeframe.from_value(research.get("interval", "1d")).value]
    research["timeframes"] = [timeframe.value for timeframe in parse_timeframes(raw_timeframes)]
    research["interval"] = Timeframe.from_value(research["timeframes"][0]).yahoo_interval


def validate_config(config: Mapping[str, Any]) -> None:
    research = config["research"]
    risk = config["risk"]
    walk_forward = config["walk_forward"]
    guardrails = config["guardrails"]

    start_date = date.fromisoformat(str(research["start_date"]))
    end_date = date.fromisoformat(str(research["end_date"]))
    if start_date >= end_date:
        raise ValueError("research.start_date must be earlier than research.end_date")

    timeframes = parse_timeframes(research["timeframes"])
    if not timeframes:
        raise ValueError("At least one timeframe must be configured")

    for instrument_name, instrument_config in config["data"]["instruments"].items():
        symbols = [instrument_config["primary"]] + list(instrument_config.get("alternates", []))
        for symbol in symbols:
            if not SAFE_SYMBOL_PATTERN.match(symbol):
                raise ValueError("Invalid symbol in %s: %s" % (instrument_name, symbol))

    for key in (
        "position_fraction",
        "max_exposure",
        "stop_loss_pct",
        "take_profit_pct",
        "daily_loss_limit_pct",
    ):
        value = float(risk[key])
        if value <= 0 or value > 1:
            raise ValueError("risk.%s must be within (0, 1]" % key)

    for key in ("transaction_cost_bps", "slippage_bps", "spread_bps", "initial_capital", "max_notional"):
        if float(risk[key]) < 0:
            raise ValueError("risk.%s must be non-negative" % key)

    if int(risk["max_concurrent_trades"]) < 1 or int(risk["max_trades_per_day"]) < 1:
        raise ValueError("risk trade limits must be at least 1")
    if int(risk["cooldown_bars_after_loss"]) < 0 or int(risk["time_stop_bars"]) < 0:
        raise ValueError("risk cooldown/time stop values must be non-negative")
    if int(risk["max_consecutive_losses_per_day"]) < 1:
        raise ValueError("risk.max_consecutive_losses_per_day must be at least 1")

    for key in ("train_months", "test_months", "step_months", "purge_bars", "embargo_bars"):
        if int(walk_forward[key]) < 0:
            raise ValueError("walk_forward.%s must be non-negative" % key)
    for key in ("train_months", "test_months", "step_months"):
        if int(walk_forward[key]) == 0:
            raise ValueError("walk_forward.%s must be greater than 0" % key)

    if float(guardrails["max_drawdown"]) <= 0 or float(guardrails["max_drawdown"]) > 1:
        raise ValueError("guardrails.max_drawdown must be within (0, 1]")
    if float(guardrails["min_oos_is_ratio"]) < 0 or float(guardrails["baseline_margin"]) < 0:
        raise ValueError("guardrails ratio/margin must be non-negative")

    if config["features"].get("enable_live_trading", False):
        raise ValueError("features.enable_live_trading must remain false in this MVP")


def load_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("config file must contain a mapping")
        deep_merge(config, loaded)
    if overrides:
        deep_merge(config, overrides)
    apply_env_overrides(config)
    resolve_research_window(config)
    _resolve_timeframes(config["research"])
    validate_config(config)
    return config
