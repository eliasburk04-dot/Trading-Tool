from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import pandas as pd

from src.lab.indicators import (
    annualized_volatility,
    bollinger_bands,
    compute_atr,
    compute_rsi,
    rate_of_change,
)
from src.lab.timeframes import Timeframe


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    complexity: float = 0.15

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        raise NotImplementedError

    def validate_params(self, params: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1]

    def session_rules(self) -> Dict[str, Any]:
        return {
            "trade_start": "09:45",
            "trade_end": "15:30",
            "flatten_time": "15:55",
            "allow_overnight": False,
        }

    def _frame(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["entry"] = False
        signals["exit"] = False
        signals["time_exit_bars"] = 0
        return signals

    def complexity_penalty(self, params: Mapping[str, Any]) -> float:
        return self.complexity + (len(params) * 0.01)


class MovingAverageCrossoverStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("ma_crossover", complexity=0.10)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "fast_window": {"type": "int", "min": 5, "max": 30},
            "slow_window": {"type": "int", "min": 20, "max": 120},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["fast_window"]) >= int(params["slow_window"]):
            raise ValueError("ma_crossover requires fast_window < slow_window")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        fast = data["Close"].rolling(window=int(params["fast_window"]), min_periods=int(params["fast_window"])).mean()
        slow = data["Close"].rolling(window=int(params["slow_window"]), min_periods=int(params["slow_window"])).mean()
        regime = (fast > slow).fillna(False).astype(bool)
        previous_regime = regime.shift(1, fill_value=False).astype(bool)
        signals = self._frame(data)
        signals["entry"] = regime & ~previous_regime
        signals["exit"] = (~regime) & previous_regime
        return signals


class DonchianBreakoutStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("donchian_breakout", complexity=0.12)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "entry_lookback": {"type": "int", "min": 10, "max": 80},
            "exit_lookback": {"type": "int", "min": 5, "max": 40},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["entry_lookback"]) <= int(params["exit_lookback"]):
            raise ValueError("donchian_breakout requires entry_lookback > exit_lookback")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        entry_high = data["High"].shift(1).rolling(window=int(params["entry_lookback"]), min_periods=int(params["entry_lookback"])).max()
        exit_low = data["Low"].shift(1).rolling(window=int(params["exit_lookback"]), min_periods=int(params["exit_lookback"])).min()
        signals = self._frame(data)
        signals["entry"] = data["Close"] > entry_high
        signals["exit"] = data["Close"] < exit_low
        return signals


class AtrTrendFilterStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("atr_trend_filter", complexity=0.16)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ema_window": {"type": "int", "min": 20, "max": 100},
            "breakout_window": {"type": "int", "min": 10, "max": 40},
            "atr_window": {"type": "int", "min": 10, "max": 30},
            "atr_min_pct": {"type": "float", "min": 0.005, "max": 0.030},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["breakout_window"]) <= 1:
            raise ValueError("atr_trend_filter.breakout_window must be greater than 1")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        ema = data["Close"].ewm(span=int(params["ema_window"]), adjust=False).mean()
        breakout = data["High"].shift(1).rolling(window=int(params["breakout_window"]), min_periods=int(params["breakout_window"])).max()
        atr = compute_atr(data, period=int(params["atr_window"]))
        atr_ratio = atr / data["Close"].replace(0.0, pd.NA)
        trend_regime = data["Close"] > ema
        signals = self._frame(data)
        signals["entry"] = trend_regime & (data["Close"] > breakout) & (atr_ratio > float(params["atr_min_pct"]))
        signals["exit"] = data["Close"] < ema
        return signals


class RsiReversionStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("rsi_reversion", complexity=0.14)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "rsi_period": {"type": "choice", "values": [2, 5, 14]},
            "entry_threshold": {"type": "choice", "values": [10, 15, 20, 25]},
            "exit_threshold": {"type": "choice", "values": [45, 50, 55, 60]},
            "trend_window": {"type": "int", "min": 20, "max": 120},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["entry_threshold"]) >= int(params["exit_threshold"]):
            raise ValueError("rsi_reversion.entry_threshold must be below exit_threshold")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        rsi = compute_rsi(data["Close"], period=int(params["rsi_period"]))
        trend = (
            data["Close"]
            > data["Close"].rolling(window=int(params["trend_window"]), min_periods=int(params["trend_window"])).mean()
        ).fillna(False).astype(bool)
        signals = self._frame(data)
        signals["entry"] = (rsi <= float(params["entry_threshold"])) & trend
        signals["exit"] = (rsi >= float(params["exit_threshold"])) | (~trend)
        return signals


class BollingerReversionStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("bollinger_reversion", complexity=0.12)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "window": {"type": "int", "min": 10, "max": 40},
            "num_std": {"type": "choice", "values": [1.5, 2.0, 2.5]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if float(params["num_std"]) <= 0:
            raise ValueError("bollinger_reversion.num_std must be positive")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        bands = bollinger_bands(data["Close"], window=int(params["window"]), num_std=float(params["num_std"]))
        signals = self._frame(data)
        signals["entry"] = data["Close"] < bands["lower"]
        signals["exit"] = data["Close"] >= bands["middle"]
        return signals


class RocMomentumStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("roc_momentum", complexity=0.15)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "roc_period": {"type": "int", "min": 5, "max": 40},
            "min_roc": {"type": "float", "min": 0.01, "max": 0.12},
            "vol_window": {"type": "int", "min": 10, "max": 40},
            "max_volatility": {"type": "float", "min": 0.10, "max": 0.60},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if float(params["max_volatility"]) <= 0:
            raise ValueError("roc_momentum.max_volatility must be positive")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        roc = rate_of_change(data["Close"], period=int(params["roc_period"]))
        volatility = annualized_volatility(data["Close"], window=int(params["vol_window"]))
        signals = self._frame(data)
        signals["entry"] = (roc >= float(params["min_roc"])) & (volatility <= float(params["max_volatility"]))
        signals["exit"] = (roc <= 0.0) | (volatility > float(params["max_volatility"]) * 1.15)
        return signals


class AtrBandBreakoutStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("atr_band_breakout", complexity=0.16)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "window": {"type": "int", "min": 10, "max": 50},
            "atr_window": {"type": "int", "min": 10, "max": 30},
            "atr_mult": {"type": "choice", "values": [1.0, 1.5, 2.0, 2.5]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["window"]) <= 1:
            raise ValueError("atr_band_breakout.window must be greater than 1")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.D1, Timeframe.H1]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        atr = compute_atr(data, period=int(params["atr_window"]))
        center = data["Close"].rolling(window=int(params["window"]), min_periods=int(params["window"])).mean()
        upper = center + float(params["atr_mult"]) * atr
        signals = self._frame(data)
        signals["entry"] = data["Close"] > upper
        signals["exit"] = data["Close"] < center
        return signals


class VwapMeanReversionStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("vwap_mean_reversion", complexity=0.18)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "atr_window": {"type": "int", "min": 3, "max": 12},
            "deviation_atr": {"type": "choice", "values": [0.4, 0.6, 0.8, 1.0]},
            "stall_bars": {"type": "choice", "values": [1, 2, 3]},
            "time_exit_bars": {"type": "choice", "values": [3, 4, 6, 8]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["time_exit_bars"]) <= 0:
            raise ValueError("vwap_mean_reversion.time_exit_bars must be positive")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.H1, Timeframe.M15, Timeframe.M5]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3.0
        cumulative_value = typical_price.mul(data["Volume"]).groupby(data.index.normalize()).cumsum()
        cumulative_volume = data["Volume"].groupby(data.index.normalize()).cumsum().replace(0, pd.NA)
        vwap = pd.to_numeric(cumulative_value / cumulative_volume, errors="coerce")
        atr = compute_atr(data, int(params["atr_window"]))
        deviation = vwap - data["Close"]
        stall = data["Close"].diff().rolling(int(params["stall_bars"]), min_periods=int(params["stall_bars"])).max() >= 0.0
        signals = self._frame(data)
        signals["entry"] = ((deviation >= float(params["deviation_atr"]) * atr) & stall.fillna(False)).fillna(False).astype(bool)
        signals["exit"] = (data["Close"] >= vwap).fillna(False).astype(bool)
        signals["time_exit_bars"] = int(params["time_exit_bars"])
        return signals


class OpeningRangeBreakoutStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("opening_range_breakout", complexity=0.20)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "opening_range_bars": {"type": "choice", "values": [2, 3, 4, 6]},
            "breakout_buffer_atr": {"type": "choice", "values": [0.0, 0.1, 0.2]},
            "atr_window": {"type": "int", "min": 3, "max": 12},
            "time_exit_bars": {"type": "choice", "values": [4, 6, 8, 12]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["opening_range_bars"]) <= 0:
            raise ValueError("opening_range_breakout.opening_range_bars must be positive")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.H1, Timeframe.M15, Timeframe.M5]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        atr = compute_atr(data, int(params["atr_window"]))
        day_group = data.index.normalize()
        opening_high = data["High"].groupby(day_group).transform(
            lambda series: series.iloc[: int(params["opening_range_bars"])].max()
        )
        opening_complete = data.groupby(day_group).cumcount() >= int(params["opening_range_bars"])
        trigger = opening_high + float(params["breakout_buffer_atr"]) * atr
        signals = self._frame(data)
        signals["entry"] = opening_complete & (data["Close"] > trigger)
        signals["exit"] = data["Close"] < opening_high
        signals["time_exit_bars"] = int(params["time_exit_bars"])
        return signals


class TrendPullbackStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("trend_pullback", complexity=0.22)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ema_fast": {"type": "int", "min": 5, "max": 15},
            "ema_slow": {"type": "int", "min": 20, "max": 50},
            "atr_window": {"type": "int", "min": 5, "max": 20},
            "pullback_atr": {"type": "choice", "values": [0.2, 0.3, 0.4]},
            "time_exit_bars": {"type": "choice", "values": [4, 6, 8]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["ema_fast"]) >= int(params["ema_slow"]):
            raise ValueError("trend_pullback requires ema_fast < ema_slow")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.H1, Timeframe.M15, Timeframe.M5]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        ema_fast = data["Close"].ewm(span=int(params["ema_fast"]), adjust=False).mean()
        ema_slow = data["Close"].ewm(span=int(params["ema_slow"]), adjust=False).mean()
        atr = compute_atr(data, int(params["atr_window"]))
        trend = ema_fast > ema_slow
        pullback = data["Close"] <= ema_fast - float(params["pullback_atr"]) * atr
        confirmation = data["Close"] > data["Open"]
        signals = self._frame(data)
        signals["entry"] = trend & pullback & confirmation
        signals["exit"] = data["Close"] < ema_slow
        signals["time_exit_bars"] = int(params["time_exit_bars"])
        return signals


class SqueezeReleaseStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("squeeze_release", complexity=0.22)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "window": {"type": "int", "min": 8, "max": 24},
            "atr_window": {"type": "int", "min": 5, "max": 20},
            "squeeze_width": {"type": "choice", "values": [0.008, 0.012, 0.016]},
            "time_exit_bars": {"type": "choice", "values": [4, 6, 8]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if float(params["squeeze_width"]) <= 0:
            raise ValueError("squeeze_release.squeeze_width must be positive")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.H1, Timeframe.M15, Timeframe.M5]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        bands = bollinger_bands(data["Close"], window=int(params["window"]), num_std=2.0)
        width = (bands["upper"] - bands["lower"]) / bands["middle"].replace(0.0, pd.NA)
        atr = compute_atr(data, int(params["atr_window"]))
        upper_keltner = bands["middle"] + 1.5 * atr
        squeeze = (width <= float(params["squeeze_width"])) & (bands["upper"] <= upper_keltner)
        release = squeeze.shift(1, fill_value=False) & (data["Close"] > bands["upper"])
        signals = self._frame(data)
        signals["entry"] = release
        signals["exit"] = data["Close"] < bands["middle"]
        signals["time_exit_bars"] = int(params["time_exit_bars"])
        return signals


class IntradayRsiReversionStrategy(StrategyDefinition):
    def __init__(self) -> None:
        super().__init__("intraday_rsi_reversion", complexity=0.18)

    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "rsi_period": {"type": "choice", "values": [2, 3]},
            "entry_threshold": {"type": "choice", "values": [8, 12, 15]},
            "exit_threshold": {"type": "choice", "values": [45, 50, 55]},
            "range_window": {"type": "int", "min": 6, "max": 20},
            "time_exit_bars": {"type": "choice", "values": [3, 4, 6]},
        }

    def validate_params(self, params: Mapping[str, Any]) -> None:
        if int(params["entry_threshold"]) >= int(params["exit_threshold"]):
            raise ValueError("intraday_rsi_reversion.entry_threshold must be below exit_threshold")

    def supports_timeframes(self) -> List[Timeframe]:
        return [Timeframe.H1, Timeframe.M15, Timeframe.M5]

    def generate_signals(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        self.validate_params(params)
        rsi = compute_rsi(data["Close"], int(params["rsi_period"]))
        returns = data["Close"].pct_change()
        range_regime = returns.rolling(int(params["range_window"]), min_periods=int(params["range_window"])).std(ddof=0) < returns.rolling(
            int(params["range_window"]), min_periods=int(params["range_window"])
        ).mean().abs().add(0.004)
        signals = self._frame(data)
        signals["entry"] = (rsi <= float(params["entry_threshold"])) & range_regime.fillna(False)
        signals["exit"] = (rsi >= float(params["exit_threshold"])) | (~range_regime.fillna(False))
        signals["time_exit_bars"] = int(params["time_exit_bars"])
        return signals


REGISTRY = {
    "ma_crossover": MovingAverageCrossoverStrategy(),
    "donchian_breakout": DonchianBreakoutStrategy(),
    "atr_trend_filter": AtrTrendFilterStrategy(),
    "rsi_reversion": RsiReversionStrategy(),
    "bollinger_reversion": BollingerReversionStrategy(),
    "roc_momentum": RocMomentumStrategy(),
    "atr_band_breakout": AtrBandBreakoutStrategy(),
    "vwap_mean_reversion": VwapMeanReversionStrategy(),
    "opening_range_breakout": OpeningRangeBreakoutStrategy(),
    "trend_pullback": TrendPullbackStrategy(),
    "squeeze_release": SqueezeReleaseStrategy(),
    "intraday_rsi_reversion": IntradayRsiReversionStrategy(),
}


def get_strategy(strategy_id: str) -> StrategyDefinition:
    if strategy_id not in REGISTRY:
        raise KeyError("Unknown strategy: %s" % strategy_id)
    return REGISTRY[strategy_id]


def list_strategies() -> Dict[str, StrategyDefinition]:
    return dict(REGISTRY)


def sample_parameters(strategy: StrategyDefinition, seed_value: int) -> Dict[str, Any]:
    rng_value = max(seed_value, 1)
    params: Dict[str, Any] = {}
    for position, (name, spec) in enumerate(strategy.param_space().items(), start=1):
        kind = spec["type"]
        if kind == "choice":
            values = list(spec["values"])
            params[name] = values[(rng_value + position) % len(values)]
        elif kind == "int":
            minimum = int(spec["min"])
            maximum = int(spec["max"])
            span = maximum - minimum
            params[name] = minimum + ((rng_value * (position + 3)) % (span + 1))
        elif kind == "float":
            minimum = float(spec["min"])
            maximum = float(spec["max"])
            ratio = ((rng_value * (position + 7)) % 10_000) / 10_000.0
            params[name] = round(minimum + ratio * (maximum - minimum), 4)
        else:
            raise ValueError("Unsupported parameter type: %s" % kind)
    strategy.validate_params(params)
    return params
