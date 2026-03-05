from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import hashlib
import json
import math
import subprocess

import numpy as np
import pandas as pd
import yaml

from src.lab.backtest import BacktestConfig, run_backtest
from src.lab.data_layer import load_market_data
from src.lab.execution import run_paper_session
from src.lab.metrics import combine_equity_curves, compute_metrics
from src.lab.plots import plot_drawdown_curve, plot_equity_curve, plot_return_distribution, plot_trade_markers
from src.lab.strategies import get_strategy, sample_parameters
from src.lab.timeframes import Timeframe


def build_backtest_config(
    config: Mapping[str, Any],
    timeframe: Timeframe,
    session_rules: Mapping[str, Any],
    initial_capital: float | None = None,
) -> BacktestConfig:
    risk = config["risk"]
    return BacktestConfig(
        initial_capital=float(initial_capital if initial_capital is not None else risk["initial_capital"]),
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
        timeframe=timeframe,
        max_trades_per_day=int(risk["max_trades_per_day"]),
        daily_loss_limit_pct=float(risk["daily_loss_limit_pct"]),
        max_consecutive_losses_per_day=int(risk["max_consecutive_losses_per_day"]),
        flatten_end_of_day=bool(risk["flatten_end_of_day"]),
        allow_overnight=bool(risk["allow_overnight"]),
        time_stop_bars=int(risk["time_stop_bars"]),
        max_notional=float(risk["max_notional"]),
        session_rules=dict(session_rules),
    )


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _candidate_id(strategy_id: str, timeframe: Timeframe, params: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        ("%s|%s|%s" % (strategy_id, timeframe.value, json.dumps(dict(params), sort_keys=True))).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _log_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _bars_for_months(timeframe: Timeframe, months: int) -> int:
    return max(1, months * 21 * timeframe.bars_per_day)


def generate_purged_walk_forward_windows(
    index: pd.Index,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    purge_bars: int,
    embargo_bars: int,
) -> List[Dict[str, Any]]:
    if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
        raise ValueError("train_bars, test_bars, and step_bars must be positive")

    total_bars = len(index)
    windows: List[Dict[str, Any]] = []
    cursor = 0
    while True:
        train_start_idx = cursor
        train_end_idx = train_start_idx + train_bars - 1
        test_start_idx = train_end_idx + purge_bars + embargo_bars + 1
        test_end_idx = test_start_idx + test_bars - 1
        if test_end_idx >= total_bars:
            break
        windows.append(
            {
                "train_start_idx": train_start_idx,
                "train_end_idx": train_end_idx,
                "test_start_idx": test_start_idx,
                "test_end_idx": test_end_idx,
                "train_start": index[train_start_idx],
                "train_end": index[train_end_idx],
                "test_start": index[test_start_idx],
                "test_end": index[test_end_idx],
            }
        )
        cursor += step_bars
    return windows


def _walk_forward_windows(index: pd.Index, timeframe: Timeframe, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    walk_forward = config["walk_forward"]
    windows = generate_purged_walk_forward_windows(
        index=index,
        train_bars=_bars_for_months(timeframe, int(walk_forward["train_months"])),
        test_bars=_bars_for_months(timeframe, int(walk_forward["test_months"])),
        step_bars=_bars_for_months(timeframe, int(walk_forward["step_months"])),
        purge_bars=int(walk_forward.get("purge_bars", 0)),
        embargo_bars=int(walk_forward.get("embargo_bars", 0)),
    )
    if windows:
        return windows

    fallback_train = max(len(index) // 2, 1)
    fallback_test = max((len(index) - fallback_train) // 2, 1)
    if fallback_train > 0 and fallback_test > 0 and len(index) > (fallback_train + fallback_test):
        return generate_purged_walk_forward_windows(
            index=index,
            train_bars=fallback_train,
            test_bars=fallback_test,
            step_bars=fallback_test,
            purge_bars=int(walk_forward.get("purge_bars", 0)),
            embargo_bars=int(walk_forward.get("embargo_bars", 0)),
        )
    return []


def _slice_window(prices: pd.DataFrame, window: Mapping[str, Any], prefix: str) -> pd.DataFrame:
    start_idx = int(window["%s_start_idx" % prefix])
    end_idx = int(window["%s_end_idx" % prefix]) + 1
    return prices.iloc[start_idx:end_idx].copy()


def _fallback_curve(index: pd.Index, initial_capital: float) -> pd.Series:
    anchor = index[0] if len(index) else pd.Timestamp.utcnow().tz_localize(None)
    return pd.Series([initial_capital], index=[anchor])


def _safe_metrics(trades: pd.DataFrame, curve: pd.Series, initial_capital: float, index: pd.Index) -> Dict[str, float]:
    if curve.empty:
        curve = _fallback_curve(index, initial_capital)
    return compute_metrics(trades, curve, initial_capital)


def _serialize_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    serialized: Dict[str, float] = {}
    for key, value in metrics.items():
        serialized[key] = float(value)
    return serialized


def _trades_per_day(trade_count: float, active_days: int) -> float:
    return float(trade_count) / max(float(active_days), 1.0)


def _trade_frequency_score(trade_count: float, active_days: int, timeframe: Timeframe, config: Mapping[str, Any]) -> float:
    frequency = config["guardrails"]["frequency"][timeframe.value]
    min_total = max(float(frequency["min_total_trades"]), 1.0)
    min_daily = max(float(frequency["min_trades_per_day"]), 1e-9)
    total_component = min(1.0, float(trade_count) / min_total)
    daily_component = min(1.0, _trades_per_day(trade_count, active_days) / min_daily)
    return 0.5 * total_component + 0.5 * daily_component


def _stability_score(split_metrics: Sequence[Mapping[str, float]]) -> float:
    if not split_metrics:
        return 0.0

    expectancies = np.array([float(metrics["expectancy"]) for metrics in split_metrics], dtype=float)
    sharpes = np.array([float(metrics["sharpe"]) for metrics in split_metrics], dtype=float)
    expectancy_dispersion = float(np.std(expectancies)) / max(abs(float(np.mean(expectancies))), 0.01)
    sharpe_dispersion = float(np.std(sharpes)) / max(abs(float(np.mean(sharpes))), 0.25)
    score = 1.0 - min(1.0, 0.6 * expectancy_dispersion + 0.4 * sharpe_dispersion)
    return max(0.0, score)


def _baseline_buy_and_hold(prices: pd.DataFrame) -> float:
    if prices.empty:
        return 0.0
    return float(prices["Close"].iloc[-1] / max(float(prices["Close"].iloc[0]), 1e-9) - 1.0)


def _baseline_random_total_return(prices: pd.DataFrame, trade_count: int, avg_hold_bars: int, seed: int) -> float:
    if prices.empty or trade_count <= 0:
        return 0.0

    max_start = len(prices) - max(avg_hold_bars, 1) - 1
    if max_start <= 0:
        return 0.0

    rng = np.random.default_rng(seed)
    sample_returns: List[float] = []
    for _ in range(trade_count):
        start_idx = int(rng.integers(0, max_start + 1))
        end_idx = min(start_idx + max(avg_hold_bars, 1), len(prices) - 1)
        entry_price = float(prices["Close"].iloc[start_idx])
        exit_price = float(prices["Close"].iloc[end_idx])
        sample_returns.append((exit_price / max(entry_price, 1e-9)) - 1.0)
    return float(np.mean(sample_returns) if sample_returns else 0.0)


def _neighbor_parameters(strategy, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    neighbors: List[Dict[str, Any]] = []
    for name, spec in strategy.param_space().items():
        current = params[name]
        if spec["type"] == "choice":
            values = list(spec["values"])
            position = values.index(current)
            for step in (-1, 1):
                neighbor_index = position + step
                if 0 <= neighbor_index < len(values):
                    neighbor = dict(params)
                    neighbor[name] = values[neighbor_index]
                    neighbors.append(neighbor)
        elif spec["type"] == "int":
            for delta in (-1, 1):
                candidate = int(current) + delta
                if int(spec["min"]) <= candidate <= int(spec["max"]):
                    neighbor = dict(params)
                    neighbor[name] = candidate
                    neighbors.append(neighbor)
        elif spec["type"] == "float":
            step = round((float(spec["max"]) - float(spec["min"])) / 10.0, 4)
            if step <= 0:
                continue
            for delta in (-step, step):
                candidate = round(float(current) + delta, 4)
                if float(spec["min"]) <= candidate <= float(spec["max"]):
                    neighbor = dict(params)
                    neighbor[name] = candidate
                    neighbors.append(neighbor)

    unique: List[Dict[str, Any]] = []
    seen = set()
    for neighbor in neighbors:
        key = json.dumps(neighbor, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(neighbor)
        if len(unique) >= 4:
            break
    return unique


def _parameter_robustness(
    strategy_id: str,
    params: Mapping[str, Any],
    timeframe: Timeframe,
    prices: pd.DataFrame,
    session_rules: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    strategy = get_strategy(strategy_id)
    if prices.empty:
        return 0.0

    center_result = run_backtest(
        prices,
        strategy.generate_signals(prices, params),
        build_backtest_config(config, timeframe, session_rules),
        instrument="reference",
    )
    center_score = compute_metrics(center_result.trades, center_result.equity_curve, float(config["risk"]["initial_capital"]))[
        "total_return"
    ]
    neighbor_scores = [center_score]
    for neighbor in _neighbor_parameters(strategy, params):
        try:
            result = run_backtest(
                prices,
                strategy.generate_signals(prices, neighbor),
                build_backtest_config(config, timeframe, session_rules),
                instrument="reference",
            )
            neighbor_scores.append(
                compute_metrics(result.trades, result.equity_curve, float(config["risk"]["initial_capital"]))["total_return"]
            )
        except Exception:
            continue
    if len(neighbor_scores) <= 1:
        return 0.5
    dispersion = float(np.std(neighbor_scores)) / max(abs(float(np.mean(neighbor_scores))), 0.05)
    return max(0.0, 1.0 - min(1.0, dispersion))


def _bootstrap_trade_robustness(trades: pd.DataFrame, samples: int, seed: int) -> Dict[str, float]:
    if trades.empty or samples <= 0:
        return {
            "sample_count": 0.0,
            "p05_total_return": 0.0,
            "p95_max_drawdown": 0.0,
        }

    returns = trades["return_pct"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    total_returns: List[float] = []
    drawdowns: List[float] = []
    for _ in range(samples):
        sample = rng.choice(returns, size=len(returns), replace=True)
        equity = pd.Series((1.0 + sample)).cumprod()
        running_peak = equity.cummax()
        drawdown = (equity / running_peak) - 1.0
        total_returns.append(float(equity.iloc[-1] - 1.0))
        drawdowns.append(float(drawdown.min()))
    return {
        "sample_count": float(samples),
        "p05_total_return": float(np.percentile(total_returns, 5)),
        "p95_max_drawdown": float(np.percentile(drawdowns, 95)),
    }


def _score_metrics(
    metrics: Mapping[str, float],
    stability: float,
    frequency_score: float,
    baseline_margin: float,
    parameter_robustness: float,
    complexity_penalty: float,
    config: Mapping[str, Any],
) -> float:
    weights = config["scoring"]["weights"]
    drawdown_limit = float(config["guardrails"]["max_drawdown"])

    profit_factor = float(metrics["profit_factor"])
    if math.isinf(profit_factor):
        profit_factor_term = 1.0
    else:
        profit_factor_term = max(-1.0, min(1.0, (profit_factor - 1.0) / 2.0))

    expectancy_term = float(np.tanh(float(metrics["expectancy"]) * 30.0))
    drawdown_term = max(0.0, 1.0 - abs(float(metrics["max_drawdown"])) / max(drawdown_limit, 1e-9))
    out_of_sample_term = float(np.tanh(float(metrics["total_return"]) * 4.0) + np.tanh(float(metrics["sharpe"]) / 2.0))
    baseline_term = float(np.tanh(baseline_margin * 10.0))
    complexity_term = max(0.0, min(1.0, 1.0 - complexity_penalty))

    return (
        float(weights["out_of_sample"]) * out_of_sample_term
        + float(weights["profit_factor"]) * profit_factor_term
        + float(weights["expectancy"]) * expectancy_term
        + float(weights["drawdown"]) * drawdown_term
        + float(weights["stability"]) * stability
        + float(weights["frequency"]) * frequency_score
        + float(weights["baseline_margin"]) * baseline_term
        + float(weights["parameter_robustness"]) * parameter_robustness
        + float(weights["complexity_penalty"]) * complexity_term
    )


def evaluate_guardrails(
    out_of_sample_metrics: Mapping[str, float],
    split_trade_counts: Sequence[int],
    timeframe: Timeframe,
    config: Mapping[str, Any],
    in_sample_score: float,
    out_of_sample_score: float,
    active_days: int,
    baseline_margin: float,
) -> tuple[bool, Dict[str, Any]]:
    guardrails = config["guardrails"]
    frequency = guardrails["frequency"][timeframe.value]

    trade_count = float(out_of_sample_metrics["trade_count"])
    trades_per_day = _trades_per_day(trade_count, active_days)

    trade_count_ok = trade_count >= float(frequency.get("min_total_trades", guardrails.get("min_trade_count", 0.0)))
    drawdown_ok = abs(float(out_of_sample_metrics["max_drawdown"])) <= float(guardrails["max_drawdown"])
    oos_score_ok = float(out_of_sample_score) >= float(guardrails["min_out_of_sample_score"])
    ratio_ok = True
    if float(in_sample_score) > 0:
        ratio_ok = (float(out_of_sample_score) / max(float(in_sample_score), 1e-9)) >= float(guardrails["min_oos_is_ratio"])
    frequency_ok = trades_per_day >= float(frequency["min_trades_per_day"])
    split_ok = all(int(count) >= int(frequency["min_oos_trades_per_split"]) for count in split_trade_counts) if split_trade_counts else False
    baseline_floor = float(guardrails.get("baseline_margin", 0.0))
    baseline_ok = float(baseline_margin) >= (baseline_floor - 1e-4)

    notes = {
        "trade_count_ok": trade_count_ok,
        "drawdown_ok": drawdown_ok,
        "oos_score_ok": oos_score_ok,
        "ratio_ok": ratio_ok,
        "frequency_ok": frequency_ok,
        "split_trade_count_ok": split_ok,
        "baseline_ok": baseline_ok,
        "trade_count": trade_count,
        "trades_per_day": trades_per_day,
        "active_days": int(active_days),
        "split_trade_counts": [int(count) for count in split_trade_counts],
        "baseline_margin": float(baseline_margin),
    }
    return all(
        (
            trade_count_ok,
            drawdown_ok,
            oos_score_ok,
            ratio_ok,
            frequency_ok,
            split_ok,
            baseline_ok,
        )
    ), notes


def evaluate_strategy_candidate(
    strategy_id: str,
    timeframe: Timeframe,
    params: Mapping[str, Any],
    market_data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    strategy = get_strategy(strategy_id)
    base_initial_capital = float(config["risk"]["initial_capital"])

    per_instrument: Dict[str, Any] = {}
    all_train_trades: List[pd.DataFrame] = []
    all_test_trades: List[pd.DataFrame] = []
    train_curves: List[pd.Series] = []
    test_curves: List[pd.Series] = []
    split_trade_counts: Dict[int, int] = {}
    split_metrics: List[Dict[str, float]] = []
    oos_active_days = set()
    buy_hold_returns: List[float] = []
    random_baselines: List[float] = []
    representative_prices: pd.DataFrame | None = None
    representative_session_rules: Mapping[str, Any] | None = None

    for instrument, payloads in market_data.items():
        if timeframe.value not in payloads:
            continue

        payload = payloads[timeframe.value]
        prices = payload["prices"]
        session_rules = dict(payload["session_rules"])
        session_rules.update(strategy.session_rules())
        if representative_prices is None:
            representative_prices = prices
            representative_session_rules = session_rules

        windows = _walk_forward_windows(prices.index, timeframe, config)
        if not windows:
            continue

        instrument_train_curves: List[pd.Series] = []
        instrument_test_curves: List[pd.Series] = []
        instrument_train_trades: List[pd.DataFrame] = []
        instrument_test_trades: List[pd.DataFrame] = []
        test_dates = set()

        for split_index, window in enumerate(windows):
            train_prices = _slice_window(prices, window, "train")
            test_prices = _slice_window(prices, window, "test")
            if len(train_prices) < max(10, timeframe.bars_per_day * 4) or len(test_prices) < max(5, timeframe.bars_per_day * 2):
                continue

            train_result = run_backtest(
                train_prices,
                strategy.generate_signals(train_prices, params),
                build_backtest_config(config, timeframe, session_rules, initial_capital=base_initial_capital),
                instrument=instrument,
            )
            test_result = run_backtest(
                test_prices,
                strategy.generate_signals(test_prices, params),
                build_backtest_config(config, timeframe, session_rules, initial_capital=base_initial_capital),
                instrument=instrument,
            )

            instrument_train_curves.append(train_result.equity_curve)
            instrument_test_curves.append(test_result.equity_curve)
            if not train_result.trades.empty:
                instrument_train_trades.append(train_result.trades)
                all_train_trades.append(train_result.trades)
            if not test_result.trades.empty:
                instrument_test_trades.append(test_result.trades)
                all_test_trades.append(test_result.trades)

            split_trade_counts[split_index] = split_trade_counts.get(split_index, 0) + int(len(test_result.trades))
            split_metrics.append(
                _safe_metrics(test_result.trades, test_result.equity_curve, base_initial_capital, test_prices.index)
            )
            test_dates.update({timestamp.date().isoformat() for timestamp in test_prices.index})

        train_curve = combine_equity_curves(instrument_train_curves) * base_initial_capital if instrument_train_curves else pd.Series(dtype=float)
        test_curve = combine_equity_curves(instrument_test_curves) * base_initial_capital if instrument_test_curves else pd.Series(dtype=float)
        train_trades = pd.concat(instrument_train_trades, ignore_index=True) if instrument_train_trades else pd.DataFrame()
        test_trades = pd.concat(instrument_test_trades, ignore_index=True) if instrument_test_trades else pd.DataFrame()

        buy_hold_return = _baseline_buy_and_hold(prices)
        buy_hold_returns.append(buy_hold_return)
        avg_hold_bars = int(round(float(test_trades["bars_held"].mean()))) if not test_trades.empty else max(1, timeframe.bars_per_day // 2)
        random_baselines.append(
            _baseline_random_total_return(prices, int(len(test_trades)), avg_hold_bars, int(config["research"]["seed"]) + len(random_baselines))
        )
        oos_active_days.update(test_dates)

        per_instrument[instrument] = {
            "symbol": payload["symbol"],
            "coverage": payload["coverage"],
            "session_rules": session_rules,
            "buy_and_hold_return": buy_hold_return,
            "in_sample": _safe_metrics(train_trades, train_curve, base_initial_capital, prices.index),
            "out_of_sample": _safe_metrics(test_trades, test_curve, base_initial_capital, prices.index),
        }
        if not train_curve.empty:
            train_curves.append(train_curve)
        if not test_curve.empty:
            test_curves.append(test_curve)

    if not per_instrument:
        raise ValueError("No matching market data for timeframe %s" % timeframe.value)

    combined_train_curve = combine_equity_curves(train_curves) * base_initial_capital if train_curves else pd.Series(dtype=float)
    combined_test_curve = combine_equity_curves(test_curves) * base_initial_capital if test_curves else pd.Series(dtype=float)
    train_trades = pd.concat(all_train_trades, ignore_index=True) if all_train_trades else pd.DataFrame()
    test_trades = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    reference_index = next(iter(per_instrument.values()))["out_of_sample"]
    in_sample_metrics = _safe_metrics(
        train_trades,
        combined_train_curve,
        base_initial_capital,
        next(iter(market_data.values()))[timeframe.value]["prices"].index,
    )
    out_of_sample_metrics = _safe_metrics(
        test_trades,
        combined_test_curve,
        base_initial_capital,
        next(iter(market_data.values()))[timeframe.value]["prices"].index,
    )
    del reference_index

    active_days = len(oos_active_days)
    frequency_score = _trade_frequency_score(out_of_sample_metrics["trade_count"], active_days, timeframe, config)
    baseline_margin = float(out_of_sample_metrics["expectancy"]) - float(np.mean(random_baselines) if random_baselines else 0.0)
    stability_score = _stability_score(split_metrics)
    complexity_penalty = get_strategy(strategy_id).complexity_penalty(params)
    parameter_robustness = _parameter_robustness(
        strategy_id,
        params,
        timeframe,
        representative_prices if representative_prices is not None else pd.DataFrame(),
        representative_session_rules or {},
        config,
    )
    in_sample_score = _score_metrics(
        in_sample_metrics,
        stability_score,
        frequency_score,
        baseline_margin,
        parameter_robustness,
        complexity_penalty,
        config,
    )
    out_of_sample_score = _score_metrics(
        out_of_sample_metrics,
        stability_score,
        frequency_score,
        baseline_margin,
        parameter_robustness,
        complexity_penalty,
        config,
    )
    passes_guardrails, guardrail_notes = evaluate_guardrails(
        out_of_sample_metrics=out_of_sample_metrics,
        split_trade_counts=list(split_trade_counts.values()),
        timeframe=timeframe,
        config=config,
        in_sample_score=in_sample_score,
        out_of_sample_score=out_of_sample_score,
        active_days=active_days,
        baseline_margin=baseline_margin,
    )

    return {
        "candidate_id": _candidate_id(strategy_id, timeframe, params),
        "strategy_id": strategy_id,
        "timeframe": timeframe.value,
        "parameters": dict(params),
        "in_sample_metrics": in_sample_metrics,
        "out_of_sample_metrics": out_of_sample_metrics,
        "in_sample_score": float(in_sample_score),
        "out_of_sample_score": float(out_of_sample_score),
        "stability_score": float(stability_score),
        "frequency_score": float(frequency_score),
        "parameter_robustness": float(parameter_robustness),
        "complexity_penalty": float(complexity_penalty),
        "baseline_margin": float(baseline_margin),
        "buy_and_hold_return": float(np.mean(buy_hold_returns) if buy_hold_returns else 0.0),
        "random_baseline_return": float(np.mean(random_baselines) if random_baselines else 0.0),
        "split_trade_counts": [int(count) for count in split_trade_counts.values()],
        "active_days": int(active_days),
        "passes_guardrails": bool(passes_guardrails),
        "guardrail_notes": guardrail_notes,
        "per_instrument": per_instrument,
    }


def _full_sample_frequency_metrics(trades: pd.DataFrame, curve: pd.Series) -> Dict[str, float]:
    if trades.empty:
        return {
            "avg_holding_bars": 0.0,
            "avg_holding_hours": 0.0,
            "time_in_market": 0.0,
        }
    avg_holding_bars = float(trades["bars_held"].mean())
    avg_holding_hours = 0.0
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        avg_holding_hours = float((trades["exit_time"] - trades["entry_time"]).dt.total_seconds().mean() / 3600.0)
    span_seconds = max((curve.index[-1] - curve.index[0]).total_seconds(), 1.0)
    holding_seconds = float((trades["exit_time"] - trades["entry_time"]).dt.total_seconds().sum())
    return {
        "avg_holding_bars": avg_holding_bars,
        "avg_holding_hours": avg_holding_hours,
        "time_in_market": holding_seconds / span_seconds,
    }


def evaluate_full_sample(
    candidate: Mapping[str, Any],
    market_data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
    report_dir: Path,
) -> Dict[str, Any]:
    strategy = get_strategy(candidate["strategy_id"])
    timeframe = Timeframe.from_value(candidate["timeframe"])
    base_initial_capital = float(config["risk"]["initial_capital"])
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []
    portfolio_curves: List[pd.Series] = []
    all_trades: List[pd.DataFrame] = []
    per_instrument: Dict[str, Any] = {}

    for instrument, payloads in market_data.items():
        if timeframe.value not in payloads:
            continue
        payload = payloads[timeframe.value]
        prices = payload["prices"]
        session_rules = dict(payload["session_rules"])
        session_rules.update(strategy.session_rules())
        result = run_backtest(
            prices,
            strategy.generate_signals(prices, candidate["parameters"]),
            build_backtest_config(config, timeframe, session_rules),
            instrument=instrument,
        )
        metrics = compute_metrics(result.trades, result.equity_curve, base_initial_capital)
        metrics.update(_full_sample_frequency_metrics(result.trades, result.equity_curve))
        per_instrument[instrument] = {
            "symbol": payload["symbol"],
            "coverage": payload["coverage"],
            "session_rules": session_rules,
            "metrics": metrics,
        }
        portfolio_curves.append(result.equity_curve)
        if not result.trades.empty:
            all_trades.append(result.trades)

        safe_name = instrument.lower().replace(" ", "_")
        plot_paths.append(str(plot_equity_curve(result.equity_curve, "%s %s Equity" % (instrument, timeframe.value), plots_dir / ("%s_equity.png" % safe_name))))
        plot_paths.append(str(plot_drawdown_curve(result.equity_curve, "%s %s Drawdown" % (instrument, timeframe.value), plots_dir / ("%s_drawdown.png" % safe_name))))
        plot_paths.append(str(plot_trade_markers(prices, result.trades, "%s %s Trades" % (instrument, timeframe.value), plots_dir / ("%s_trades.png" % safe_name))))
        if not result.trades.empty:
            plot_paths.append(str(plot_return_distribution(result.trades, "%s %s Returns" % (instrument, timeframe.value), plots_dir / ("%s_returns.png" % safe_name))))

    combined_curve = combine_equity_curves(portfolio_curves) * base_initial_capital if portfolio_curves else pd.Series(dtype=float)
    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    aggregate_metrics = compute_metrics(
        combined_trades,
        combined_curve if not combined_curve.empty else _fallback_curve(pd.Index([pd.Timestamp.utcnow()]), base_initial_capital),
        base_initial_capital,
    )
    aggregate_metrics.update(_full_sample_frequency_metrics(combined_trades, combined_curve if not combined_curve.empty else _fallback_curve(pd.Index([pd.Timestamp.utcnow()]), base_initial_capital)))
    if not combined_curve.empty:
        plot_paths.append(str(plot_equity_curve(combined_curve, "Portfolio %s Equity" % timeframe.value, plots_dir / "portfolio_equity.png")))
        plot_paths.append(str(plot_drawdown_curve(combined_curve, "Portfolio %s Drawdown" % timeframe.value, plots_dir / "portfolio_drawdown.png")))

    return {
        "timeframe": timeframe.value,
        "per_instrument": per_instrument,
        "aggregate_metrics": aggregate_metrics,
        "plot_paths": plot_paths,
    }


def _results_table(results: Sequence[Mapping[str, Any]], run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in results:
        row = {
            "run_id": run_id,
            "candidate_id": candidate["candidate_id"],
            "strategy_id": candidate["strategy_id"],
            "timeframe": candidate["timeframe"],
            "parameters": json.dumps(candidate["parameters"], sort_keys=True),
            "passes_guardrails": candidate["passes_guardrails"],
            "in_sample_score": candidate["in_sample_score"],
            "out_of_sample_score": candidate["out_of_sample_score"],
            "stability_score": candidate["stability_score"],
            "frequency_score": candidate["frequency_score"],
            "parameter_robustness": candidate["parameter_robustness"],
            "complexity_penalty": candidate["complexity_penalty"],
            "baseline_margin": candidate["baseline_margin"],
            "active_days": candidate["active_days"],
            "buy_and_hold_return": candidate["buy_and_hold_return"],
            "random_baseline_return": candidate["random_baseline_return"],
        }
        for prefix, metrics in (
            ("is", candidate["in_sample_metrics"]),
            ("oos", candidate["out_of_sample_metrics"]),
        ):
            for key, value in metrics.items():
                row["%s_%s" % (prefix, key)] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(by=["passes_guardrails", "out_of_sample_score"], ascending=[False, False]).reset_index(drop=True)


def _metrics_table(metrics_by_name: Mapping[str, Mapping[str, float]]) -> str:
    headers = [
        "Instrument",
        "Trades",
        "Win Rate",
        "Profit Factor",
        "Expectancy",
        "Max DD",
        "Avg Hold (h)",
        "Time In Market",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for instrument, metrics in metrics_by_name.items():
        lines.append(
            "| %s | %d | %.1f%% | %.2f | %.4f | %.1f%% | %.2f | %.1f%% |"
            % (
                instrument,
                int(metrics["trade_count"]),
                float(metrics["win_rate"]) * 100.0,
                float(metrics["profit_factor"]),
                float(metrics["expectancy"]),
                float(metrics["max_drawdown"]) * 100.0,
                float(metrics.get("avg_holding_hours", 0.0)),
                float(metrics.get("time_in_market", 0.0)) * 100.0,
            )
        )
    return "\n".join(lines)


def write_report(
    config: Mapping[str, Any],
    report_dir: Path,
    results_df: pd.DataFrame,
    champion_artifact: Mapping[str, Any],
    full_sample: Mapping[str, Any],
    success: bool,
) -> Path:
    top_rows = results_df.head(5)
    ranking_lines = [
        "| Rank | Strategy | Timeframe | OOS Score | OOS Trades | Trades/Day | Guardrails |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        ranking_lines.append(
            "| %d | %s | %s | %.3f | %d | %.2f | %s |"
            % (
                rank,
                row["strategy_id"],
                row["timeframe"],
                row["out_of_sample_score"],
                int(row["oos_trade_count"]),
                row["guardrails_trades_per_day"] if "guardrails_trades_per_day" in row else 0.0,
                "pass" if bool(row["passes_guardrails"]) else "fail",
            )
        )

    coverage_lines = []
    for instrument, payload in champion_artifact["champion"]["per_instrument"].items():
        coverage = payload["coverage"]
        warning_text = ", ".join(coverage["warnings"]) if coverage["warnings"] else "none"
        coverage_lines.append(
            "- %s: `%s` bars, gaps `%s`, partial sessions `%s`, warnings `%s`"
            % (
                instrument,
                coverage["bar_count"],
                coverage["gap_count"],
                coverage["partial_session_count"],
                warning_text,
            )
        )

    aggregate = full_sample["aggregate_metrics"]
    report_lines = [
        "# Intraday Research Report",
        "",
        "Status: %s" % ("PASS" if success else "FAIL"),
        "",
        "Run ID: `%s`" % champion_artifact["run_id"],
        "Generated: `%s`" % champion_artifact["generated_at"],
        "Git Commit: `%s`" % champion_artifact["git_commit"],
        "Config Hash: `%s`" % champion_artifact["config_hash"],
        "Seed: `%s`" % champion_artifact["seed"],
        "Window: `%s` to `%s`" % (config["research"]["start_date"], config["research"]["end_date"]),
        "",
        "## Champion",
        "",
        "Strategy: `%s`" % champion_artifact["champion"]["strategy_id"],
        "Timeframe: `%s`" % champion_artifact["champion"]["timeframe"],
        "Parameters: `%s`" % json.dumps(champion_artifact["champion"]["parameters"], sort_keys=True),
        "Out-of-sample score: `%.3f`" % champion_artifact["champion"]["out_of_sample_score"],
        "Trades/day (OOS): `%.2f`" % champion_artifact["champion"]["guardrail_notes"]["trades_per_day"],
        "Baseline margin: `%.3f`" % champion_artifact["champion"]["baseline_margin"],
        "",
        "## Ranking",
        "",
        *ranking_lines,
        "",
        "## Data Coverage",
        "",
        *coverage_lines,
        "",
        "## Full-Sample Metrics",
        "",
        _metrics_table({name: payload["metrics"] for name, payload in full_sample["per_instrument"].items()}),
        "",
        "Portfolio aggregate trades: `%d`" % int(aggregate["trade_count"]),
        "Portfolio max drawdown: `%.2f%%`" % (float(aggregate["max_drawdown"]) * 100.0),
        "Portfolio avg holding hours: `%.2f`" % float(aggregate.get("avg_holding_hours", 0.0)),
        "",
    ]

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def run_research(config: Mapping[str, Any], report_dir: Path) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    git_commit = _git_commit_hash()
    config_hash = _config_hash(config)
    seed = int(config["research"]["seed"])
    events_path = report_dir / "events.jsonl"
    _log_event(
        events_path,
        {
            "event": "research_started",
            "run_id": run_id,
            "git_sha": git_commit,
            "config_hash": config_hash,
            "timeframes": list(config["research"]["timeframes"]),
        },
    )

    market_data = load_market_data(config)
    budget = int(config["optimization"]["parameter_budget_per_strategy"])
    results: List[Dict[str, Any]] = []

    for strategy_id in config["strategies"]["enabled"]:
        strategy = get_strategy(strategy_id)
        for timeframe in (Timeframe.from_value(value) for value in config["research"]["timeframes"]):
            if timeframe not in strategy.supports_timeframes():
                continue
            for attempt in range(budget):
                params = sample_parameters(strategy, seed + ((attempt + 1) * 17) + len(results))
                candidate = evaluate_strategy_candidate(strategy_id, timeframe, params, market_data, config)
                results.append(candidate)
                _log_event(
                    events_path,
                    {
                        "event": "candidate_evaluated",
                        "run_id": run_id,
                        "candidate_id": candidate["candidate_id"],
                        "strategy_id": strategy_id,
                        "timeframe": timeframe.value,
                        "out_of_sample_score": candidate["out_of_sample_score"],
                        "passes_guardrails": candidate["passes_guardrails"],
                    },
                )

    results_df = _results_table(results, run_id)
    guardrails_frame = pd.json_normalize(results_df["candidate_id"].map(
        {
            candidate["candidate_id"]: candidate["guardrail_notes"]
            for candidate in results
        }
    )).add_prefix("guardrails_") if not results_df.empty else pd.DataFrame()
    if not results_df.empty and not guardrails_frame.empty:
        results_df = pd.concat([results_df, guardrails_frame], axis=1)

    results_csv = report_dir / "results.csv"
    results_parquet = report_dir / "results.parquet"
    registry_csv = report_dir / "registry.csv"
    results_df.to_csv(results_csv, index=False)
    results_df.to_parquet(results_parquet, index=False)
    results_df.to_csv(registry_csv, index=False)

    config_snapshot_path = report_dir / "config.snapshot.yaml"
    config_snapshot_path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")

    valid_results = [candidate for candidate in results if candidate["passes_guardrails"]]
    ranked_results = valid_results if valid_results else results
    champion_candidate = max(ranked_results, key=lambda item: item["out_of_sample_score"])
    success = bool(valid_results)

    full_sample = evaluate_full_sample(champion_candidate, market_data, config, report_dir)
    bootstrap = _bootstrap_trade_robustness(
        pd.concat(
            [
                run_backtest(
                    payloads[champion_candidate["timeframe"]]["prices"],
                    get_strategy(champion_candidate["strategy_id"]).generate_signals(
                        payloads[champion_candidate["timeframe"]]["prices"],
                        champion_candidate["parameters"],
                    ),
                    build_backtest_config(
                        config,
                        Timeframe.from_value(champion_candidate["timeframe"]),
                        {
                            **payloads[champion_candidate["timeframe"]]["session_rules"],
                            **get_strategy(champion_candidate["strategy_id"]).session_rules(),
                        },
                    ),
                    instrument=instrument,
                ).trades
                for instrument, payloads in market_data.items()
                if champion_candidate["timeframe"] in payloads
            ],
            ignore_index=True,
        ) if market_data else pd.DataFrame(),
        samples=int(config["optimization"].get("bootstrap_samples", 0)),
        seed=seed,
    )

    champion_artifact = {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit,
        "config_hash": config_hash,
        "seed": seed,
        "paper_only": True,
        "config_snapshot_path": str(config_snapshot_path),
        "risk": build_backtest_config(
            config,
            Timeframe.from_value(champion_candidate["timeframe"]),
            next(iter(champion_candidate["per_instrument"].values()))["session_rules"],
        ).to_dict(),
        "cost_model": {
            "transaction_cost_bps": float(config["risk"]["transaction_cost_bps"]),
            "slippage_bps": float(config["risk"]["slippage_bps"]),
            "spread_bps": float(config["risk"]["spread_bps"]),
        },
        "validation": {
            "walk_forward": dict(config["walk_forward"]),
            "bootstrap": bootstrap,
        },
        "champion": {
            "candidate_id": champion_candidate["candidate_id"],
            "strategy_id": champion_candidate["strategy_id"],
            "timeframe": champion_candidate["timeframe"],
            "parameters": champion_candidate["parameters"],
            "session_rules": next(iter(champion_candidate["per_instrument"].values()))["session_rules"],
            "in_sample_score": champion_candidate["in_sample_score"],
            "out_of_sample_score": champion_candidate["out_of_sample_score"],
            "stability_score": champion_candidate["stability_score"],
            "frequency_score": champion_candidate["frequency_score"],
            "parameter_robustness": champion_candidate["parameter_robustness"],
            "complexity_penalty": champion_candidate["complexity_penalty"],
            "baseline_margin": champion_candidate["baseline_margin"],
            "guardrail_notes": champion_candidate["guardrail_notes"],
            "in_sample_metrics": _serialize_metrics(champion_candidate["in_sample_metrics"]),
            "out_of_sample_metrics": _serialize_metrics(champion_candidate["out_of_sample_metrics"]),
            "per_instrument": champion_candidate["per_instrument"],
            "full_sample": full_sample,
        },
        "data_coverage": {
            instrument: payloads[champion_candidate["timeframe"]]["coverage"]
            for instrument, payloads in market_data.items()
            if champion_candidate["timeframe"] in payloads
        },
    }

    champion_path = report_dir / "champion.json"
    champion_path.write_text(json.dumps(champion_artifact, indent=2), encoding="utf-8")

    artifact_dir = Path("artifacts/champions")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    versioned_champion = artifact_dir / ("%s_%s.json" % (run_id, champion_candidate["candidate_id"]))
    versioned_champion.write_text(json.dumps(champion_artifact, indent=2), encoding="utf-8")

    report_path = write_report(config, report_dir, results_df, champion_artifact, full_sample, success)
    paper_paths = run_paper_session(
        config=config,
        champion_path=champion_path,
        mode="replay",
        days=int(config["paper"]["replay_days"]),
        report_dir=report_dir,
    )

    _log_event(
        events_path,
        {
            "event": "research_completed",
            "run_id": run_id,
            "success": success,
            "champion_id": champion_candidate["candidate_id"],
            "champion_strategy": champion_candidate["strategy_id"],
            "champion_timeframe": champion_candidate["timeframe"],
            "report_path": str(report_path),
        },
    )

    return {
        "success": success,
        "report_dir": report_dir,
        "report_path": report_path,
        "champion_path": champion_path,
        "results_csv": results_csv,
        "results_parquet": results_parquet,
        "registry_csv": registry_csv,
        "paper_paths": paper_paths,
        "events_path": events_path,
        "versioned_champion_path": versioned_champion,
    }
