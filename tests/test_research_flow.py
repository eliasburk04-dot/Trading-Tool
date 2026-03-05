from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.lab.cli import main
from src.lab.data_layer import BarStore
from src.lab.config import DEFAULT_CONFIG, load_config
from src.lab.research import run_research
from tests.conftest import build_intraday_frame, build_price_frame


def _write_cached_market_data(cache_dir: Path, start_date: str, end_date: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed, symbol in enumerate(["GC=F", "^IXIC", "^GSPC"], start=21):
        build_price_frame(start=start_date, periods=780, seed=seed).to_parquet(
            cache_dir / "legacy" / "unused_%s_%s.parquet" % (seed, symbol.replace("=", "_").replace("^", "_"))
        )


def _build_config(cache_dir: Path, start_date: str, end_date: str) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["research"]["start_date"] = start_date
    config["research"]["end_date"] = end_date
    config["research"]["seed"] = 23
    config["data"]["cache_dir"] = str(cache_dir)
    config["optimization"]["parameter_budget_per_strategy"] = 2
    config["guardrails"]["max_drawdown"] = 0.40
    config["guardrails"]["min_out_of_sample_score"] = 0.02
    config["guardrails"]["min_oos_is_ratio"] = 0.2
    config["guardrails"]["frequency"] = {
        "H1": {
            "min_total_trades": 6,
            "min_trades_per_day": 0.5,
            "min_oos_trades_per_split": 2,
        }
    }
    config["research"]["timeframes"] = ["H1"]
    config["strategies"]["enabled"] = ["vwap_mean_reversion", "opening_range_breakout"]
    config["paper"]["replay_days"] = 30
    config["walk_forward"]["train_months"] = 2
    config["walk_forward"]["test_months"] = 1
    config["walk_forward"]["step_months"] = 1
    config["walk_forward"]["purge_bars"] = 2
    config["walk_forward"]["embargo_bars"] = 2
    return config


def test_run_research_direct_produces_artifacts(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    report_dir = tmp_path / "research-report"
    start_date = "2024-01-02"
    end_date = "2024-04-30"
    store = BarStore(cache_dir)
    for seed, symbol in enumerate(["GC=F", "^IXIC", "^GSPC"], start=21):
        frame = build_intraday_frame(start=start_date, days=65, timeframe="1h", seed=seed)
        path = store.cache_path(symbol=symbol, timeframe="H1", start_date=start_date, end_date=end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_build_config(cache_dir, start_date, end_date), sort_keys=False),
        encoding="utf-8",
    )
    config = load_config(config_path)

    result = run_research(config, report_dir)

    assert result["success"] is True
    assert result["champion_path"].exists()
    assert result["report_path"].exists()
    assert result["paper_paths"]["orders"].exists()
    assert result["registry_csv"].exists()
    results = pd.read_csv(result["results_csv"])
    champion = json.loads(result["champion_path"].read_text(encoding="utf-8"))
    assert not results.empty
    assert champion["champion"]["timeframe"] == "H1"
    assert "out_of_sample_score" in results.columns


def test_cli_main_dispatches_research(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    expected_report_dir = tmp_path / "output"
    calls = {}

    def fake_load_config(path, overrides=None):
        calls["config_path"] = Path(path)
        calls["overrides"] = overrides
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["research"]["report_dir"] = str(expected_report_dir)
        config["research"]["start_date"] = "2022-01-03"
        config["research"]["end_date"] = "2024-12-31"
        return config

    def fake_run_research(config, report_dir):
        calls["report_dir"] = Path(report_dir)
        return {
            "success": True,
            "report_dir": Path(report_dir),
            "champion_path": Path(report_dir) / "champion.json",
        }

    monkeypatch.setattr("src.lab.cli.load_config", fake_load_config)
    monkeypatch.setattr("src.lab.cli.run_research", fake_run_research)

    exit_code = main(["research", "--config", str(config_path), "--last-years", "2"])

    assert exit_code == 0
    assert calls["config_path"] == config_path
    assert calls["overrides"]["research"]["last_years"] == 2
    assert calls["report_dir"] == expected_report_dir


def test_cli_main_dispatches_paper_run(monkeypatch, tmp_path) -> None:
    champion_path = tmp_path / "champion.json"
    champion_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    calls = {}

    def fake_load_config(path, overrides=None):
        calls["config_path"] = Path(path)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["research"]["start_date"] = "2024-01-02"
        config["research"]["end_date"] = "2024-04-30"
        return config

    def fake_run_paper_session(config, champion_path_arg, mode, days, report_dir):
        calls["champion_path"] = Path(champion_path_arg)
        calls["mode"] = mode
        calls["days"] = days
        calls["report_dir"] = Path(report_dir)
        return {"success": True, "report_dir": Path(report_dir)}

    monkeypatch.setattr("src.lab.cli.load_config", fake_load_config)
    monkeypatch.setattr("src.lab.cli.run_paper_session", fake_run_paper_session)

    exit_code = main(
        [
            "paper-run",
            "--config",
            str(config_path),
            "--champion",
            str(champion_path),
            "--mode",
            "replay",
            "--days",
            "5",
        ]
    )

    assert exit_code == 0
    assert calls["champion_path"] == champion_path
    assert calls["mode"] == "replay"
    assert calls["days"] == 5
