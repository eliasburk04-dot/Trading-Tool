from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from src.lab.config import DEFAULT_CONFIG
from src.lab.data_layer import BarStore
from tests.conftest import build_intraday_frame


def test_research_and_paper_run_cli_produce_intraday_artifacts(tmp_path, repo_root: Path) -> None:
    cache_dir = tmp_path / "cache"
    report_dir = tmp_path / "reports"
    paper_dir = tmp_path / "paper"
    cache_dir.mkdir()
    report_dir.mkdir()
    paper_dir.mkdir()

    start_date = "2024-01-02"
    end_date = "2024-04-30"
    store = BarStore(cache_dir)
    for seed, symbol in enumerate(["GC=F", "^IXIC", "^GSPC"], start=11):
        for timeframe in ["H1", "M15"]:
            frame = build_intraday_frame(start=start_date, days=65, timeframe="1h" if timeframe == "H1" else "15m", seed=seed)
            path = store.cache_path(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)

    config = DEFAULT_CONFIG.copy()
    config["research"] = dict(
        DEFAULT_CONFIG["research"],
        start_date=start_date,
        end_date=end_date,
        seed=19,
        timeframes=["H1", "M15"],
    )
    config["data"] = dict(DEFAULT_CONFIG["data"], cache_dir=str(cache_dir))
    config["walk_forward"] = {
        "train_months": 2,
        "test_months": 1,
        "step_months": 1,
        "purge_bars": 2,
        "embargo_bars": 2,
    }
    config["optimization"] = {
        "parameter_budget_per_strategy": 2,
        "min_samples_before_stop": 2,
    }
    config["guardrails"] = {
        "max_drawdown": 0.35,
        "min_out_of_sample_score": 0.05,
        "min_oos_is_ratio": 0.3,
        "frequency": {
            "H1": {"min_total_trades": 6, "min_trades_per_day": 0.5, "min_oos_trades_per_split": 2},
            "M15": {"min_total_trades": 12, "min_trades_per_day": 1.0, "min_oos_trades_per_split": 2},
        },
    }
    config["strategies"] = {"enabled": ["vwap_mean_reversion", "opening_range_breakout", "intraday_rsi_reversion"]}

    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    research_process = subprocess.run(
        [
            str(repo_root / "tool"),
            "research",
            "--config",
            str(config_path),
            "--timeframes",
            "H1,M15",
            "--report-dir",
            str(report_dir),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert research_process.returncode == 0, research_process.stderr or research_process.stdout

    champion_path = report_dir / "champion.json"
    report_path = report_dir / "report.md"
    results_csv = report_dir / "results.csv"
    results_parquet = report_dir / "results.parquet"
    registry_csv = report_dir / "registry.csv"
    events_log = report_dir / "events.jsonl"

    assert champion_path.exists()
    assert report_path.exists()
    assert results_csv.exists()
    assert results_parquet.exists()
    assert registry_csv.exists()
    assert events_log.exists()
    assert any(path.suffix == ".png" for path in (report_dir / "plots").iterdir())

    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_csv)

    assert champion["paper_only"] is True
    assert champion["champion"]["strategy_id"] in {"vwap_mean_reversion", "opening_range_breakout", "intraday_rsi_reversion"}
    assert champion["champion"]["timeframe"] in {"H1", "M15"}
    assert not results.empty

    paper_process = subprocess.run(
        [
            str(repo_root / "tool"),
            "paper-run",
            "--config",
            str(config_path),
            "--champion",
            str(champion_path),
            "--mode",
            "replay",
            "--days",
            "5",
            "--report-dir",
            str(paper_dir),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert paper_process.returncode == 0, paper_process.stderr or paper_process.stdout
    assert (paper_dir / "paper" / "fills.csv").exists()
    assert (paper_dir / "paper" / "state.json").exists()
    assert (paper_dir / "paper" / "events.jsonl").exists()
