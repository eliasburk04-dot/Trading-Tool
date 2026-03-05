from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json

from src.lab.data_layer import load_market_data
from src.lab.execution.engine import FillModel, PaperExecutionEngine
from src.lab.execution.state import ExecutionState


def run_paper_session(
    config: Mapping[str, Any],
    champion_path: Path,
    mode: str,
    days: int,
    report_dir: Path,
) -> Dict[str, Path]:
    champion = json.loads(Path(champion_path).read_text(encoding="utf-8"))
    state_path = report_dir / "paper" / "state.json"
    existing_state = ExecutionState.from_file(state_path) if state_path.exists() else None
    engine = PaperExecutionEngine(
        champion=champion,
        report_dir=report_dir,
        mode=mode,
        fill_model=FillModel(
            spread_bps=float(config["risk"].get("spread_bps", 1.0)),
            slippage_bps=float(config["risk"]["slippage_bps"]),
            seed=int(config["research"]["seed"]),
        ),
        existing_state=existing_state,
    )
    market_data = load_market_data(config)
    return engine.run(market_data, days)


__all__ = ["FillModel", "run_paper_session"]
