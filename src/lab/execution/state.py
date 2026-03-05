from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping
import json

from src.lab.risk import DailyRiskState


@dataclass
class ExecutionState:
    cash: float
    realized_pnl: float
    open_positions: Dict[str, Dict[str, Any]]
    last_processed_at: str | None
    risk: DailyRiskState
    orders: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["risk"] = asdict(self.risk)
        return payload

    def persist(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_file(cls, path: Path) -> "ExecutionState":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["risk"] = DailyRiskState(**payload["risk"])
        return cls(**payload)
