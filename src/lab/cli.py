from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import sys

from src.lab.config import load_config
from src.lab.execution import run_paper_session
from src.lab.research import run_research


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    research_overrides: Dict[str, Any] = {}
    if args.last_years is not None:
        research_overrides["last_years"] = args.last_years
        research_overrides["start_date"] = None
        research_overrides["end_date"] = None
    if args.start_date is not None:
        research_overrides["start_date"] = args.start_date
    if args.end_date is not None:
        research_overrides["end_date"] = args.end_date
    if args.seed is not None:
        research_overrides["seed"] = args.seed
    if getattr(args, "timeframes", None):
        research_overrides["timeframes"] = [piece.strip() for piece in args.timeframes.split(",") if piece.strip()]

    overrides: Dict[str, Any] = {}
    if research_overrides:
        overrides["research"] = research_overrides
    return overrides


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy research + backtest lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    research_parser = subparsers.add_parser("research", help="Run the full research workflow")
    research_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    research_parser.add_argument("--last-years", type=int, default=None)
    research_parser.add_argument("--start-date", type=str, default=None)
    research_parser.add_argument("--end-date", type=str, default=None)
    research_parser.add_argument("--seed", type=int, default=None)
    research_parser.add_argument("--timeframes", type=str, default=None)
    research_parser.add_argument("--report-dir", type=Path, default=None)

    paper_parser = subparsers.add_parser("paper-run", help="Run paper execution from a champion artifact")
    paper_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    paper_parser.add_argument("--champion", type=Path, required=True)
    paper_parser.add_argument("--mode", choices=["replay", "paper-live"], default="replay")
    paper_parser.add_argument("--days", type=int, default=None)
    paper_parser.add_argument("--report-dir", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "research":
        overrides = _build_overrides(args)
        config = load_config(args.config, overrides=overrides)
        report_dir = args.report_dir or Path(config["research"]["report_dir"])
        result = run_research(config, report_dir)
        print("Report directory:", result["report_dir"])
        print("Champion artifact:", result["champion_path"])
        return 0 if result["success"] else 1

    if args.command == "paper-run":
        config = load_config(args.config)
        report_dir = args.report_dir or Path(config["research"]["report_dir"])
        result = run_paper_session(
            config,
            args.champion,
            args.mode,
            int(args.days if args.days is not None else config["paper"]["replay_days"]),
            report_dir,
        )
        print("Paper report directory:", report_dir)
        if "fills" in result:
            print("Fills:", result["fills"])
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
