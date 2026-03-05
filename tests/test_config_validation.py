from __future__ import annotations

import copy
import os

import pytest

from src.lab.config import DEFAULT_CONFIG, load_config, validate_config


def test_validate_config_rejects_invalid_symbol_and_dates() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["research"]["start_date"] = "2025-02-01"
    config["research"]["end_date"] = "2024-02-01"
    config["data"]["instruments"]["Gold"]["primary"] = "GC=F;"

    with pytest.raises(ValueError):
        validate_config(config)


def test_load_config_applies_env_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "research:",
                "  start_date: '2022-01-03'",
                "  end_date: '2024-12-31'",
                "data:",
                "  cache_dir: data/cache",
                "strategies:",
                "  enabled: [ma_crossover]",
            ]
        ),
        encoding="utf-8",
    )
    os.environ["LAB_RISK__POSITION_FRACTION"] = "0.15"
    os.environ["LAB_RESEARCH__SEED"] = "17"

    try:
        config = load_config(config_path)
    finally:
        os.environ.pop("LAB_RISK__POSITION_FRACTION", None)
        os.environ.pop("LAB_RESEARCH__SEED", None)

    assert config["risk"]["position_fraction"] == 0.15
    assert config["research"]["seed"] == 17
