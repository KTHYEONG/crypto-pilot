from __future__ import annotations

import argparse

from src.application.futures.runner.compound_config import build_compound_run_config


def test_build_compound_config_is_fixed_to_hourly() -> None:
    config = build_compound_run_config(argparse.Namespace(sync="skip", seed=42))
    assert config.base_timeframe == "1h"
