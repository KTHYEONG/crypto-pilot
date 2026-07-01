from __future__ import annotations

from typing import Any

import pytest

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle, RunWindow


def make_run_config(phase: str = "l3") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h", date="2026-05-01", trials=3,
        phase=phase, sync="skip", refresh_universe=False, sync_metrics=False,  # type: ignore[arg-type]
    )

def make_window() -> RunWindow:
    from datetime import date
    return RunWindow(
        fetch_start="2024-01-01", is_start="2024-04-01", oos_start="2025-01-01",
        end_date="2026-05-01", fetch_start_date=date(2024,1,1), is_start_date=date(2024,4,1),
        oos_start_date=date(2025,1,1), end_date_value=date(2026,5,1),
    )

def make_data_bundle() -> MarketDataBundle:
    return MarketDataBundle(
        data_maps={"BTCUSDT": {"4h": object()}},
        oos_data_maps={"BTCUSDT": {"4h": object()}},
        valid_symbols=("BTCUSDT",),
    )


from src.application.futures.runner.config import build_run_config_from_args


class TestBuildRunConfig:
    @pytest.mark.parametrize(
        ("args_dict", "expected_match"),
        [
            ({"phase": "l3", "timeframe": "4h", "trials": 0, "sync": "skip"}, "trials must be >= 1"),
            ({"phase": "quick-backtest", "timeframe": "4h", "trials": 1, "sync": "skip"}, "removed phase"),
        ],
    )
    def test_build_run_config_rejects_invalid_or_removed_inputs(
        self, args_dict: dict[str, Any], expected_match: str,
    ) -> None:
        with pytest.raises(ValueError, match=expected_match):
            build_run_config_from_args(args_dict)
