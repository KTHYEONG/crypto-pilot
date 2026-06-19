from __future__ import annotations

import pandas as pd

from src.domain.futures.universe.filters import Stage3Config
from src.domain.futures.universe.filters import apply_liquidity_stage


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "adv_usdt_median": [100_000_000.0],
            "amihud_30d": [5.0e-10],
            "screening_clip_usdt": [100_000.0],
        }
    )


def test_apply_liquidity_stage_when_oi_to_adv_exceeds_threshold_rejects_symbol() -> None:
    frame = _base_frame()
    frame["sum_open_interest_value"] = [1_500_000_000.0]

    filtered, report = apply_liquidity_stage(frame, config=Stage3Config())

    assert filtered.empty
    assert report.loc[0, "reason"] == "oi_adv_crowded"
    assert float(report.loc[0, "oi_to_adv"]) == 15.0


def test_apply_liquidity_stage_when_oi_column_missing_does_not_reject_symbol() -> None:
    filtered, report = apply_liquidity_stage(_base_frame(), config=Stage3Config())

    assert list(filtered["symbol"]) == ["BTCUSDT"]
    assert report.loc[0, "reason"] == "pass"
    assert pd.isna(report.loc[0, "oi_to_adv"])


def test_apply_liquidity_stage_includes_oi_to_adv_report_column() -> None:
    frame = _base_frame()
    frame["oi_usdt_median"] = [600_000_000.0]

    _, report = apply_liquidity_stage(
        frame,
        config=Stage3Config(enable_oi_adv_crowding_gate=False),
    )

    assert "oi_to_adv" in report.columns
    assert float(report.loc[0, "oi_to_adv"]) == 6.0
