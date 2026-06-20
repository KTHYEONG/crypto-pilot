from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.domain.futures.optimization import opt_data_utils


def test_safe_read_funding_parquet_normalizes_duplicate_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)
    path = tmp_path / "HOOKUSDT_funding.parquet"
    path.touch()

    bad_df = pd.DataFrame(
        [[1711929600000, "HOOKUSDT", 0.0001, "DUP"]],
        columns=["timestamp", "1", "funding_rate", "1"],
    )
    monkeypatch.setattr(opt_data_utils.pd, "read_parquet", lambda *_args, **_kwargs: bad_df)  # type: ignore[attr-defined]

    out = opt_data_utils._safe_read_funding_parquet("HOOKUSDT")
    assert out is not None
    assert list(out.columns) == ["timestamp", "funding_rate", "datetime"]
    assert len(out) == 1


def test_evaluate_symbol_data_sufficiency_historical_stage5_union_relaxes_fetch_oos() -> None:
    dt = pd.date_range("2022-10-01", "2025-10-15", freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    symbol_map = {"4h": frame}

    strict = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="TESTUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=252,
        scope_name="stage6_selected",
    )
    relaxed = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="TESTUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=252,
        scope_name="historical_stage5_union",
    )

    assert strict["pass"] is False
    assert strict["reason"] == "fetch_window_short"
    assert relaxed["pass"] is True


def test_evaluate_symbol_data_sufficiency_gap_too_large_fails() -> None:
    """S1: 96% bar coverage but 25h gap → gap_too_large failure."""
    # Build contiguous 4h bars from 2022-10-01 to 2026-03-31, then insert a 28h gap
    dt_full = pd.date_range("2022-10-01", "2026-03-31", freq="4h", tz="UTC")
    gap_start_idx = len(dt_full) // 2
    # Remove 7 consecutive bars (7 intervals = 6 missing bars = 24h+ gap)
    dt_gapped = pd.DatetimeIndex(
        list(dt_full[:gap_start_idx]) + list(dt_full[gap_start_idx + 7 :])
    )
    frame = pd.DataFrame(
        {
            "datetime": dt_gapped,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    symbol_map = {"4h": frame}

    res = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="GAPUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=252,
        scope_name="stage6_selected",
    )

    assert res["pass"] is False
    assert res["reason"] == "gap_too_large"
    assert res["max_gap_bars"] >= 6


def test_evaluate_symbol_data_sufficiency_small_gap_passes() -> None:
    """Small gap (≤5 bars = 20h) must not trigger gap_too_large."""
    dt_full = pd.date_range("2022-10-01", "2026-03-31", freq="4h", tz="UTC")
    gap_start_idx = len(dt_full) // 2
    # Remove 5 consecutive bars (5 intervals = 4 missing bars < threshold)
    dt_gapped = pd.DatetimeIndex(
        list(dt_full[:gap_start_idx]) + list(dt_full[gap_start_idx + 5 :])
    )
    frame = pd.DataFrame(
        {
            "datetime": dt_gapped,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    symbol_map = {"4h": frame}

    res = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="SMALLGAPUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=252,
        scope_name="stage6_selected",
    )

    assert res["pass"] is True
    assert res["max_gap_bars"] < 6


def test_evaluate_symbol_data_sufficiency_with_onboard_date() -> None:
    # 2023-10-01 ~ 2026-03-31 데이터 시뮬레이션 (상장일이 2023-10-01인 코인)
    dt = pd.date_range("2023-10-01", "2026-03-31", freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    symbol_map = {"4h": frame}

    # onboard_date 미지정 시: 2022-10-01(fetch_start) 데이터가 없으므로 fetch_window_short 로 실패해야 함.
    res_no_onboard = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="PEPEUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=0,
        scope_name="stage6_selected",
    )
    assert res_no_onboard["pass"] is False
    assert res_no_onboard["reason"] == "fetch_window_short"

    # onboard_date="2023-10-01" 지정 시: effective_fetch_start 가 2023-10-01로 보정되어 패스해야 함.
    res_with_onboard = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="PEPEUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2022-10-01",
        is_start="2023-10-01",
        oos_start="2025-10-01",
        oos_end="2026-03-31",
        require_exec_1m=False,
        warmup_bars_required=0,
        scope_name="stage6_selected",
        onboard_date="2023-10-01",
    )
    assert res_with_onboard["pass"] is True

