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
    monkeypatch.setattr(opt_data_utils.pd, "read_parquet", lambda *_args, **_kwargs: bad_df)

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
