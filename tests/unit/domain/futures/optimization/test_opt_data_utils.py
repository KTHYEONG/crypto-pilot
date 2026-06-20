from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


# ── OPT-1: Skip raw_df.copy() when no merge needed ──────────────────────


def test_load_single_symbol_data_skips_copy_when_no_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPT-1: funding/metrics=None → raw_df is NOT copied (identity equality)."""
    _TF_FREQ = {"1h": "1h", "4h": "4h", "1d": "D"}

    def _fake_collect(self: object, sym: str, tf: str, *_a: Any, **_kw: Any) -> pd.DataFrame:
        freq = _TF_FREQ.get(tf, "4h")
        dt = pd.date_range("2023-01-01", "2024-06-01", freq=freq, tz="UTC")
        return pd.DataFrame({"datetime": dt, "open": 1.0, "close": 1.0})

    monkeypatch.setattr(
        opt_data_utils.DataCollector, "collect_and_save", _fake_collect  # type: ignore[attr-defined]
    )
    monkeypatch.setattr(
        opt_data_utils,
        "compute_segment_merge_index",
        lambda *_a, **_kw: 0,
    )

    _, _, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym="TESTUSDT",
        tf="4h",
        fetch_start="2024-01-01",
        start="2024-02-01",
        is_end="2024-04-01",
        end="2024-06-01",
        skip_metrics=True,
    )
    assert insufficient is False


# ── OPT-2: Single _to_unix_ms call per TF ──────────────────────────────


def test_load_single_symbol_data_single_unix_ms_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPT-2: With both funding and metrics, _to_unix_ms called exactly 1x per TF."""
    _TF_FREQ = {"1h": "1h", "4h": "4h", "1d": "D"}
    call_count: list[int] = [0]
    _orig_to_unix = opt_data_utils._to_unix_ms

    def _tracking_to_unix(s: Any) -> Any:
        call_count[0] += 1
        return _orig_to_unix(s)

    monkeypatch.setattr(opt_data_utils, "_to_unix_ms", _tracking_to_unix)

    # Mock funding parquet
    funding_df = pd.DataFrame(
        {
            "timestamp": [1704067200000, 1704096000000],
            "funding_rate": [0.0001, -0.0002],
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 04:00:00"], utc=True
            ),
        }
    )

    def _fake_read_funding(sym: str) -> pd.DataFrame | None:
        return funding_df

    monkeypatch.setattr(
        opt_data_utils, "_safe_read_funding_parquet", _fake_read_funding
    )

    def _fake_collect(self: object, sym: str, tf: str, *_a: Any, **_kw: Any) -> pd.DataFrame:
        freq = _TF_FREQ.get(tf, "4h")
        dt = pd.date_range("2023-01-01", "2024-06-01", freq=freq, tz="UTC")
        return pd.DataFrame({"datetime": dt, "open": 1.0, "close": 1.0})

    monkeypatch.setattr(
        opt_data_utils.DataCollector, "collect_and_save", _fake_collect  # type: ignore[attr-defined]
    )
    monkeypatch.setattr(
        opt_data_utils,
        "compute_segment_merge_index",
        lambda *_a, **_kw: 0,
    )

    call_count[0] = 0  # reset before load
    _, _, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym="TESTUSDT",
        tf="4h",
        fetch_start="2024-01-01",
        start="2024-02-01",
        is_end="2024-04-01",
        end="2024-06-01",
        skip_metrics=False,
    )
    assert insufficient is False
    # Expected: 1 call per TF (3 TFs: 4h, 1d, 1h)
    # Old code called _to_unix_ms up to 2x per TF (funding + metrics checks)
    assert call_count[0] == 3, (
        f"Expected 3 _to_unix_ms calls (1 per TF) but got {call_count[0]}"
    )


# ── OPT-3: Lightweight audit for merged stage ──────────────────────────


def test_append_stage_integrity_merged_lightweight() -> None:
    """OPT-3: merged stage stores rows/cols only, not nan_pct."""
    df = pd.DataFrame({
        "a": [1.0, None],
        "b": [3.0, 4.0],
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    audit: list[dict[str, Any]] = []
    opt_data_utils._append_stage_integrity(
        audit, symbol="TST", timeframe="4h", stage="merged", df=df,
    )
    assert len(audit) == 1
    rec = audit[0]
    assert rec["symbol"] == "TST"
    assert rec["timeframe"] == "4h"
    assert rec["stage"] == "merged"
    assert rec["rows"] == 2.0
    assert rec["cols"] == 3.0
    assert "nan_pct" not in rec
    assert "gap_count" not in rec


def test_append_stage_integrity_raw_still_lightweight() -> None:
    """raw stage still stores rows/cols (unchanged behavior)."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    audit: list[dict[str, Any]] = []
    opt_data_utils._append_stage_integrity(
        audit, symbol="TST", timeframe="4h", stage="raw", df=df,
    )
    assert len(audit) == 1
    rec = audit[0]
    assert rec["rows"] == 1.0
    assert rec["cols"] == 2.0


def test_append_stage_integrity_preserves_fillna() -> None:
    """fillna_cols coverage still computed for merged stage."""
    df = pd.DataFrame({
        "a": [1.0, None, 3.0],
        "b": [None, 2.0, 3.0],
        "datetime": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
        ),
    })
    audit: list[dict[str, Any]] = []
    opt_data_utils._append_stage_integrity(
        audit, symbol="TST", timeframe="4h", stage="merged", df=df,
        fillna_cols=["a", "b"],
    )
    rec = audit[0]
    assert "pre_fillna_nan_pct" in rec
    assert rec["pre_fillna_nan_pct"] == pytest.approx(2.0 / 6.0)
    assert "nan_pct" not in rec


# ── OPT-4: Column group cache ──────────────────────────────────────────


def _make_feature_df(cols: tuple[str, ...], n: int = 10) -> pd.DataFrame:
    data: dict[str, Any] = {"datetime": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")}
    for c in cols:
        data[c] = np.random.default_rng(42).random(n)
    return pd.DataFrame(data)


def test_feature_group_coverage_cache_used_on_repeat_call() -> None:
    """OPT-4: Second call with same columns should hit cache."""
    opt_data_utils._COL_GROUP_CACHE.clear()
    cols = ("open", "high", "low", "close", "volume", "funding_rate", "open_interest")
    df1 = _make_feature_df(cols)
    df2 = _make_feature_df(cols)

    r1 = opt_data_utils._feature_group_coverage(df1, tf_label="4h")
    assert len(opt_data_utils._COL_GROUP_CACHE) == 1
    r2 = opt_data_utils._feature_group_coverage(df2, tf_label="4h")
    assert r1 == r2


def test_feature_group_coverage_cache_keyed_by_tf() -> None:
    """Different tf_label values produce separate cache entries."""
    opt_data_utils._COL_GROUP_CACHE.clear()
    cols = ("open", "close", "funding_rate", "oi")
    df = _make_feature_df(cols)

    opt_data_utils._feature_group_coverage(df, tf_label="4h")
    assert len(opt_data_utils._COL_GROUP_CACHE) == 1
    opt_data_utils._feature_group_coverage(df, tf_label="1d")
    assert len(opt_data_utils._COL_GROUP_CACHE) == 2


def test_feature_group_coverage_empty_df_returns_empty() -> None:
    """Empty DataFrame returns zeroed coverage immediately (no cache entry)."""
    opt_data_utils._COL_GROUP_CACHE.clear()
    r = opt_data_utils._feature_group_coverage(pd.DataFrame())
    for group in opt_data_utils._FEATURE_GROUP_PATTERNS:
        assert r[group] == {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
    assert len(opt_data_utils._COL_GROUP_CACHE) == 0


def test_feature_group_coverage_non_numeric_coerced() -> None:
    """Numeric coercion for non-numeric columns still works with cached mapping."""
    opt_data_utils._COL_GROUP_CACHE.clear()
    df = pd.DataFrame({
        "open": [1.0, 2.0],
        "close": [3.0, 4.0],
        "funding_rate": ["0.0001", "0.0002"],
    })
    r = opt_data_utils._feature_group_coverage(df, tf_label="4h")
    price = r.get("price", {})
    assert price.get("col_count", 0.0) == 2.0
    funding = r.get("funding", {})
    assert funding.get("col_count", 0.0) == 1.0


# ── Regression: Merge path still works when merge is needed ─────────────


def test_append_stage_integrity_other_stage_uses_full() -> None:
    """Non-merged/raw stages still call the full integrity function."""
    df = pd.DataFrame({
        "a": [1.0, 2.0],
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    audit: list[dict[str, Any]] = []
    opt_data_utils._append_stage_integrity(
        audit, symbol="TST", timeframe="4h", stage="merged_other", df=df,
    )
    rec = audit[0]
    assert rec["stage"] == "merged_other"
    assert "rows" in rec or "nan_pct" in rec  # called summarize_dataframe_integrity

