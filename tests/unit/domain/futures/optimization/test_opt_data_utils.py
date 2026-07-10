from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.domain.futures.optimization import opt_data_utils
from src.domain.futures.optimization.opt_data_utils import _scan_enriched_dataset  # type: ignore[attr-defined]


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
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: bad_df)

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


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_enriched_parquet(path: Path, dt_range: pd.DatetimeIndex) -> None:
    """Write a minimal enriched parquet with sorted int64 timestamp column.

    Mimics the wide_df.to_parquet(enriched_path) write from load_single_symbol_data.
    timestamp is unix-ms int64, sorted ascending so row-group statistics are valid.
    """
    timestamps = dt_range.tz_localize(None).astype("datetime64[ns]").astype("int64") // 10**6
    table = pa.table(
        {
            "datetime": pa.array(dt_range.to_pydatetime(), type=pa.timestamp("us", tz="UTC")),
            "timestamp": pa.array(timestamps.tolist(), type=pa.int64()),
            "open": pa.array([1.0] * len(dt_range), type=pa.float64()),
            "close": pa.array([1.0] * len(dt_range), type=pa.float64()),
        }
    )
    pq.write_table(table, str(path))  # type: ignore[no-untyped-call]


def _setup_multi_tf_enriched(tmp_path: Path, safe_sym: str) -> None:
    """Write 4h + 1h + 1D enriched parquets with dep raw files older than enriched.

    Covers all TFs required by the default ``tfs_to_load`` set in
    ``load_single_symbol_data`` ({tf, "1d", "1h", "4h"}) and satisfies
    ``compute_segment_merge_index(temp_is[tf], temp_is["1d"])`` dependency.
    """
    ranges: dict[str, pd.DatetimeIndex] = {
        "4h": pd.date_range("2020-01-01", "2025-12-31", freq="4h", tz="UTC"),
        "1h": pd.date_range("2020-01-01", "2025-12-31", freq="1h", tz="UTC"),
        "1d": pd.date_range("2020-01-01", "2025-12-31", freq="1D", tz="UTC"),
    }
    for tf_key, rng in ranges.items():
        ep = tmp_path / f"{safe_sym}_{tf_key}_enriched.parquet"
        _make_enriched_parquet(ep, rng)
        rp = tmp_path / f"{safe_sym}_{tf_key}.parquet"
        rp.touch()
        os.utime(rp, (time.time() - 10, time.time() - 10))


# ── Scenario 1: Happy Path — pushdown returns window-clipped df ───────────────


def test_load_single_symbol_data_cache_hit_pushdown_clips_to_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: Given enriched parquets for all default TFs + sub-window request →
    returned df datetime min/max within [req_start_dt, req_end_dt].
    collector.collect_and_save must NOT be called (cache-hit path).
    """
    # Arrange
    safe_sym = "BTCUSDT"
    tf_l = "4h"
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)

    _setup_multi_tf_enriched(tmp_path, safe_sym)

    collect_call_count: list[int] = [0]

    def _fake_collect(self: Any, sym: str, tf: str, *_a: Any, **_kw: Any) -> pd.DataFrame:
        collect_call_count[0] += 1
        return pd.DataFrame()

    monkeypatch.setattr(opt_data_utils.DataCollector, "collect_and_save", _fake_collect)  # type: ignore[attr-defined]
    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *_a, **_kw: 0)

    # Act — request sub-window 2022-01-01 to 2023-12-31
    _sym, _t_is, t_oos, insufficient = opt_data_utils.load_single_symbol_data(
        sym=safe_sym,
        tf=tf_l,
        fetch_start="2022-01-01",
        start="2022-07-01",
        is_end="2023-10-01",
        end="2023-12-31",
        skip_metrics=True,
    )

    # Assert
    assert insufficient is False
    assert collect_call_count[0] == 0, "collect_and_save must not be called on cache-hit"
    assert t_oos is not None
    df_result: pd.DataFrame = t_oos[tf_l]
    req_start = pd.Timestamp("2022-01-01", tz="UTC")
    req_end = pd.Timestamp("2023-12-31", tz="UTC")
    assert df_result["datetime"].min() >= req_start
    assert df_result["datetime"].max() <= req_end


# ── Scenario 2: R1 skip — funding/metrics not read on cache-hit + no exec_1m ─


def test_load_single_symbol_data_cache_hit_skips_funding_metrics_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: Given valid cache for all default TFs + load_exec_1m=False →
    _safe_read_funding_parquet must NOT be called (lazy load skipped entirely).
    """
    # Arrange
    safe_sym = "ETHUSDT"
    tf_l = "4h"
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)

    _setup_multi_tf_enriched(tmp_path, safe_sym)

    funding_read_count: list[int] = [0]

    def _fake_read_funding(sym: str) -> pd.DataFrame | None:
        funding_read_count[0] += 1
        return None

    monkeypatch.setattr(opt_data_utils, "_safe_read_funding_parquet", _fake_read_funding)
    monkeypatch.setattr(opt_data_utils.DataCollector, "collect_and_save", lambda *_a, **_kw: pd.DataFrame())  # type: ignore[attr-defined]
    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *_a, **_kw: 0)

    # Act
    _, _, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym=safe_sym,
        tf=tf_l,
        fetch_start="2022-01-01",
        start="2022-07-01",
        is_end="2023-10-01",
        end="2023-12-31",
        skip_metrics=False,  # metrics enabled — cache-hit + no exec_1m must skip funding I/O
        load_exec_1m=False,
    )

    # Assert
    assert insufficient is False
    assert funding_read_count[0] == 0, (
        "_safe_read_funding_parquet must not be called on cache-hit with load_exec_1m=False"
    )


# ── Scenario 3: Edge — empty window → insufficient=True ───────────────────────


def test_load_single_symbol_data_cache_hit_empty_window_returns_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: Request window outside data range → df is empty → insufficient=True."""
    # Arrange
    safe_sym = "SOLUSDT"
    tf_l = "4h"
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)

    # enriched data only covers 2020-2021; request window is 2022 (outside range)
    full_range = pd.date_range("2020-01-01", "2021-12-31", freq="4h", tz="UTC")
    enriched_path = tmp_path / f"{safe_sym}_{tf_l}_enriched.parquet"
    _make_enriched_parquet(enriched_path, full_range)

    raw_path = tmp_path / f"{safe_sym}_{tf_l}.parquet"
    raw_path.touch()
    os.utime(raw_path, (time.time() - 10, time.time() - 10))

    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *_a, **_kw: 0)

    # Act — request window entirely outside enriched data range
    _, _, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym=safe_sym,
        tf=tf_l,
        fetch_start="2022-06-01",
        start="2022-07-01",
        is_end="2022-10-01",
        end="2022-12-31",
        skip_metrics=True,
        target_tfs=[tf_l],
    )

    # Assert: empty df after pushdown/mask → insufficient=True
    assert insufficient is True


# ── Scenario 4: Fallback — filters exception → full-read+mask produces same df ─


def test_load_single_symbol_data_cache_hit_fallback_on_pushdown_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: When pd.read_parquet with filters raises, fallback to full-read+mask.
    Result df must be identical to what the mask would produce.
    """
    # Arrange
    safe_sym = "BNBUSDT"
    tf_l = "4h"
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)

    _setup_multi_tf_enriched(tmp_path, safe_sym)

    _orig_read_parquet = pd.read_parquet
    call_count: list[int] = [0]

    def _patched_read_parquet(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        # Raise on pushdown attempt (filters kwarg present); succeed on fallback call
        if "filters" in kwargs:
            call_count[0] += 1
            raise RuntimeError("simulated pushdown failure")
        return _orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _patched_read_parquet)
    monkeypatch.setattr(opt_data_utils.DataCollector, "collect_and_save", lambda *_a, **_kw: pd.DataFrame())  # type: ignore[attr-defined]
    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *_a, **_kw: 0)

    # Act
    _, _, t_oos, insufficient = opt_data_utils.load_single_symbol_data(
        sym=safe_sym,
        tf=tf_l,
        fetch_start="2022-01-01",
        start="2022-07-01",
        is_end="2023-10-01",
        end="2023-12-31",
        skip_metrics=True,
    )

    # Assert: fallback path succeeded, pushdown was attempted (call_count > 0)
    assert call_count[0] > 0, "pushdown must have been attempted before fallback"
    assert insufficient is False
    assert t_oos is not None
    df_result: pd.DataFrame = t_oos[tf_l]
    req_start = pd.Timestamp("2022-01-01", tz="UTC")
    req_end = pd.Timestamp("2023-12-31", tz="UTC")
    assert df_result["datetime"].min() >= req_start
    assert df_result["datetime"].max() <= req_end


# ── Scenario 5a: _scan_enriched_dataset happy-path — window clip + key mapping ─


def test_scan_enriched_dataset_clips_to_window_and_maps_keys(
    tmp_path: Path,
) -> None:
    """S5a: N-symbol x M-TF enriched fixtures → each DataFrame in scan result
    has correct key format "{safe_sym}_{tf_l}" and datetime within window.
    """
    # Arrange: 2 symbols x 2 TFs
    syms = [("BTCUSDT", "4h"), ("ETHUSDT", "1d")]
    full_range = pd.date_range("2020-01-01", "2025-12-31", freq="4h", tz="UTC")
    paths: list[Path] = []
    for safe_sym, tf_l in syms:
        path = tmp_path / f"{safe_sym}_{tf_l}_enriched.parquet"
        _make_enriched_parquet(path, full_range)
        paths.append(path)

    req_start = pd.Timestamp("2023-01-01", tz="UTC")
    req_end = pd.Timestamp("2023-06-30", tz="UTC")
    start_ms = int(req_start.value // 1_000_000)
    end_ms = int(req_end.value // 1_000_000)

    # Act
    result = _scan_enriched_dataset(paths, start_ms, end_ms)

    # Assert: one entry per (sym, tf)
    assert set(result.keys()) == {"BTCUSDT_4h", "ETHUSDT_1d"}
    for key, df in result.items():
        assert not df.empty, f"{key} must not be empty for overlapping window"
        assert df["datetime"].min() >= req_start, f"{key} min out of window"
        assert df["datetime"].max() <= req_end, f"{key} max out of window"


# ── Fix 1: exec_1m removed from admission gate (breadth fix) ──────────────────


def _make_ohlcv_frame(start: str, end: str, freq: str = "4h") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100.0,
        }
    )


def test_evaluate_symbol_data_sufficiency_passes_with_low_exec_1m_coverage() -> None:
    # S1-01: symbol with full 4h coverage but low exec_1m → pass=True
    frame_4h = _make_ohlcv_frame("2023-08-01", "2026-06-30", freq="4h")
    exec_1m = _make_ohlcv_frame("2026-06-01", "2026-06-30", freq="1min")
    symbol_map = {"4h": frame_4h, "exec_1m": exec_1m}

    result = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="MIDCAPUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2023-08-30",
        is_start="2023-10-31",
        oos_start="2026-01-01",
        oos_end="2026-06-30",
        require_exec_1m=True,
        warmup_bars_required=60,
    )

    assert result["pass"] is True
    assert result["reason"] == "ok"
    assert result["exec_1m_ok"] is False
    assert result["exec_1m_coverage"] < 0.95


def test_evaluate_symbol_data_sufficiency_still_rejects_short_warmup_regardless_of_exec_1m() -> None:
    # S2-01: warmup fail → pass=False regardless of exec_1m status
    # Frame starts close to is_start so bars_before_is < warmup_bars_required,
    # but still before fetch_start so fetch_ok=True (warmup takes priority).
    frame_4h = _make_ohlcv_frame("2023-08-28", "2026-06-30", freq="4h")
    symbol_map = {"4h": frame_4h}

    result = opt_data_utils.evaluate_symbol_data_sufficiency(
        symbol="NEWLISTUSDT",
        tf="4h",
        symbol_map=symbol_map,
        fetch_start="2023-08-30",
        is_start="2023-08-31",
        oos_start="2026-01-01",
        oos_end="2026-06-30",
        require_exec_1m=False,
        warmup_bars_required=60,
    )

    assert result["pass"] is False
    assert result["reason"] == "warmup_insufficient"


# ── Scenario 5b: _scan_enriched_dataset — missing timestamp col → graceful skip ─


def test_scan_enriched_dataset_skips_file_without_timestamp_column(
    tmp_path: Path,
) -> None:
    """S5b: enriched file lacking 'timestamp' column must be silently skipped.
    Result dict must NOT contain its key, and no exception propagates.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Arrange: write parquet without timestamp column
    path = tmp_path / "XYZUSDT_4h_enriched.parquet"
    dt_range = pd.date_range("2023-01-01", periods=10, freq="4h", tz="UTC")
    table = pa.table(
        {
            "datetime": pa.array(dt_range.to_pydatetime(), type=pa.timestamp("us", tz="UTC")),
            "close": pa.array([1.0] * 10, type=pa.float64()),
            # intentionally NO 'timestamp' column
        }
    )
    pq.write_table(table, str(path))  # type: ignore[no-untyped-call]

    start_ms = int(pd.Timestamp("2023-01-01", tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp("2023-06-30", tz="UTC").value // 1_000_000)

    # Act
    result = _scan_enriched_dataset([path], start_ms, end_ms)

    # Assert: key absent — graceful skip
    assert "XYZUSDT_4h" not in result


# ── Scenario 5c: _scan_enriched_dataset — window outside data range → empty df ─


def test_scan_enriched_dataset_returns_empty_df_for_non_overlapping_window(
    tmp_path: Path,
) -> None:
    """S5c: request window outside data range → result df is empty.
    Arrow prunes all row-groups; key present with 0-row df.
    """
    # Arrange: data 2020-2021 only
    path = tmp_path / "SOLUSDT_4h_enriched.parquet"
    data_range = pd.date_range("2020-01-01", "2021-12-31", freq="4h", tz="UTC")
    _make_enriched_parquet(path, data_range)

    # Request window 2023 — entirely outside data range
    start_ms = int(pd.Timestamp("2023-01-01", tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp("2023-06-30", tz="UTC").value // 1_000_000)

    # Act
    result = _scan_enriched_dataset([path], start_ms, end_ms)

    # Assert: key present but df is empty (all row-groups pruned)
    assert "SOLUSDT_4h" in result
    assert result["SOLUSDT_4h"].empty


# ── Scenario 6: exec_1m opt-out — Arrow fast-path bypassed, all syms → fallback ─


def test_load_futures_data_maps_exec_1m_routes_all_to_threadpool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S6: When use_exec_1m=True, valid_enriched symbols must NOT go through the Arrow
    path — they are routed to load_single_symbol_data ThreadPool fallback instead.
    Verified by: _scan_enriched_dataset call_count == 0 and load_single_symbol_data
    is called for every symbol.
    """
    # Arrange
    safe_sym = "BTCUSDT"
    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setenv("FUTURES_EXECUTION_MODE", "intrabar_1m")

    _setup_multi_tf_enriched(tmp_path, safe_sym)

    scan_call_count: list[int] = [0]
    _orig_scan = opt_data_utils._scan_enriched_dataset  # type: ignore[attr-defined]

    def _counting_scan(*args: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
        scan_call_count[0] += 1
        return _orig_scan(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(opt_data_utils, "_scan_enriched_dataset", _counting_scan)

    load_single_syms: list[str] = []
    _orig_load_single = opt_data_utils.load_single_symbol_data

    def _tracking_load_single(sym: str, *args: Any, **kwargs: Any) -> Any:
        load_single_syms.append(sym)
        # Return insufficient to keep test fast (no real data processing needed)
        return sym, None, None, True

    monkeypatch.setattr(opt_data_utils, "load_single_symbol_data", _tracking_load_single)
    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *_a, **_kw: 0)

    # Act
    _data_maps, _oos_data_maps, _valid_symbols = opt_data_utils.load_futures_data_maps_for_symbols(
        symbols=[safe_sym],
        tf="4h",
        fetch_start="2022-01-01",
        start="2022-07-01",
        is_end="2023-10-01",
        end="2023-12-31",
        skip_metrics=True,
        load_exec_1m=True,
    )

    # Assert: Arrow scan NOT called; symbol routed to load_single_symbol_data
    assert scan_call_count[0] == 0, "_scan_enriched_dataset must not be called when exec_1m=True"
    assert safe_sym in load_single_syms, "BTCUSDT must be processed via load_single_symbol_data fallback"


# ─── OPT-1: searchsorted equivalence ────────────────────────────────────────


def test_searchsorted_mask_equivalence(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """S2/S3: searchsorted must produce valid indices for boundary cases."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.domain.futures.optimization import opt_data_utils

    monkeypatch.setattr(opt_data_utils, "FUTURES_DATA_DIR", tmp_path)

    sym_raw = "BTC/USDT"
    safe_sym = sym_raw.replace("/", "_")
    tfs = ["4h", "1d", "1h"]
    base = pd.Timestamp("2021-01-01", tz="UTC")
    n = 6000
    datetimes = [base + pd.Timedelta(hours=i * 4) for i in range(n)]
    for tf_l in tfs:
        df = pd.DataFrame(
            {
                "timestamp": [int(t.value // 1_000_000) for t in datetimes],
                "datetime": datetimes,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )
        pq.write_table(pa.Table.from_pandas(df), tmp_path / f"{safe_sym}_{tf_l}_enriched.parquet")

    monkeypatch.setattr(opt_data_utils, "compute_segment_merge_index", lambda *a, **kw: 0)
    monkeypatch.setattr(opt_data_utils, "_append_stage_integrity", lambda *a, **kw: None)
    monkeypatch.setattr(opt_data_utils, "_feature_group_coverage", lambda *a, **kw: {})

    data_maps, _oos_maps, valid = opt_data_utils.load_futures_data_maps_for_symbols(
        symbols=[sym_raw],
        tf="4h",
        fetch_start="2022-01-01",
        start="2023-03-01",
        is_end="2023-06-01",
        end="2023-10-01",
        skip_metrics=True,
        load_exec_1m=False,
    )

    assert sym_raw in valid, f"{sym_raw} should be valid"
    is_map = data_maps[sym_raw]
    for tf_l in tfs:
        skey = f"is_start_idx_{tf_l}"
        assert skey in is_map, f"Missing is_start_idx_{tf_l}"
        assert 0 <= is_map[skey] <= len(is_map.get(tf_l, []))
