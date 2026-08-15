"""TDD test scenarios for L0/L1 memory-bound dataflow.

Scenarios M01-M03 from docs/specs/l0_l1_memory_bound_dataflow.md.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.application.futures.optimization.strategy_service import (
    _oos_covers_is_range,
    pick_strategy_data_maps,
)
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig
from src.domain.futures.alpha_foundry.memory import L0MemoryBudget, LtfExec1mPlan
from src.domain.futures.optimization.opt_data_utils import load_ltf_exec_1m_frame
from src.domain.futures.strategy.common.alignment import (
    _ALIGNED_DATA_MAPS_CACHE,
    AlignedMarketData,
    clear_aligned_data_maps_cache,
)

# ── M01: Core loader exec_1m isolation ─────────────────────────────────


def test_active_pipeline_requires_exec_1m_returns_false() -> None:
    """[M01] [LIMIT-01] _requires_exec_1m returns False even for gate mode."""
    from src.application.futures.runner.active_pipeline import _requires_exec_1m

    class _MockConfig:
        alpha_foundry = type("AF", (), {"mode": "gate"})()

    assert not _requires_exec_1m(_MockConfig())


def test_load_ltf_exec_1m_frame_missing_file_returns_none() -> None:
    """[M01] Missing parquet file returns None, not exception."""
    result = load_ltf_exec_1m_frame(
        symbol="NONEXISTENT",
        data_root=Path("/tmp"),  # noqa: S108
        start_datetime=pd.Timestamp("2026-01-01", tz="UTC"),
        end_datetime=pd.Timestamp("2026-01-02", tz="UTC"),
    )
    assert result is None


def test_load_ltf_exec_1m_frame_missing_optional_trades_loads_required_columns(tmp_path: Path) -> None:
    """[M03] Optional trades is synthesized downstream, not required from parquet."""
    futures_dir = tmp_path / "futures"
    futures_dir.mkdir()
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC"),
            "high": [2.0, 2.0],
            "low": [1.0, 1.0],
            "close": [1.5, 1.5],
            "volume": [10.0, 10.0],
            "taker_buy_base_volume": [5.0, 5.0],
            "quote_vol": [15.0, 15.0],
        }
    )
    frame.to_parquet(futures_dir / "BTCUSDT_1m.parquet", index=False)

    result = load_ltf_exec_1m_frame(
        symbol="BTCUSDT",
        data_root=tmp_path,
        start_datetime=pd.Timestamp("2026-01-01", tz="UTC"),
        end_datetime=pd.Timestamp("2026-01-02", tz="UTC"),
    )

    assert result is not None
    assert "trades" not in result.columns


def test_load_ltf_exec_1m_frame_reads_partitioned_ohlcv_layout(tmp_path: Path) -> None:
    """[M03] Streaming loader reads the canonical partitioned 1m cache."""
    cache_path = tmp_path / "futures" / "ohlcv" / "1m" / "BTCUSDT.parquet"
    cache_path.parent.mkdir(parents=True)
    timestamps = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [int(timestamp.value // 1_000_000) for timestamp in timestamps],
            "high": [2.0, 2.0],
            "low": [1.0, 1.0],
            "close": [1.5, 1.5],
            "volume": [10.0, 10.0],
            "taker_buy_base_volume": [5.0, 5.0],
            "quote_vol": [15.0, 15.0],
        }
    )
    frame.to_parquet(cache_path, index=False)

    result = load_ltf_exec_1m_frame(
        symbol="BTCUSDT",
        data_root=tmp_path,
        start_datetime=pd.Timestamp("2026-01-01", tz="UTC"),
        end_datetime=pd.Timestamp("2026-01-02", tz="UTC"),
    )

    assert result is not None
    assert len(result) == 2


# ── M02: OOS covers IS range ───────────────────────────────────────────


def test_oos_covers_is_range_true_when_covering() -> None:
    """[M02] [LIMIT-02] OOS covering full IS range returns True."""
    is_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-05", "2026-01-10", freq="1h", tz="UTC"),
            "close": np.random.randn(121),
        }
    )
    oos_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", "2026-01-15", freq="1h", tz="UTC"),
            "close": np.random.randn(337),
        }
    )
    assert _oos_covers_is_range(is_df, oos_df)


def test_oos_covers_is_range_false_when_not_covering() -> None:
    """[M02] [LIMIT-02] OOS missing IS start returns False."""
    is_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-05", "2026-01-10", freq="1h", tz="UTC"),
            "close": np.random.randn(121),
        }
    )
    oos_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-08", "2026-01-15", freq="1h", tz="UTC"),
            "close": np.random.randn(169),
        }
    )
    assert not _oos_covers_is_range(is_df, oos_df)


def test_oos_covers_is_range_empty_false() -> None:
    """[M02] Empty DataFrames return False."""
    is_df = pd.DataFrame({"datetime": pd.Series(dtype="datetime64[ns, UTC]"), "close": pd.Series(dtype=float)})
    oos_df = pd.DataFrame({"datetime": pd.Series(dtype="datetime64[ns, UTC]"), "close": pd.Series(dtype=float)})
    assert not _oos_covers_is_range(is_df, oos_df)


def test_pick_strategy_data_maps_reuses_oos_when_covering() -> None:
    """[M02] [LIMIT-02] When OOS covers IS, returned frame is the OOS reference (no concat)."""
    oos_dates = pd.date_range("2026-01-01", "2026-01-15", freq="1h", tz="UTC")
    is_dates = pd.date_range("2026-01-05", "2026-01-10", freq="1h", tz="UTC")
    oos_df = pd.DataFrame({"datetime": oos_dates, "close": np.random.randn(len(oos_dates))})
    is_df = pd.DataFrame({"datetime": is_dates, "close": np.random.randn(len(is_dates))})

    oos_maps = {"BTCUSDT": {"4h": oos_df}}
    is_maps = {"BTCUSDT": {"4h": is_df, "is_start_idx_4h": 0}}

    result = pick_strategy_data_maps(oos_maps, is_maps, ["BTCUSDT"], "4h")
    assert "4h" in result["BTCUSDT"]
    # OOS reference reuse should be the same object
    assert result["BTCUSDT"]["4h"] is oos_df


# ── M05: Alignment cache ───────────────────────────────────────────────


def test_clear_aligned_data_maps_cache_empties_cache() -> None:
    """[M05] [LIMIT-05] clear_aligned_data_maps_cache removes all entries."""
    _ALIGNED_DATA_MAPS_CACHE[(-1, ("TEST",), "4h")] = (None, {"TEST": (10, 5)})
    assert len(_ALIGNED_DATA_MAPS_CACHE) > 0
    clear_aligned_data_maps_cache()
    assert len(_ALIGNED_DATA_MAPS_CACHE) == 0


def test_ltf_bridge_uses_coverage_plan_and_streamer(mocker) -> None:  # type: ignore[no-untyped-def]
    """[M03/M04] Bridge never reads exec_1m maps and delegates bounded loading."""
    from src.domain.futures.strategy_runtime.bridge import _build_ltf_native_panels_for_l0

    datetimes = np.array(["2026-01-01T00:00:00", "2026-01-01T04:00:00"], dtype="datetime64[ns]")
    ones = np.ones((2, 1), dtype=np.float64)
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=ones,
        high_2d=ones,
        low_2d=ones,
        close_2d=ones,
        volume_2d=ones,
        funding_2d=ones,
        active_mask=np.ones((2, 1), dtype=bool),
        warm_mask=np.ones((2, 1), dtype=bool),
        entry_block_mask=np.zeros((2, 1), dtype=bool),
        kill_mask=np.zeros((2, 1), dtype=bool),
    )
    coverage = SimpleNamespace(covered_symbols=frozenset({"BTCUSDT"}))
    plan = LtfExec1mPlan(symbols=("BTCUSDT",), max_workers=1, skip_reason=None)
    mocker.patch(
        "src.domain.futures.alpha_foundry.entry_timing.resolve_1m_coverage_tier",
        return_value=coverage,
    )
    mocker.patch(
        "src.domain.futures.alpha_foundry.memory.resolve_effective_memory_budget",
        return_value=L0MemoryBudget(limit_mb=10_240, safety_margin_mb=512),
    )
    mocker.patch(
        "src.domain.futures.alpha_foundry.memory.resolve_ltf_exec_1m_plan",
        return_value=plan,
    )
    stream = mocker.patch(
        "src.domain.futures.signals.ltf_alpha.build_ltf_native_alpha_panels_streaming",
        return_value=(),
    )
    load_frame = mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_ltf_exec_1m_frame",
        return_value=None,
    )

    result = _build_ltf_native_panels_for_l0(
        data_maps={"BTCUSDT": {"exec_1m": pd.DataFrame()}},
        symbols=("BTCUSDT",),
        aligned=aligned,
        cfg=SimpleNamespace(),
        runtime_config=AlphaFoundryRuntimeConfig(),
    )

    assert result == ()
    assert stream.call_args.kwargs["plan"] == plan
    assert stream.call_args.kwargs["load_frame"]("BTCUSDT") is None
    load_frame.assert_called_once()
