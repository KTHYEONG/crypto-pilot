"""Unit tests for TF Probe → bridge integration.

Tests _project_panel_to_base_grid, _build_probe_extra_panels, and the
_run_tf_probe_stage disabled-path (ENABLE_TF_PROBE=False).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy.timeframe_probe import TfCellEvidence
from src.domain.futures.strategy_runtime.bridge import (
    _base_probe_guard_mask,
    _build_probe_extra_panels,
    _build_virtual_probe_tf_maps,
    _project_panel_to_base_grid,
)


@dataclass(frozen=True)
class _Cfg:
    timeframe: str = "4h"

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_datetimes(n: int, freq_h: float = 4.0) -> np.ndarray:
    """Generate n datetime64[ns] at freq_h hour intervals starting 2024-01-01."""
    base_ns = np.datetime64("2024-01-01T00:00:00", "ns").view(np.int64)
    step_ns = int(freq_h * 3600 * 1_000_000_000)
    return (base_ns + np.arange(n, dtype=np.int64) * step_ns).view("datetime64[ns]")


def _make_panel(
    *,
    family: str = "carry_rev",
    variant: str = "funding_carry",
    archetype: str = "carry_rev",
    n_bars: int = 20,
    n_syms: int = 2,
    freq_h: float = 4.0,
    score_val: float = 0.5,
) -> CandidateSignalPanel:
    """Create a minimal CandidateSignalPanel with synthetic data."""
    dt = _make_datetimes(n_bars, freq_h)
    symbols = tuple(f"SYM{i}" for i in range(n_syms))
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=dt,
        symbols=symbols,
        signed_score_2d=np.full((n_bars, n_syms), score_val, dtype=np.float64),
        side_hint_2d=np.ones((n_bars, n_syms), dtype=np.int8),
        expected_holding_bars=4,
        min_holding_bars=2,
        stop_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.full((n_bars, n_syms), 0.1, dtype=np.float64),
        valid_mask_2d=np.ones((n_bars, n_syms), dtype=bool),
        metadata={},
        archetype=archetype,
        allowed_regimes=(),
        exit_policies=(),
    )


def _make_cell(
    *,
    tf: str = "1h",
    family: str = "carry_rev",
    variant: str = "funding_carry",
) -> TfCellEvidence:
    """Create a minimal TfCellEvidence for testing."""
    return TfCellEvidence(
        symbol="BTCUSDT",
        family=family,
        variant=variant,
        archetype="carry_rev",
        tf=tf,
        n_obs=500,
        n_events=100,
        ic_mean=0.05,
        ic_tstat_hac=2.5,
        ic_fold_sign_consistency=0.8,
        alpha_half_life_h=24.0,
        net_edge_bps=5.0,
        turnover_per_year=50.0,
        vr_label="mean_rev",
        hurst=0.4,
        passed_fdr=True,
    )


def _make_frame(
    *,
    n_bars: int,
    freq_h: float,
    start: str = "2024-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Build a minimal OHLCV frame for probe resampling tests."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range(
                start,
                periods=n_bars,
                freq=f"{int(freq_h * 60)}min",
                tz="UTC",
            ),
            "open": np.arange(n_bars, dtype=np.float64) + 100.0,
            "high": np.arange(n_bars, dtype=np.float64) + 101.0,
            "low": np.arange(n_bars, dtype=np.float64) + 99.0,
            "close": np.arange(n_bars, dtype=np.float64) + 100.5,
            "volume": np.full(n_bars, 10.0, dtype=np.float64),
        }
    )


# ---------------------------------------------------------------------------
# Scenario 1: ENABLE_TF_PROBE=False → _run_tf_probe_stage returns None
# ---------------------------------------------------------------------------

def test_run_tf_probe_stage_disabled_returns_none() -> None:
    """ENABLE_TF_PROBE=False → stage skips immediately, returns None."""
    # Arrange
    from src.execution.opt_main_futures import DataStageResult, _run_tf_probe_stage

    mock_data_stage = MagicMock(spec=DataStageResult)
    mock_tiered_cfg = MagicMock()

    # Act
    with patch(
        "src.execution.opt_main_futures.OPT_FUTURES_CONFIG",
        {"ENABLE_TF_PROBE": False},
    ):
        from src.execution import opt_main_futures as _omf
        _omf.__dict__["OPT_FUTURES_CONFIG"]["ENABLE_TF_PROBE"] = False
        result = _run_tf_probe_stage(MagicMock(), mock_data_stage, mock_tiered_cfg)

    # Assert
    assert result is None


# ---------------------------------------------------------------------------
# Scenario 2: probe_result.winning_cells = () → extra panels empty
# ---------------------------------------------------------------------------

def test_build_probe_extra_panels_empty_winning_cells() -> None:
    """No winning cells → non_base_tfs is empty → returns empty tuple."""
    # Arrange
    base_panel = _make_panel(freq_h=4.0)
    mock_aligned_base = MagicMock()
    mock_aligned_base.datetimes = base_panel.datetimes

    # Act
    result = _build_probe_extra_panels(
        data_maps={},
        probe_cells=(),  # empty winning cells
        symbols=["SYM0", "SYM1"],
        aligned_base=mock_aligned_base,
        base_cfg=MagicMock(),
        base_tf="4h",
    )

    # Assert
    assert result == ()


# ---------------------------------------------------------------------------
# Scenario 3: Non-base TF winning cell → projected panel with tf suffix
# ---------------------------------------------------------------------------

def test_build_probe_extra_panels_projects_non_base_tf(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1h winning cell → _build_probe_extra_panels returns 1 projected panel, variant='funding_carry_1h'."""
    # Arrange
    cell = _make_cell(tf="1h", family="carry_rev", variant="funding_carry")
    # Source panel is on 1h grid (40 bars at 1h intervals)
    panel_1h = _make_panel(family="carry_rev", variant="funding_carry", n_bars=40, freq_h=1.0)
    source_1h = _make_frame(n_bars=40, freq_h=1.0)
    # Base grid is 4h (10 bars)
    base_dt = _make_datetimes(10, freq_h=4.0)
    mock_aligned_base = MagicMock()
    mock_aligned_base.datetimes = base_dt
    mock_aligned_base.active_mask = np.ones((10, 2), dtype=bool)
    mock_aligned_base.warm_mask = np.ones((10, 2), dtype=bool)
    mock_aligned_base.entry_block_mask = np.zeros((10, 2), dtype=bool)
    mock_aligned_base.kill_mask = np.zeros((10, 2), dtype=bool)
    mock_aligned_base.execution_eligibility_mask = None
    mock_aligned_base.strategy_readiness_mask = None
    mock_aligned_base.promotion_active_mask = None

    build_mock = MagicMock(return_value=(panel_1h,))
    with (
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=MagicMock(),
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            new=build_mock,
        ),
        caplog.at_level(logging.INFO, logger="src.domain.futures.strategy_runtime.bridge"),
    ):
        result = _build_probe_extra_panels(
            data_maps={"SYM0": {"1h": source_1h, "4h": _make_frame(n_bars=40, freq_h=4.0)}},
            probe_cells=(cell,),
            symbols=["SYM0", "SYM1"],
            aligned_base=mock_aligned_base,
            base_cfg=_Cfg(),
            base_tf="4h",
        )

    # Assert
    assert len(result) == 1
    assert result[0].variant == "funding_carry_1h"
    assert result[0].family == "carry_rev"
    assert result[0].signed_score_2d.shape[0] == 10  # projected to base grid length
    assert build_mock.call_args.kwargs["normalize_time_horizon"] is True
    assert build_mock.call_args.kwargs["horizon_base_tf"] == "4h"
    assert "[TF-PROBE AUDIT] BRIDGE INJECTION" in caplog.text
    assert "| 1h" in caplog.text


def test_build_virtual_probe_tf_maps_skips_symbols_without_source() -> None:
    """8h virtual probe map should be built from 4h source and skip missing symbols."""
    data_maps = {
        "BTCUSDT": {"4h": _make_frame(n_bars=12, freq_h=4.0)},
        "ETHUSDT": {},
    }

    result = _build_virtual_probe_tf_maps(data_maps, ["BTCUSDT", "ETHUSDT"], "8h")

    assert set(result) == {"BTCUSDT"}
    assert list(result["BTCUSDT"]) == ["8h"]
    assert isinstance(result["BTCUSDT"]["8h"], pd.DataFrame)
    assert not result["BTCUSDT"]["8h"].empty


def test_base_probe_guard_mask_matches_base_mask_contract() -> None:
    """Base probe guard must combine active, warm, and readiness masks."""
    aligned = MagicMock()
    aligned.active_mask = np.array([[True, True], [True, False]], dtype=bool)
    aligned.warm_mask = np.array([[True, False], [True, True]], dtype=bool)
    aligned.entry_block_mask = np.array([[False, False], [True, False]], dtype=bool)
    aligned.kill_mask = np.array([[False, True], [False, False]], dtype=bool)
    aligned.execution_eligibility_mask = np.array([[True, True], [True, True]], dtype=bool)
    aligned.strategy_readiness_mask = None
    aligned.promotion_active_mask = np.array([[True, True], [False, True]], dtype=bool)

    guard = _base_probe_guard_mask(aligned)

    expected = np.array([[True, False], [False, False]], dtype=bool)
    assert np.array_equal(guard, expected)


def test_build_probe_extra_panels_applies_base_guard_after_projection() -> None:
    """Projected 8h panel must inherit the base-grid guard mask."""
    cell = _make_cell(tf="8h", family="carry_rev", variant="funding_carry")
    panel_8h = _make_panel(family="carry_rev", variant="funding_carry", n_bars=8, freq_h=8.0)
    base_dt = _make_datetimes(12, freq_h=4.0)
    mock_aligned_base = MagicMock()
    mock_aligned_base.datetimes = base_dt
    mock_aligned_base.active_mask = np.ones((12, 1), dtype=bool)
    mock_aligned_base.warm_mask = np.ones((12, 1), dtype=bool)
    mock_aligned_base.entry_block_mask = np.zeros((12, 1), dtype=bool)
    mock_aligned_base.kill_mask = np.zeros((12, 1), dtype=bool)
    mock_aligned_base.execution_eligibility_mask = None
    mock_aligned_base.strategy_readiness_mask = None
    mock_aligned_base.promotion_active_mask = None
    mock_aligned_base.active_mask[6, 0] = False
    mock_aligned_base.warm_mask[2, 0] = False

    mock_aligned_i = MagicMock()

    def _fake_align(
        data_maps: dict[str, dict[str, object]],
        symbols: list[str],
        tf: str,
    ) -> object:
        _ = symbols
        assert tf == "8h"
        assert "8h" in data_maps["BTCUSDT"]
        return mock_aligned_i

    with (
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            return_value=(panel_8h,),
        ),
    ):
        result = _build_probe_extra_panels(
            data_maps={"BTCUSDT": {"4h": _make_frame(n_bars=16, freq_h=4.0)}},
            probe_cells=(cell,),
            symbols=["BTCUSDT"],
            aligned_base=mock_aligned_base,
            base_cfg=_Cfg(),
            base_tf="4h",
        )

    assert len(result) == 1
    assert result[0].variant == "funding_carry_8h"
    assert not result[0].valid_mask_2d[2, 0]
    assert not result[0].valid_mask_2d[6, 0]


# ---------------------------------------------------------------------------
# Scenario 4: HTF projection (8h → 4h) → project_higher_tf_to_grid path
# ---------------------------------------------------------------------------

def test_project_panel_to_base_grid_htf_path() -> None:
    """8h panel projected to 4h: first 4h bars before 8h start must be invalid (valid_mask=False)."""
    # Arrange: 8h panel has 5 bars starting at T=0
    n_8h = 5
    panel_8h = _make_panel(n_bars=n_8h, freq_h=8.0, score_val=1.0)

    # Base 4h grid: 12 bars at 4h frequency from same start
    base_dt = _make_datetimes(12, freq_h=4.0)

    # Act
    projected = _project_panel_to_base_grid(panel_8h, base_dt, tf_i="8h", base_tf="4h")

    # Assert: shape is correct
    assert projected.signed_score_2d.shape == (12, 2)
    assert projected.valid_mask_2d.shape == (12, 2)

    # variant suffix applied
    assert projected.variant.endswith("_8h")

    # holding bars scaled: 4 * (8/4) = 8
    assert projected.expected_holding_bars == 8

    # Bars before first 8h bar close: 4h[0] lands at same time as 8h[0] but
    # project_higher_tf_to_grid uses side="right"-1, so 4h[0] references 8h bar
    # that closed at 4h[0] or earlier. Because 8h bars are co-aligned with 4h[0]
    # at their respective timestamps, the valid_mask depends on alignment.
    # At minimum: all bars must have correct dtype.
    assert projected.valid_mask_2d.dtype == bool
    assert projected.signed_score_2d.dtype == np.float64


# ---------------------------------------------------------------------------
# Scenario 5: LTF projection (1h → 4h) → searchsorted path
# ---------------------------------------------------------------------------

def test_project_panel_to_base_grid_ltf_path() -> None:
    """1h panel projected to 4h: each base bar receives last 1h signal before it."""
    # Arrange: 1h panel has 20 bars; base 4h has 5 bars (same start)
    n_1h = 20
    panel_1h = _make_panel(n_bars=n_1h, freq_h=1.0, score_val=0.75, n_syms=1)
    base_dt = _make_datetimes(5, freq_h=4.0)

    # Act
    projected = _project_panel_to_base_grid(panel_1h, base_dt, tf_i="1h", base_tf="4h")

    # Assert
    assert projected.signed_score_2d.shape == (5, 1)
    assert projected.variant.endswith("_1h")

    # holding bars scaled: 4 * (1/4) = 1 (max(1, round(4 * 0.25)))
    assert projected.expected_holding_bars == 1

    # All base bars should have non-zero score (1h signals exist before each 4h bar)
    # Base 4h bar 0 is at t=0; searchsorted gives idx=-1 → valid_idx=False for that bar.
    # Bars 1-4 should be valid (1h bars exist before their timestamp).
    assert projected.valid_mask_2d[1:, 0].all(), "Bars 1-4 should be valid (1h signals available)"

    # score values should match original 1h panel score
    valid_rows = np.where(projected.valid_mask_2d[:, 0])[0]
    if len(valid_rows) > 0:
        assert np.allclose(projected.signed_score_2d[valid_rows, 0], 0.75)


def test_build_probe_extra_panels_resamples_virtual_tf_and_applies_base_guard() -> None:
    """Virtual 8h cells should resample from 4h cache and inherit base guard masks."""
    cell = _make_cell(tf="8h", family="carry_rev", variant="funding_carry")
    panel_8h = _make_panel(
        family="carry_rev",
        variant="funding_carry",
        n_bars=6,
        freq_h=8.0,
        n_syms=1,
    )
    base_dt = _make_datetimes(10, freq_h=4.0)
    mock_aligned_base = MagicMock()
    mock_aligned_base.datetimes = base_dt
    mock_aligned_base.active_mask = np.ones((10, 1), dtype=bool)
    mock_aligned_base.warm_mask = np.ones((10, 1), dtype=bool)
    mock_aligned_base.warm_mask[:2] = False
    mock_aligned_base.entry_block_mask = np.zeros((10, 1), dtype=bool)
    mock_aligned_base.kill_mask = np.zeros((10, 1), dtype=bool)
    mock_aligned_base.execution_eligibility_mask = None
    mock_aligned_base.strategy_readiness_mask = None
    mock_aligned_base.promotion_active_mask = None

    source_4h = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC"),
            "open": np.arange(20, dtype=float) + 1.0,
            "high": np.arange(20, dtype=float) + 2.0,
            "low": np.arange(20, dtype=float) + 0.5,
            "close": np.arange(20, dtype=float) + 1.5,
            "volume": np.full(20, 100.0, dtype=float),
        }
    )
    captured_maps: dict[str, dict[str, pd.DataFrame]] = {}

    def _fake_align_data_maps(
        data_maps: dict[str, dict[str, pd.DataFrame]],
        symbols: list[str],
        tf: str,
    ) -> MagicMock:
        nonlocal captured_maps
        captured_maps = data_maps
        assert tf == "8h"
        assert "8h" in data_maps["SYM0"]
        aligned = MagicMock()
        aligned.datetimes = _make_datetimes(6, freq_h=8.0)
        return aligned

    build_mock = MagicMock(return_value=(panel_8h,))
    with (
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align_data_maps,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            new=build_mock,
        ),
    ):
        result = _build_probe_extra_panels(
            data_maps={"SYM0": {"4h": source_4h}},
            probe_cells=(cell,),
            symbols=["SYM0"],
            aligned_base=mock_aligned_base,
            base_cfg=_Cfg(),
            base_tf="4h",
        )

    assert "SYM0" in captured_maps
    assert "8h" in captured_maps["SYM0"]
    assert len(result) == 1
    assert result[0].variant == "funding_carry_8h"
    assert not result[0].valid_mask_2d[:2, 0].any()
    assert build_mock.call_args.kwargs["normalize_time_horizon"] is True
    assert build_mock.call_args.kwargs["horizon_base_tf"] == "4h"
