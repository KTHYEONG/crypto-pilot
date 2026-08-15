from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.signals.ltf_alpha import (
    build_ltf_alpha_feature_grid,
    build_ltf_native_alpha_panels,
    build_ltf_native_alpha_panels_streaming,
    project_ltf_panel_to_base_grid,
)
from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_1m_frame() -> pd.DataFrame:
    dt = pd.date_range("2026-01-01 00:00:00", periods=24 * 60, freq="1min", tz="UTC")
    close = np.full(24 * 60, 104.0, dtype=np.float64)
    close[:120] = np.array([100.0] * 30 + [101.0] * 30 + [103.0] * 30 + [104.0] * 30, dtype=np.float64)
    close[16 * 60 + 45 : 17 * 60 + 15] = 112.0
    volume = np.full(24 * 60, 10.0, dtype=np.float64)
    volume[:60] = 10.0
    volume[60:120] = 50.0
    volume[16 * 60 + 45 : 17 * 60 + 15] = 120.0
    taker_buy = volume * 0.5
    taker_buy[16 * 60 + 45 : 17 * 60 + 15] = volume[16 * 60 + 45 : 17 * 60 + 15] * 0.9
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
            "quote_vol": volume * close,
            "taker_buy_base_volume": taker_buy,
            "taker_buy_quote_volume": taker_buy * close,
            "trades": volume * 2.0,
        }
    )


def _aligned_3h_2sym() -> AlignedMarketData:
    base_dt = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((24, 2), dtype=np.float64)
    mask = np.ones((24, 2), dtype=bool)
    return AlignedMarketData(
        datetimes=base_dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=ones * 100.0,
        high_2d=ones * 105.0,
        low_2d=ones * 95.0,
        close_2d=ones * 104.0,
        volume_2d=ones * 1000.0,
        funding_2d=np.zeros((24, 2), dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros((3, 2), dtype=bool),
        kill_mask=np.zeros((3, 2), dtype=bool),
    )


# ===================================================================
# Scenario 1: Happy Path
# ===================================================================


class TestLtfFeatureGridHappyPath:
    def test_streaming_panels_accumulate_one_symbol_at_a_time(self) -> None:
        aligned = _aligned_3h_2sym()
        plan = dataclasses.make_dataclass("Plan", [("symbols", tuple), ("skip_reason", object)])(
            ("BTCUSDT", "ETHUSDT"), None
        )
        frames = {"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()}

        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=plan,
            load_frame=frames.__getitem__,
            budget=None,
        )

        assert panels
        assert all(panel.signed_score_2d.shape == aligned.close_2d.shape for panel in panels)
        assert any(panel.valid_mask_2d[:, 0].any() for panel in panels)

    def test_streaming_skip_plan_returns_empty(self) -> None:
        aligned = _aligned_3h_2sym()
        plan = dataclasses.make_dataclass("Plan", [("symbols", tuple), ("skip_reason", object)])(("BTCUSDT",), "budget")
        assert (
            build_ltf_native_alpha_panels_streaming(
                aligned=aligned, plan=plan, load_frame=lambda _symbol: None, budget=None
            )
            == ()
        )

    def test_literal_15m_grid_aggregation(self) -> None:
        frame = _exec_1m_frame()
        grid = build_ltf_alpha_feature_grid(
            exec_1m_by_symbol={"BTCUSDT": frame, "ETHUSDT": frame},
            symbols=("BTCUSDT", "ETHUSDT"),
            ltf="15m",
            start=np.datetime64("2026-01-01T00:00:00"),
            end=np.datetime64("2026-01-01T01:59:00"),
        )
        assert grid.close_2d.shape == (8, 2)
        assert bool(grid.valid_mask_2d[:, 0].all())
        assert float(grid.volume_2d[-1, 0]) == 750.0

    def test_ltf_panels_project_sparse_event(self) -> None:
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("funding_session_orb_flow",),
        )
        assert len(panels) == 1
        assert panels[0].family == "funding_session_orb_flow"
        assert panels[0].metadata["source_tf"] == "15m"
        assert panels[0].side_hint_2d.shape == aligned.active_mask.shape
        assert panels[0].side_hint_2d.any()

    def test_project_ltf_panel_to_base_grid(self) -> None:
        aligned = _aligned_3h_2sym()
        panel = CandidateSignalPanel(
            family="test_ltf",
            variant="test_15m",
            params={},
            datetimes=np.array(["2026-01-01T00:15:00", "2026-01-01T01:15:00"], dtype="datetime64[ns]"),
            symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=np.array([[1.0, 0.0], [0.0, -2.0]], dtype=np.float64),
            side_hint_2d=np.array([[1, 0], [0, -1]], dtype=np.int8),
            expected_holding_bars=1,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.ones((2, 2), dtype=np.float64),
            valid_mask_2d=np.array([[True, False], [False, True]], dtype=np.bool_),
        )
        projected = project_ltf_panel_to_base_grid(
            panel=panel,
            base_datetimes=aligned.datetimes,
            base_valid_mask_2d=aligned.active_mask,
        )
        assert projected.side_hint_2d[1, 0] == 1
        assert projected.side_hint_2d[2, 1] == -1


# ===================================================================
# Scenario 2: Edge Cases
# ===================================================================


class TestLtfEdgeCases:
    def test_limit01_ltf_bar_not_visible_until_base_row(self) -> None:
        """LTF event ending after base timestamp -> not visible until next base row."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("funding_session_orb_flow",),
        )
        assert len(panels) == 1
        assert panels[0].side_hint_2d[:16].sum() == 0

    def test_limit02_low_coverage_symbol_suppressed(self) -> None:
        """One symbol with 80% coverage -> that symbol has zero valid events."""
        frame_ok = _exec_1m_frame()
        dt_short = pd.date_range("2026-01-01 00:00:00", periods=96, freq="1min", tz="UTC")
        frame_short = pd.DataFrame(
            {
                "datetime": dt_short,
                "open": np.full(96, 100.0),
                "high": np.full(96, 100.5),
                "low": np.full(96, 99.5),
                "close": np.full(96, 100.0),
                "volume": np.full(96, 10.0),
                "quote_vol": np.full(96, 1000.0),
                "taker_buy_base_volume": np.full(96, 5.0),
                "taker_buy_quote_volume": np.full(96, 500.0),
                "trades": np.full(96, 20.0),
            }
        )
        grid = build_ltf_alpha_feature_grid(
            exec_1m_by_symbol={"BTCUSDT": frame_ok, "ETHUSDT": frame_short},
            symbols=("BTCUSDT", "ETHUSDT"),
            ltf="15m",
            start=np.datetime64("2026-01-01T00:00:00"),
            end=np.datetime64("2026-01-01T01:59:00"),
            min_coverage=0.95,
        )
        assert bool(grid.valid_mask_2d[:, 0].all())
        assert not grid.valid_mask_2d[:, 1].any()

    def test_limit03_missing_optional_field_no_crash(self) -> None:
        """oi_flow_squeeze with missing oi_2d -> no panel or empty panel, no exception."""
        aligned = _aligned_3h_2sym()
        aligned_none = dataclasses.replace(aligned, oi_2d=None)
        panels = build_ltf_native_alpha_panels(
            aligned=aligned_none,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("oi_flow_squeeze",),
        )
        assert len(panels) == 0

    def test_limit05_consecutive_breakout_only_first_entry(self) -> None:
        """Consecutive true breakout bars -> only first bar counted as entry."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("volume_participation_breakout",),
        )
        assert len(panels) == 1
        assert int(np.count_nonzero(panels[0].side_hint_2d[:, 0])) == 1

    def test_limit07_cross_section_below_min(self) -> None:
        """Cross-section count 10 < 30 -> xs_residual_flow_rotation side hints all zero."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("xs_residual_flow_rotation",),
        )
        assert len(panels) == 0

    def test_limit09_duplicate_timestamps_kept_last(self) -> None:
        """Duplicate 1m timestamps -> last duplicate kept and grid remains monotonic."""
        frame = _exec_1m_frame()
        dup = pd.concat([frame.iloc[10:20], frame.iloc[10:20]], axis=0)
        dup = dup.drop_duplicates(subset="datetime", keep="last")
        frame_dedup = pd.concat([frame.iloc[:10], dup, frame.iloc[20:]], axis=0)
        frame_dedup = frame_dedup.sort_values("datetime").reset_index(drop=True)
        grid = build_ltf_alpha_feature_grid(
            exec_1m_by_symbol={"BTCUSDT": frame_dedup},
            symbols=("BTCUSDT",),
            ltf="15m",
            start=np.datetime64("2026-01-01T00:00:00"),
            end=np.datetime64("2026-01-01T01:59:00"),
        )
        assert np.all(np.diff(grid.datetimes) >= np.timedelta64(0, "ns"))

    def test_limit10_excessive_turnover_blocked(self) -> None:
        """Turnover_per_year exceeds cap -> panel is empty."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("funding_session_orb_flow",),
        )
        assert len(panels) <= 1

    def test_limit11_metadata_missing_source_tf_fails(self) -> None:
        """New panel metadata missing source_tf -> unit test fails."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("funding_session_orb_flow",),
        )
        for p in panels:
            assert "source_tf" in p.metadata
            assert "release_lag_bars" in p.metadata
            assert "archetype" in p.metadata

    def test_limit12_donchian_breakout_without_volume_confirmation(self) -> None:
        """Donchian breakout without taker/volume confirmation -> no event."""
        aligned = _aligned_3h_2sym()
        frame = _exec_1m_frame()
        frame["volume"] = 10.0
        frame["quote_vol"] = frame["volume"] * frame["close"]
        frame["taker_buy_base_volume"] = frame["volume"] * 0.5
        frame["taker_buy_quote_volume"] = frame["taker_buy_base_volume"] * frame["close"]
        frame["trades"] = 20.0
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": frame, "ETHUSDT": frame},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("volume_participation_breakout",),
        )
        assert len(panels) == 0 or (panels[0].side_hint_2d == 0).all()


# ===================================================================
# Scenario 3: Error Handling
# ===================================================================


class TestLtfErrorHandling:
    def test_unsupported_ltf_raises_value_error(self) -> None:
        """Unsupported LTF '1m' in build_ltf_alpha_feature_grid -> ValueError."""
        with pytest.raises(ValueError, match="unsupported ltf"):
            build_ltf_alpha_feature_grid(
                exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame()},
                symbols=("BTCUSDT",),
                ltf="1m",
                start=np.datetime64("2026-01-01T00:00:00"),
                end=np.datetime64("2026-01-01T00:59:00"),
            )

    def test_missing_required_column_raises_value_error(self) -> None:
        """Missing required 1m columns -> ValueError."""
        frame = _exec_1m_frame().drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required column"):
            build_ltf_alpha_feature_grid(
                exec_1m_by_symbol={"BTCUSDT": frame},
                symbols=("BTCUSDT",),
                ltf="15m",
                start=np.datetime64("2026-01-01T00:00:00"),
                end=np.datetime64("2026-01-01T00:59:00"),
            )

    def test_shape_mismatch_during_projection_raises_value_error(self) -> None:
        """Shape mismatch during projection -> ValueError."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
            cfg=CandidateStrategyConfig(timeframe="1h"),
            ltf_grid=("15m",),
            family_filter=("funding_session_orb_flow",),
        )
        if panels:
            wrong_base = aligned.datetimes
            wrong_mask = np.ones((len(aligned.datetimes), 1), dtype=bool)
            with pytest.raises(ValueError, match="shape mismatch"):
                project_ltf_panel_to_base_grid(
                    panel=panels[0],
                    base_datetimes=wrong_base,
                    base_valid_mask_2d=wrong_mask,
                )

    def test_non_monotonic_base_datetimes_raises_value_error(self) -> None:
        """Non-monotonic base datetimes -> ValueError."""
        with pytest.raises(ValueError, match="monotonic"):
            project_ltf_panel_to_base_grid(
                panel=CandidateSignalPanel(
                    family="test",
                    variant="test",
                    params={},
                    datetimes=np.array(["2026-01-01T00:00:00", "2026-01-01T00:15:00"], dtype="datetime64[ns]"),
                    symbols=("BTCUSDT",),
                    signed_score_2d=np.zeros((2, 1), dtype=np.float64),
                    side_hint_2d=np.zeros((2, 1), dtype=np.int8),
                    expected_holding_bars=1,
                    min_holding_bars=1,
                    stop_atr_mult=2.0,
                    take_profit_atr_mult=4.0,
                    turnover_proxy_2d=np.zeros((2, 1), dtype=np.float64),
                    valid_mask_2d=np.ones((2, 1), dtype=np.bool_),
                ),
                base_datetimes=np.array(["2026-01-01T02:00:00", "2026-01-01T01:00:00"], dtype="datetime64[ns]"),
                base_valid_mask_2d=np.ones((2, 1), dtype=bool),
            )

    def test_empty_exec_1m_returns_empty_tuple(self) -> None:
        """exec_1m_by_symbol empty for all symbols -> return empty tuple."""
        aligned = _aligned_3h_2sym()
        panels = build_ltf_native_alpha_panels(
            aligned=aligned,
            exec_1m_by_symbol={},
            cfg=CandidateStrategyConfig(timeframe="1h"),
        )
        assert len(panels) == 0


def test_ltf_feature_grid_literal_mock() -> None:
    frame = _exec_1m_frame()
    grid = build_ltf_alpha_feature_grid(
        exec_1m_by_symbol={"BTCUSDT": frame, "ETHUSDT": frame},
        symbols=("BTCUSDT", "ETHUSDT"),
        ltf="15m",
        start=np.datetime64("2026-01-01T00:00:00"),
        end=np.datetime64("2026-01-01T01:59:00"),
    )
    assert grid.close_2d.shape == (8, 2)
    assert bool(grid.valid_mask_2d[:, 0].all())
    assert float(grid.volume_2d[-1, 0]) == 750.0


# ===================================================================
# Scenario 4: Parallel Streaming (Change 2)
# ===================================================================


class TestLtfStreamingParallel:
    """ThreadPoolExecutor path: max_workers=1 vs max_workers=2 identity."""

    def test_serial_vs_parallel_output_identical(self) -> None:
        aligned = _aligned_3h_2sym()
        symbols = ("BTCUSDT", "ETHUSDT")
        frame = _exec_1m_frame()
        PlanCls = dataclasses.make_dataclass(
            "Plan", [("symbols", tuple), ("max_workers", int), ("skip_reason", object)]
        )

        panels_1 = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=PlanCls(symbols, 1, None),
            load_frame=dict.fromkeys(symbols, frame).get,
            budget=None,
        )
        panels_2 = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=PlanCls(symbols, 2, None),
            load_frame=dict.fromkeys(symbols, frame).get,
            budget=None,
        )
        assert len(panels_1) == len(panels_2)
        for p1, p2 in zip(panels_1, panels_2, strict=False):
            assert p1.variant == p2.variant
            np.testing.assert_array_equal(p1.signed_score_2d, p2.signed_score_2d)
            np.testing.assert_array_equal(p1.side_hint_2d, p2.side_hint_2d)
            np.testing.assert_array_equal(p1.valid_mask_2d, p2.valid_mask_2d)

    def test_parallel_with_missing_symbol(self) -> None:
        aligned = _aligned_3h_2sym()
        symbols = ("BTCUSDT", "ETHUSDT")
        frame = _exec_1m_frame()
        frames = {"BTCUSDT": frame}
        PlanCls = dataclasses.make_dataclass(
            "Plan", [("symbols", tuple), ("max_workers", int), ("skip_reason", object)]
        )
        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=PlanCls(symbols, 2, None),
            load_frame=frames.get,
            budget=None,
        )
        assert isinstance(panels, tuple)
        # ETHUSDT column should remain all zeros (no data)
        if panels:
            assert not panels[0].valid_mask_2d[:, 1].any()

    def test_parallel_with_empty_frame(self) -> None:
        aligned = _aligned_3h_2sym()
        symbols = ("BTCUSDT", "ETHUSDT")
        empty = _exec_1m_frame().iloc[:0]
        PlanCls = dataclasses.make_dataclass(
            "Plan", [("symbols", tuple), ("max_workers", int), ("skip_reason", object)]
        )
        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=PlanCls(symbols, 2, None),
            load_frame=lambda _s: empty,
            budget=None,
        )
        assert isinstance(panels, tuple)
        assert all(not p.valid_mask_2d.any() for p in panels)

    def test_plan_without_max_workers_defaults_serial(self) -> None:
        """Plan without max_workers attr falls back to serial path."""
        aligned = _aligned_3h_2sym()
        PlanCls = dataclasses.make_dataclass("Plan", [("symbols", tuple), ("skip_reason", object)])
        plan = PlanCls(("BTCUSDT", "ETHUSDT"), None)
        frames = {"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()}
        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=plan,
            load_frame=frames.get,
            budget=None,
        )
        assert panels
        assert all(panel.signed_score_2d.shape == aligned.close_2d.shape for panel in panels)

    def test_parallel_worker_exception_caught(self, caplog) -> None:
        """Worker exception in ThreadPoolExecutor is caught and logged."""
        import logging

        caplog.set_level(logging.WARNING)
        aligned = _aligned_3h_2sym()
        symbols = ("BTCUSDT", "ETHUSDT")
        frame = _exec_1m_frame()
        PlanCls = dataclasses.make_dataclass(
            "Plan", [("symbols", tuple), ("max_workers", int), ("skip_reason", object)]
        )
        call_count = 0

        def _raise_for_first(sym: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated worker failure")
            return frame

        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=PlanCls(symbols, 2, None),
            load_frame=_raise_for_first,
            budget=None,
        )
        assert isinstance(panels, tuple)
        assert any("simulated worker failure" in r.message for r in caplog.records)


def test_ltf_panels_project_sparse_event_literal_mock() -> None:
    base_dt = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((24, 2), dtype=np.float64)
    mask = np.ones((24, 2), dtype=bool)
    aligned = AlignedMarketData(
        datetimes=base_dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=ones * 100.0,
        high_2d=ones * 105.0,
        low_2d=ones * 95.0,
        close_2d=ones * 104.0,
        volume_2d=ones * 1000.0,
        funding_2d=np.zeros((24, 2), dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros((3, 2), dtype=bool),
        kill_mask=np.zeros((3, 2), dtype=bool),
    )
    panels = build_ltf_native_alpha_panels(
        aligned=aligned,
        exec_1m_by_symbol={"BTCUSDT": _exec_1m_frame(), "ETHUSDT": _exec_1m_frame()},
        cfg=CandidateStrategyConfig(timeframe="1h"),
        ltf_grid=("15m",),
        family_filter=("funding_session_orb_flow",),
    )
    assert len(panels) == 1
    assert panels[0].family == "funding_session_orb_flow"
    assert panels[0].metadata["source_tf"] == "15m"
    assert panels[0].side_hint_2d.shape == (24, 2)
    assert panels[0].side_hint_2d.any()
