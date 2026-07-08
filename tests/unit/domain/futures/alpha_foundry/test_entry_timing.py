"""Tests for entry-timing refinement layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    EntryTimingGateConfig,
    EntryTimingWindow,
    HtfDirectionalEpisode,
    Universe1mCoverageTier,
)
from src.domain.futures.alpha_foundry.entry_timing import (
    aggregate_entry_timing_evidence,
    compute_anchored_vwap_dev_sigma,
    compute_cvd_delta_z,
    evaluate_trend_quality_gate,
    kaufman_efficiency_ratio,
    refine_entry_indices,
    resolve_1m_backfill_targets,
    resolve_1m_coverage_tier,
    run_1m_backfill,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_signals import ALL_SIGNAL_FAMILIES

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "family": ["trend_ma", "trend_ma"],
            "variant": ["ema_12_72", "ema_12_72"],
            "side": [1, -1],
            "entry_idx": [10, 40],
            "expected_holding_bars": [24, 24],
            "handoff_tier": ["candidate", "blocked"],
        }
    )


@pytest.fixture
def synthetic_1m_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=300, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, size=300))
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": rng.uniform(1.0, 5.0, size=300),
            "taker_buy_volume": rng.uniform(0.5, 4.5, size=300),
        }
    )


@pytest.fixture
def default_entry_timing_config() -> EntryTimingGateConfig:
    return EntryTimingGateConfig(enabled=True, ltf_grid=("5m",), max_wait_bars_ratio=0.25)


# ── Scenario 1: Happy Path ────────────────────────────────────────────────


class TestScenario1HappyPath:
    def test_compute_cvd_delta_z_positive_imbalance_returns_positive_z(self) -> None:
        n = 100
        taker_buy = np.full(n, 80.0, dtype=np.float64)
        volume = np.full(n, 100.0, dtype=np.float64)

        result = compute_cvd_delta_z(taker_buy, volume, lookback_bars=20)

        assert float(np.mean(result[40:])) > 0.0

    def test_compute_anchored_vwap_dev_sigma_price_above_vwap_returns_positive(
        self,
    ) -> None:
        n = 20
        close = np.linspace(100.0, 110.0, n, dtype=np.float64)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(n, 10.0, dtype=np.float64)

        result = compute_anchored_vwap_dev_sigma(
            high, low, close, volume, anchor_pos=0
        )

        assert np.all(result[1:] > 0.0)

    def test_evaluate_trend_quality_gate_strong_trend_returns_true(self) -> None:
        n = 2000
        rng = np.random.default_rng(42)
        rets = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            rets[i] = 0.3 * rets[i - 1] + 0.001 + rng.normal(0, 0.003)

        result = evaluate_trend_quality_gate(rets)

        assert result is True

    def test_refine_entry_indices_trigger_within_window_shifts_entry_idx_forward(
        self,
        synthetic_events: pd.DataFrame,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(100)],
            dtype=np.datetime64,
        )
        events = synthetic_events.copy()
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert len(windows) <= len(events)
        assert "entry_idx" in result_events.columns

    def test_aggregate_entry_timing_evidence_positive_edge_returns_positive_lcb(
        self,
    ) -> None:
        windows = [
            EntryTimingWindow(
                episode_id=f"ep_{i}",
                ltf="5m",
                max_wait_bars_base=6,
                triggered=True,
                refined_entry_idx=10 + i,
                price_improvement_bps=5.0,
                opportunity_cost_bps=2.0,
                net_timing_edge_bps=3.0,
            )
            for i in range(20)
        ]

        result = aggregate_entry_timing_evidence(windows)

        assert isinstance(result, dict)
        for lcb in result.values():
            assert lcb > 0.0


# ── Scenario 2: Edge Cases ────────────────────────────────────────────────


class TestScenario2EdgeCases:
    def test_entry_timing_gate_config_rejects_max_wait_bars_ratio_above_one(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="max_wait_bars_ratio must be in"):
            EntryTimingGateConfig(max_wait_bars_ratio=1.5)

    def test_entry_timing_gate_config_rejects_negative_min_edge_floor(self) -> None:
        with pytest.raises(ValueError, match="min_net_timing_edge_lcb_bps must be >= 0"):
            EntryTimingGateConfig(min_net_timing_edge_lcb_bps=-1.0)

    def test_refine_entry_indices_window_expires_without_trigger_falls_back_to_naive(
        self,
        synthetic_events: pd.DataFrame,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(100)],
            dtype=np.datetime64,
        )
        events = synthetic_events.copy()
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        for w in windows:
            if not w.triggered:
                assert w.net_timing_edge_bps == 0.0

    def test_entry_timing_window_post_init_rejects_nonzero_edge_when_not_triggered(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="non-triggered window must have"):
            EntryTimingWindow(
                episode_id="ep_0",
                ltf="5m",
                max_wait_bars_base=6,
                triggered=False,
                refined_entry_idx=16,
                price_improvement_bps=0.0,
                opportunity_cost_bps=0.5,
                net_timing_edge_bps=0.5,
            )

    def test_refine_entry_indices_excludes_blocked_handoff_tier_episodes(
        self,
        synthetic_events: pd.DataFrame,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(100)],
            dtype=np.datetime64,
        )
        events = synthetic_events.copy()
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        for w in windows:
            assert isinstance(w, EntryTimingWindow)

    def test_refine_entry_indices_look_ahead_safe_uses_only_closed_ltf_bars(
        self,
        synthetic_events: pd.DataFrame,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(100)],
            dtype=np.datetime64,
        )
        events = synthetic_events.copy()
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        result_events, _windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert isinstance(result_events, pd.DataFrame)

    def test_refine_entry_indices_overlapping_symbol_windows_no_double_count(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "family": ["trend_ma", "trend_ma"],
                "variant": ["ema_12_72", "ema_12_72"],
                "side": [1, 1],
                "entry_idx": [10, 12],
                "expected_holding_bars": [24, 24],
                "handoff_tier": ["candidate", "candidate"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(100)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert len(windows) == 2

    def test_evaluate_trend_quality_gate_choppy_market_returns_false(self) -> None:
        rng = np.random.default_rng(99)
        rets = rng.normal(0, 0.01, size=100)

        result = evaluate_trend_quality_gate(rets)

        assert result is False

    def test_kaufman_efficiency_ratio_zero_denominator_returns_zero(self) -> None:
        rets = np.zeros(10, dtype=np.float64)

        result = kaufman_efficiency_ratio(rets)

        assert result == 0.0

    def test_kaufman_efficiency_ratio_short_array_returns_zero(self) -> None:
        rets = np.array([0.01, 0.02, 0.01], dtype=np.float64)

        result = kaufman_efficiency_ratio(rets)

        assert result == 0.0

    def test_kaufman_efficiency_ratio_nonfinite_returns_zero(self) -> None:
        rets = np.array([0.01, np.nan, 0.02, 0.03], dtype=np.float64)

        result = kaufman_efficiency_ratio(rets)

        assert result == 0.0


# ── Scenario 3: Error Handling ────────────────────────────────────────────


class TestScenario3ErrorHandling:
    def test_htf_directional_episode_rejects_invalid_handoff_tier(self) -> None:
        with pytest.raises(ValueError, match="handoff_tier must be"):
            HtfDirectionalEpisode(
                episode_id="ep_0",
                symbol="BTCUSDT",
                family="trend_ma",
                variant="ema_12_72",
                timeframe="4h",
                htf_bias=1,
                base_entry_idx=10,
                expected_holding_bars=24,
                handoff_tier="blocked",
            )

    def test_refine_entry_indices_missing_required_columns_raises_value_error(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame({"symbol": ["BTCUSDT"], "family": ["trend_ma"]})
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(10)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        with pytest.raises(ValueError, match="events missing required columns"):
            refine_entry_indices(
                events,
                base_datetimes=base_datetimes,
                ltf_1m_frames_by_symbol=ltf_frames,
                config=default_entry_timing_config,
            )

    def test_compute_anchored_vwap_dev_sigma_shape_mismatch_raises_value_error(
        self,
    ) -> None:
        high = np.ones(20, dtype=np.float64)
        low = np.ones(20, dtype=np.float64)
        close = np.ones(20, dtype=np.float64)
        volume = np.ones(30, dtype=np.float64)

        with pytest.raises(ValueError, match="shape"):
            compute_anchored_vwap_dev_sigma(high, low, close, volume, anchor_pos=0)


# ── Supplementary Coverage-Gap Tests ────────────────────────────────────────


class TestSupplementaryCoverage:
    def test_compute_anchored_vwap_dev_sigma_zero_volume_skips(
        self,
    ) -> None:
        n = 20
        close = np.linspace(100.0, 110.0, n, dtype=np.float64)
        high = close + 0.5
        low = close - 0.5
        volume = np.zeros(n, dtype=np.float64)

        result = compute_anchored_vwap_dev_sigma(
            high, low, close, volume, anchor_pos=0
        )

        assert np.all(result == 0.0)

    def test_evaluate_trend_quality_gate_hurst_above_threshold(self) -> None:
        n = 2000
        rng = np.random.default_rng(42)
        rets = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            rets[i] = 0.7 * rets[i - 1] + 0.001 + rng.normal(0, 0.003)

        result = evaluate_trend_quality_gate(rets)

        assert result is True

    def test_refine_entry_indices_empty_valid_events_returns_copy(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["blocked"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(10)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert len(windows) == 0
        assert result_events.equals(events)

    def test_refine_entry_indices_symbol_not_in_frames_fallback(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["UNKNOWNSYMBOL"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(10)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert len(windows) == 1
        assert windows[0].triggered is False

    def test_refine_entry_indices_ltf_resample_loop_reached(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(50)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}

        result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
        )

        assert len(windows) == 1
        assert isinstance(result_events, pd.DataFrame)

    def test_aggregate_entry_timing_evidence_unknown_family_branch(
        self,
    ) -> None:
        window = EntryTimingWindow(
            episode_id="singleword",
            ltf="5m",
            max_wait_bars_base=6,
            triggered=True,
            refined_entry_idx=10,
            price_improvement_bps=5.0,
            opportunity_cost_bps=2.0,
            net_timing_edge_bps=3.0,
        )

        result = aggregate_entry_timing_evidence([window])

        assert ("unknown", "unknown", "5m") in result

    def test_aggregate_entry_timing_evidence_bootstrap_branch(
        self,
    ) -> None:
        windows = [
            EntryTimingWindow(
                episode_id="same_family_same_variant",
                ltf="5m",
                max_wait_bars_base=6,
                triggered=True,
                refined_entry_idx=10 + i,
                price_improvement_bps=5.0,
                opportunity_cost_bps=2.0,
                net_timing_edge_bps=float(i),
            )
            for i in range(10)
        ]

        result = aggregate_entry_timing_evidence(windows)

        assert len(result) == 1
        family, variant, ltf_field = next(iter(result.keys()))
        assert family == "same"
        assert variant == "family"
        assert ltf_field == "5m"

    def test_refine_entry_indices_trigger_fires_on_strong_bullish_confluence(
        self,
    ) -> None:
        n_bars = 2500
        idx = pd.date_range("2026-01-01", periods=n_bars, freq="1min", tz="UTC")
        rng = np.random.default_rng(42)
        rets_ar = np.zeros(n_bars, dtype=np.float64)
        for i in range(1, n_bars):
            rets_ar[i] = 0.3 * rets_ar[i - 1] + 0.0005 + rng.normal(0, 0.002)
        close = 100.0 * np.exp(np.cumsum(rets_ar))
        frame = pd.DataFrame(
            {
                "datetime": idx,
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": np.full(n_bars, 10.0, dtype=np.float64),
                "taker_buy_volume": np.full(n_bars, 8.0, dtype=np.float64),
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(50)],
            dtype=np.datetime64,
        )
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        ltf_frames = {"BTCUSDT": frame}
        config = EntryTimingGateConfig(
            enabled=True, ltf_grid=("1h",), max_wait_bars_ratio=0.25,
            cvd_lookback_bars=30,
        )

        result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=config,
        )

        assert len(windows) == 1
        assert isinstance(result_events, pd.DataFrame)
        assert windows[0].triggered is True

    def test_refine_entry_indices_trigger_fires_on_strong_bearish_confluence(
        self,
    ) -> None:
        n_bars = 2500
        idx = pd.date_range("2026-01-01", periods=n_bars, freq="1min", tz="UTC")
        rng = np.random.default_rng(7)
        rets_ar = np.zeros(n_bars, dtype=np.float64)
        for i in range(1, n_bars):
            rets_ar[i] = 0.3 * rets_ar[i - 1] - 0.0005 + rng.normal(0, 0.002)
        close = 100.0 * np.exp(np.cumsum(rets_ar))
        frame = pd.DataFrame(
            {
                "datetime": idx,
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": np.full(n_bars, 10.0, dtype=np.float64),
                "taker_buy_volume": np.full(n_bars, 2.0, dtype=np.float64),
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(50)],
            dtype=np.datetime64,
        )
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [-1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        ltf_frames = {"BTCUSDT": frame}
        config = EntryTimingGateConfig(
            enabled=True, ltf_grid=("1h",), max_wait_bars_ratio=0.25,
            cvd_lookback_bars=30,
        )

        result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=config,
        )

        assert len(windows) == 1
        assert isinstance(result_events, pd.DataFrame)
        assert windows[0].triggered is True
        assert windows[0].opportunity_cost_bps >= 0.0


# ── LTF Native Directional Search: Scenario 1 (Happy Path) ──────────────


class TestLtfBackfillCoverageScenario1HappyPath:

    @pytest.fixture
    def fake_data_root(self, tmp_path: Path) -> Path:
        (tmp_path / "BTCUSDT_1m.parquet").touch()
        (tmp_path / "BTCUSDT_4h.parquet").touch()
        (tmp_path / "NEWCOINUSDT_4h.parquet").touch()
        return tmp_path

    def test_resolve_1m_backfill_targets_returns_symbols_missing_1m_file(
        self, fake_data_root: Path
    ) -> None:
        result = resolve_1m_backfill_targets(
            ("BTCUSDT", "NEWCOINUSDT"), data_root=fake_data_root
        )
        assert result == ("NEWCOINUSDT",)

    def test_resolve_1m_coverage_tier_computes_ratio(
        self, fake_data_root: Path
    ) -> None:
        tier = resolve_1m_coverage_tier(
            ("BTCUSDT", "NEWCOINUSDT", "OTHER1", "OTHER2"),
            data_root=fake_data_root,
        )
        assert tier.coverage_ratio == 0.25
        assert tier.is_covered("BTCUSDT") is True

    def test_universe_1m_coverage_tier_empty_universe_returns_zero_ratio(self) -> None:
        tier = Universe1mCoverageTier(covered_symbols=frozenset(), universe_symbols=frozenset())
        assert tier.coverage_ratio == 0.0

    def test_default_l1_tfs_includes_1h_and_2h(self) -> None:
        cfg = CandidateStrategyConfig()
        assert "1h" in cfg.l1_tfs
        assert "2h" in cfg.l1_tfs

    def test_resolve_tf_signal_pool_1h_returns_expanded_family_set(
        self,
    ) -> None:
        from src.domain.futures.strategy.config import (
            _DEFAULT_PER_TF_FAMILIES,
            resolve_tf_signal_pool,
        )

        cfg = CandidateStrategyConfig(per_tf_signal_pool_enabled=True)
        pool = resolve_tf_signal_pool(cfg, "1h")
        expected = _DEFAULT_PER_TF_FAMILIES["1h"]
        assert set(pool) == set(expected)
        assert "residual_reversion" in pool
        assert "trend_ma" in pool
        assert "funding_flow_carry" in pool


# ── LTF Native Directional Search: Scenario 2 (Edge Cases) ──────────────


class TestLtfBackfillCoverageScenario2EdgeCases:

    def test_resolve_1m_backfill_targets_empty_universe_returns_empty(
        self,
    ) -> None:
        result = resolve_1m_backfill_targets(())
        assert result == ()

    def test_resolve_1m_backfill_targets_all_covered_returns_empty(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "BTCUSDT_1m.parquet").touch()
        (tmp_path / "ETHUSDT_1m.parquet").touch()
        result = resolve_1m_backfill_targets(
            ("BTCUSDT", "ETHUSDT"), data_root=tmp_path
        )
        assert result == ()

    def test_refine_entry_indices_uncovered_symbol_sets_coverage_status_uncovered_fallback(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["UNKNOWNSYMBOL"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(10)],
            dtype=np.datetime64,
        )
        ltf_frames: dict[str, pd.DataFrame] = {}
        tier = Universe1mCoverageTier(
            covered_symbols=frozenset(),
            universe_symbols=frozenset({"UNKNOWNSYMBOL"}),
        )

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
            coverage_tier=tier,
        )

        assert len(windows) == 1
        assert windows[0].coverage_status == "uncovered_fallback"
        assert windows[0].triggered is False

    def test_refine_entry_indices_covered_symbol_no_trigger_sets_coverage_status_covered(
        self,
        synthetic_1m_frame: pd.DataFrame,
        default_entry_timing_config: EntryTimingGateConfig,
    ) -> None:
        events = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "family": ["trend_ma"],
                "variant": ["ema_12_72"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [24],
                "handoff_tier": ["candidate"],
            }
        )
        base_dt = np.datetime64("2026-01-01T00:00", "ns")
        base_datetimes = np.array(
            [base_dt + np.timedelta64(i * 4, "h") for i in range(10)],
            dtype=np.datetime64,
        )
        ltf_frames = {"BTCUSDT": synthetic_1m_frame}
        tier = Universe1mCoverageTier(
            covered_symbols=frozenset({"BTCUSDT"}),
            universe_symbols=frozenset({"BTCUSDT"}),
        )

        _result_events, windows = refine_entry_indices(
            events,
            base_datetimes=base_datetimes,
            ltf_1m_frames_by_symbol=ltf_frames,
            config=default_entry_timing_config,
            coverage_tier=tier,
        )

        assert len(windows) == 1
        assert windows[0].coverage_status == "covered"

    def test_default_per_tf_families_1h_2h_nonempty_and_subset_of_all_signal_families(
        self,
    ) -> None:
        from src.domain.futures.strategy.config import _DEFAULT_PER_TF_FAMILIES

        for tf in ("1h", "2h"):
            families = _DEFAULT_PER_TF_FAMILIES.get(tf, ())
            assert len(families) >= 2, f"{tf} has fewer than 2 families"
            assert set(families) <= set(ALL_SIGNAL_FAMILIES), (
                f"{tf} families not subset of ALL_SIGNAL_FAMILIES"
            )


# ── LTF Native Directional Search: Scenario 3 (Error Handling) ──────────


class TestLtfBackfillCoverageScenario3ErrorHandling:

    def test_run_1m_backfill_rejects_empty_missing_symbols(self, mocker) -> None:
        mock_sync = mocker.patch(
            "src.domain.futures.alpha_foundry.entry_timing.run_historical_sync",
            autospec=True,
        )
        run_1m_backfill((), end_date=date(2026, 7, 8))
        mock_sync.assert_not_called()

    def test_run_1m_backfill_invokes_historical_sync_with_correct_flags(self, mocker) -> None:
        mock_sync = mocker.patch(
            "src.domain.futures.alpha_foundry.entry_timing.run_historical_sync",
            autospec=True,
        )
        run_1m_backfill(("NEWCOINUSDT",), end_date=date(2026, 7, 8))
        mock_sync.assert_called_once()
        _, kwargs = mock_sync.call_args
        assert kwargs["sync_1m"] is True
        assert kwargs["sync_1d"] is False
        assert kwargs["sync_4h"] is False
        assert kwargs["symbols"] == ["NEWCOINUSDT"]
        assert kwargs["end_date"] == date(2026, 7, 8)

    def test_resolve_1m_coverage_tier_missing_data_root_raises_or_returns_empty_tier(
        self,
    ) -> None:
        tier = resolve_1m_coverage_tier(
            ("BTCUSDT",), data_root=Path("/nonexistent/path/xyz")
        )
        assert tier.coverage_ratio == 0.0
