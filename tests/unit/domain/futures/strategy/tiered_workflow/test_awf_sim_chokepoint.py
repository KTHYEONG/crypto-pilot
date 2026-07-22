from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    build_l2_simulation_cache,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
)
from src.domain.futures.strategy.walk_forward import WFFold


def _make_aligned(n_bars: int = 20, n_sym: int = 2) -> MagicMock:
    close = np.ones((n_bars, n_sym), dtype=np.float64) * 100.0
    aligned = MagicMock()
    aligned.symbols = ("BTC", "ETH") if n_sym >= 2 else ("BTC",)
    aligned.close_2d = close
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    aligned.funding_2d = np.zeros((n_bars, n_sym), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_sym), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_sym, dtype=np.float64)
    return aligned


def _make_signal_batch(n_bars: int = 20) -> ValidatedSignalBatch:
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    return ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=0.0,
                expected_gross_bps=20.0,
                q10_net_bps=0.0,
                q10_gross_bps=10.0,
                q90_net_bps=0.0,
                q90_gross_bps=30.0,
                expected_holding_bars=1,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="ETH",
                strategy_id="trend:fast",
                activation_context="all",
                side=-1,
                expected_net_bps=0.0,
                expected_gross_bps=5.0,
                q10_net_bps=0.0,
                q10_gross_bps=2.0,
                q90_net_bps=0.0,
                q90_gross_bps=8.0,
                expected_holding_bars=1,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=1,
        end_idx=3,
        symbols=("BTC", "ETH"),
        registry_version="test",
        model_version="test",
    )


def _make_config() -> Layer2AllocationConfig:
    return Layer2AllocationConfig(
        k_rank=2,
        rank_buffer=0,
        kelly_fraction=0.5,
        no_trade_band=0.0,
        rebalance_bars=1,
    )


class TestL2ChokepointDiagnostics:
    @pytest.fixture(autouse=True)
    def _force_opt_main_futures_propagate(self):
        logger = logging.getLogger("opt_main_futures")
        saved = logger.propagate
        logger.propagate = True
        try:
            yield
        finally:
            logger.propagate = saved

    def test_chokepoint_logs_are_monotonically_non_increasing(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        logging.getLogger("opt_main_futures").setLevel(logging.DEBUG)
        aligned = _make_aligned(n_bars=10)
        signal_batch = _make_signal_batch(n_bars=10)
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        fold = WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=0, oos_end=3)
        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")

        _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=(fold,),
            config=config,
            caps=caps,
            tf="4h",
        )

        chokepoint_records = [r for r in caplog.records if "stage=post_" in r.getMessage()]
        stage_tags = set()
        for rec in chokepoint_records:
            msg = rec.getMessage()
            for tag in ("post_resolve", "post_c4_filter", "post_bucket_routing", "post_netting"):
                if f"stage={tag}" in msg:
                    stage_tags.add(tag)
        expected = {"post_resolve", "post_c4_filter", "post_bucket_routing", "post_netting"}
        missing = expected - stage_tags
        assert not missing, f"Missing [L2-CHOKEPOINT] stages: {missing}"

    def test_chokepoint_logs_not_fired_at_info_level(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        aligned = _make_aligned(n_bars=10)
        signal_batch = _make_signal_batch(n_bars=10)
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        fold = WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=0, oos_end=3)
        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")

        _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=(fold,),
            config=config,
            caps=caps,
            tf="4h",
        )

        chokepoint_records = [r for r in caplog.records if "stage=post_" in r.getMessage()]
        assert len(chokepoint_records) == 0


class TestC4HandoffOverride:
    """Tests for the C4 TF-inclusion gate handoff override fix (Part B Phase 2)."""

    @pytest.fixture(autouse=True)
    def _force_opt_main_futures_propagate(self):
        logger = logging.getLogger("opt_main_futures")
        saved = logger.propagate
        logger.propagate = True
        try:
            yield
        finally:
            logger.propagate = saved

    def _build_c4_test_cache(self, n_bars: int = 10, n_sym: int = 2):
        aligned = _make_aligned(n_bars=n_bars, n_sym=n_sym)
        signal_batch = _make_signal_batch(n_bars=n_bars)
        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
        return aligned, signal_batch, cache

    def _c4_fold(self, n_bars: int = 10) -> WFFold:
        """Fold with fit_start < oos_start to trigger C4 gate computation."""
        return WFFold(fit_start=0, fit_end=n_bars // 2, cal_start=n_bars // 2, cal_end=n_bars // 2, oos_start=n_bars // 2, oos_end=n_bars)

    def test_c4_gate_handoff_override_includes_excluded_tf_with_admitted_sleeves(self, caplog) -> None:
        """C4 excludes TF but handoff admitted sleeves → override fires."""
        aligned, signal_batch, cache = self._build_c4_test_cache(n_bars=10)
        n_sleeves = cache.signal_mask_2d.shape[1]
        mask = np.ones(n_sleeves, dtype=np.bool_)
        sleeve_tf = cache.sleeve_to_tf[0]  # actual TF from built cache (e.g. "")
        cache = replace(
            cache,
            per_tf_edge_by_fold=({"other_tf": 0.1, sleeve_tf: -0.01},),
            handoff_sleeve_mask_by_fold=(mask,),
        )
        fold = self._c4_fold()
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        caplog.set_level(logging.DEBUG)
        logging.getLogger("opt_main_futures").setLevel(logging.DEBUG)

        _run_awf_simulation(
            cache=cache, signal_batch=signal_batch, aligned=aligned,
            awf_folds=(fold,), config=config, caps=caps, tf="4h",
        )

        override_logs = [
            r.getMessage() for r in caplog.records
            if "handoff_override" in r.getMessage()
        ]
        assert override_logs, "Expected [L2-TFGATE] handoff_override adds=... log"

    def test_c4_gate_handoff_override_noop_when_mask_by_fold_empty(self, caplog) -> None:
        """Empty handoff_sleeve_mask_by_fold → override is a no-op."""
        aligned, signal_batch, cache = self._build_c4_test_cache(n_bars=10)
        sleeve_tf = cache.sleeve_to_tf[0]
        cache = replace(
            cache,
            per_tf_edge_by_fold=({sleeve_tf: -0.01},),
            handoff_sleeve_mask_by_fold=(),
        )
        fold = self._c4_fold()
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        caplog.set_level(logging.DEBUG)
        logging.getLogger("opt_main_futures").setLevel(logging.DEBUG)

        _run_awf_simulation(
            cache=cache, signal_batch=signal_batch, aligned=aligned,
            awf_folds=(fold,), config=config, caps=caps, tf="4h",
        )

        override_logs = [
            r.getMessage() for r in caplog.records
            if "handoff_override" in r.getMessage()
        ]
        assert not override_logs, "handoff_override should NOT fire with empty mask"

    def test_c4_gate_handoff_override_idempotent_when_already_included(self, caplog) -> None:
        """_included already has all handoff-admitted TFs → no change."""
        aligned, signal_batch, cache = self._build_c4_test_cache(n_bars=10)
        n_sleeves = cache.signal_mask_2d.shape[1]
        mask = np.ones(n_sleeves, dtype=np.bool_)
        sleeve_tf = cache.sleeve_to_tf[0]  # actual TF from built cache (e.g. "")
        cache = replace(
            cache,
            per_tf_edge_by_fold=({sleeve_tf: 0.1},),
            handoff_sleeve_mask_by_fold=(mask,),
        )
        fold = self._c4_fold()
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        caplog.set_level(logging.DEBUG)
        logging.getLogger("opt_main_futures").setLevel(logging.DEBUG)

        _run_awf_simulation(
            cache=cache, signal_batch=signal_batch, aligned=aligned,
            awf_folds=(fold,), config=config, caps=caps, tf="4h",
        )

        override_logs = [
            r.getMessage() for r in caplog.records
            if "handoff_override" in r.getMessage()
        ]
        assert not override_logs, "handoff_override should NOT fire when all TFs already included"

    def test_c4_gate_empty_included_fallback_path_unchanged(self, caplog) -> None:
        """Empty _included fallback (ALL TFs) unchanged by this fix."""
        aligned, signal_batch, cache = self._build_c4_test_cache(n_bars=10)
        n_sleeves = cache.signal_mask_2d.shape[1]
        mask = np.ones(n_sleeves, dtype=np.bool_)
        sleeve_tf = cache.sleeve_to_tf[0]
        cache = replace(
            cache,
            per_tf_edge_by_fold=({sleeve_tf: -0.01},),
            handoff_sleeve_mask_by_fold=(mask,),
        )
        fold = self._c4_fold()
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        caplog.set_level(logging.DEBUG)
        logging.getLogger("opt_main_futures").setLevel(logging.DEBUG)

        _run_awf_simulation(
            cache=cache, signal_batch=signal_batch, aligned=aligned,
            awf_folds=(fold,), config=config, caps=caps, tf="4h",
        )

        fallback_logs = [
            r.getMessage() for r in caplog.records
            if "included_tfs=∅" in r.getMessage()
        ]
        assert fallback_logs, "Expected [L2-TFGATE] included_tfs=∅ ... fallback: ALL TFs"
        override_logs = [
            r.getMessage() for r in caplog.records
            if "handoff_override" in r.getMessage()
        ]
        assert not override_logs, "handoff_override should NOT fire when _included is empty"
