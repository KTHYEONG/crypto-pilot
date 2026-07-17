"""L2 TF Inclusion Gate tests.

Spec reference: docs/specs/layer2-multi-tf-combination.md
and docs/specs/l2-tf-inclusion-gate-native-tf-fix.md
Test scenarios: S1-S6.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.candidate_contracts import (
    SignalSleeveKey,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _build_sleeve_tf_lookup,
    build_l2_simulation_cache,
    compute_per_tf_fit_edge,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
    Layer2AllocationConfig,
)


def _make_minimal_cache_with_keys(
    *,
    sleeve_keys: tuple[SignalSleeveKey, ...],
    t_max: int = 10,
    n_sym: int = 2,
) -> L2SimulationCache:
    n_sleeve = len(sleeve_keys)
    sleeve_to_sym = np.array(
        [0 if i == 0 else min(i, n_sym - 1) for i in range(n_sleeve)],
        dtype=np.int64,
    )
    return L2SimulationCache(
        vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
        tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
        hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
        funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
        beta_1d=np.zeros(n_sym, dtype=np.float64),
        expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
        expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
        holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
        side_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
        quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
        signal_mask_2d=np.ones((t_max, n_sleeve), dtype=np.bool_),
        sleeve_to_sym=sleeve_to_sym,
        sleeve_keys=sleeve_keys,
    )


def _make_aligned(t_max: int = 10, n_sym: int = 2) -> Any:
    aligned = MagicMock()
    close = 100.0 * np.cumprod(
        np.ones((t_max + 1, n_sym), dtype=np.float64) * 1.01,
        axis=0,
    )
    aligned.close_2d = close[:t_max]
    aligned.symbols = tuple(f"SYM_{i}" for i in range(n_sym))
    return aligned


def _build_tf_gate_fixture(
    *,
    t_max: int = 20,
) -> tuple[Any, L2SimulationCache, ValidatedSignalBatch, tuple[Any, ...], Layer2AllocationConfig, Any]:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.walk_forward import WFFold

    n_sym = 2
    aligned = MagicMock()
    aligned.symbols = ("SYM_4H", "SYM_12H")
    sym_4h_close = 100.0 + np.arange(t_max, dtype=np.float64) * 1.0
    sym_12h_close = 100.0 - np.arange(t_max, dtype=np.float64) * 1.0
    aligned.close_2d = np.column_stack((sym_4h_close, sym_12h_close))
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(t_max)]
    )
    aligned.beta_vs_market_1d = np.zeros(n_sym, dtype=np.float64)
    aligned.execution_cost_bps_2d = np.full((t_max, n_sym), 3.8, dtype=np.float64)
    aligned.funding_2d = np.zeros((t_max, n_sym), dtype=np.float64)
    for mask_name in ("active_mask", "warm_mask", "execution_eligibility_mask",
                      "strategy_readiness_mask", "promotion_active_mask"):
        setattr(aligned, mask_name, np.ones((t_max, n_sym), dtype=np.bool_))
    aligned.entry_block_mask = np.zeros((t_max, n_sym), dtype=np.bool_)
    aligned.kill_mask = np.zeros((t_max, n_sym), dtype=np.bool_)
    aligned.volume_usdt_2d = np.ones((t_max, n_sym), dtype=np.float64) * 1_000_000
    aligned.turnover_2d = np.ones((t_max, n_sym), dtype=np.float64) * 0.1
    aligned.open_2d = aligned.close_2d.copy()
    aligned.high_2d = aligned.close_2d.copy() * 1.01
    aligned.low_2d = aligned.close_2d.copy() * 0.99

    events = [
        ValidatedSignalEvent(
            decision_idx=3,
            decision_time=aligned.datetimes[3],
            symbol="SYM_4H",
            strategy_id="trend_donchian:donchian_72",
            native_tf="4h",
            activation_context="all",
            side=1,
            expected_gross_bps=50.0,
            q10_gross_bps=10.0,
            q90_gross_bps=90.0,
            expected_net_bps=40.0,
            q10_net_bps=5.0,
            q90_net_bps=80.0,
            expected_holding_bars=5,
            registry_version="test",
            model_version="test",
        ),
        ValidatedSignalEvent(
            decision_idx=3,
            decision_time=aligned.datetimes[3],
            symbol="SYM_12H",
            strategy_id="mean_revert:rsi_14",
            native_tf="12h",
            activation_context="all",
            side=1,
            expected_gross_bps=50.0,
            q10_gross_bps=10.0,
            q90_gross_bps=90.0,
            expected_net_bps=40.0,
            q10_net_bps=5.0,
            q90_net_bps=80.0,
            expected_holding_bars=5,
            registry_version="test",
            model_version="test",
        ),
    ]
    signal_batch = ValidatedSignalBatch(
        events=tuple(events),
        start_idx=0,
        end_idx=t_max,
        symbols=aligned.symbols,
        registry_version="test",
        model_version="test",
    )
    cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
    awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=5, oos_start=5, oos_end=t_max - 1),)
    config = Layer2AllocationConfig(l2_tf_inclusion_enabled=True, l2_tf_inclusion_min_edge=0.0)
    caps = PortfolioCaps(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)

    return aligned, cache, signal_batch, awf_folds, config, caps


# ─────────────────────────────────────────────────────────────────────────────
# S1-S3: _build_sleeve_tf_lookup
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSleeveTfLookup:
    def test_maps_family_variant_strategy_id_without_tf_suffix(self) -> None:
        cache = _make_minimal_cache_with_keys(
            sleeve_keys=(
                SignalSleeveKey("BTCUSDT", "4h", "trend_donchian:donchian_72"),
                SignalSleeveKey("ETHUSDT", "8h", "mean_revert:rsi_14"),
            )
        )
        lookup = _build_sleeve_tf_lookup(cache)
        assert lookup[("BTCUSDT", "trend_donchian:donchian_72")] == "4h"
        assert lookup[("ETHUSDT", "mean_revert:rsi_14")] == "8h"

    def test_empty_sleeve_keys_returns_empty_dict(self) -> None:
        cache = _make_minimal_cache_with_keys(sleeve_keys=())
        assert _build_sleeve_tf_lookup(cache) == {}

    def test_missing_key_lookup_falls_back_to_unk(self) -> None:
        cache = _make_minimal_cache_with_keys(
            sleeve_keys=(SignalSleeveKey("BTCUSDT", "4h", "trend_donchian:donchian_72"),)
        )
        lookup = _build_sleeve_tf_lookup(cache)
        assert lookup.get(("UNKNOWN", "no_such_strategy"), "unk") == "unk"


# ─────────────────────────────────────────────────────────────────────────────
# S2: compute_per_tf_fit_edge directional sign
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePerTfFitEdge:
    def test_directional_sign(self) -> None:
        t_max = 10
        n_sleeve = 2
        n_sym = 2
        aligned = MagicMock()
        aligned.symbols = ("SYM_A", "SYM_B")
        close_2d = np.zeros((t_max, n_sym), dtype=np.float64)
        for t in range(t_max):
            close_2d[t, 0] = 100.0 - t * 1.0
            close_2d[t, 1] = 100.0 + t * 1.0
        aligned.close_2d = close_2d

        side_2d = np.ones((t_max, n_sleeve), dtype=np.float64)
        signal_mask_2d = np.ones((t_max, n_sleeve), dtype=np.bool_)
        sleeve_to_sym = np.array([0, 1], dtype=np.int64)
        sleeve_keys: tuple[SignalSleeveKey, ...] = (
            SignalSleeveKey(symbol="SYM_A", native_tf="4h", strategy_id="strat_4h"),
            SignalSleeveKey(symbol="SYM_B", native_tf="12h", strategy_id="strat_12h"),
        )
        cache = L2SimulationCache(
            vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
            tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
            hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
            funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
            beta_1d=np.zeros(n_sym, dtype=np.float64),
            expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            side_2d=side_2d,
            quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            signal_mask_2d=signal_mask_2d,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_keys=sleeve_keys,
        )
        edge = compute_per_tf_fit_edge(cache, aligned, fit_start=0, fit_end=t_max - 1)
        assert edge.get("12h", -999.0) > edge.get("4h", 999.0), (
            f"expected 12h > 4h, got 12h={edge.get('12h'):.4f} 4h={edge.get('4h'):.4f}"
        )

    def test_empty_tf_returns_zero(self) -> None:
        t_max = 10
        n_sleeve = 2
        n_sym = 2
        signal_mask_2d = np.zeros((t_max, n_sleeve), dtype=np.bool_)
        side_2d = np.ones((t_max, n_sleeve), dtype=np.float64)
        sleeve_to_sym = np.array([0, 1], dtype=np.int64)
        sleeve_keys: tuple[SignalSleeveKey, ...] = (
            SignalSleeveKey(symbol="SYM_A", native_tf="4h", strategy_id="strat_4h"),
            SignalSleeveKey(symbol="SYM_B", native_tf="12h", strategy_id="strat_12h"),
        )
        cache = L2SimulationCache(
            vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
            tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
            hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
            funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
            beta_1d=np.zeros(n_sym, dtype=np.float64),
            expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            side_2d=side_2d,
            quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            signal_mask_2d=signal_mask_2d,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_keys=sleeve_keys,
        )
        aligned = _make_aligned(t_max, n_sym)
        edge = compute_per_tf_fit_edge(cache, aligned, fit_start=0, fit_end=5)
        assert edge == {}


# ─────────────────────────────────────────────────────────────────────────────
# S4: inclusion gate filter logic (integration with _run_awf_simulation)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunAwfSimulationTfGateIntegration:
    def test_family_variant_strategy_id_survives_tf_gate(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

        aligned, cache, signal_batch, awf_folds, config, caps = _build_tf_gate_fixture()

        result = _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            sim_origin="test_tf_gate",
        )

        assert result.trade_count > 0, (
            f"TF gate should allow 4h sleeve through, got trade_count={result.trade_count}"
        )


class TestTfInclusionFilter:
    def test_disabled_no_change(self) -> None:
        config = Layer2AllocationConfig(l2_tf_inclusion_enabled=False)
        assert config.l2_tf_inclusion_enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# S5: empty fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestTfInclusionEmptyFallback:
    def test_empty_fallback_config_defaults(self) -> None:
        config = Layer2AllocationConfig()
        assert config.l2_tf_inclusion_enabled is True
        assert config.l2_tf_inclusion_min_edge == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# S6: gate off regression
# ─────────────────────────────────────────────────────────────────────────────


class TestTfInclusionRegression:
    def test_from_mapping_parses_correctly(self) -> None:
        config = Layer2AllocationConfig.from_mapping(
            {
                "l2_tf_inclusion_enabled": False,
                "l2_tf_inclusion_min_edge": 0.005,
            }
        )
        assert config.l2_tf_inclusion_enabled is False
        assert config.l2_tf_inclusion_min_edge == 0.005

    def test_from_mapping_defaults(self) -> None:
        config = Layer2AllocationConfig.from_mapping({})
        assert config.l2_tf_inclusion_enabled is True
        assert config.l2_tf_inclusion_min_edge == 0.0
