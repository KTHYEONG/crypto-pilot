from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.dataclasses import RegimeCellPolicy
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    _build_bucket_reliability,
    _parse_meta_group_ids,
    apply_bucket_conditional_weight,
    apply_regime_cell_policy,
    filter_sleeves_by_bucket,
)


def _sig(raw_mu: float, quality_weight: float = 1.0) -> SymbolSignal:
    return SymbolSignal(
        raw_mu=raw_mu,
        volatility=1e-3,
        n_obs=100,
        t_stat=2.0,
        valid=True,
        beta_btc=None,
        quality_weight=quality_weight,
    )


def test_regime_bucket_reliability_allows_consistent_fit_cal() -> None:
    reliability = _build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=12.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is True
    assert reliability.action == "allow"
    assert reliability.reliability >= 0.55


def test_regime_bucket_reliability_pools_sign_flip() -> None:
    reliability = _build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=-4.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is False
    assert reliability.action == "pool"


# ── L3: Adaptive Regime-Reliability ──────────────────────────────────────

from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    bear_edge_per_bar_bps,
    compute_regime_reliability_multiplier,
)


class TestRegimeReliabilityMultiplier:
    def test_reliability_mult_negative_edge_downweights(self) -> None:
        result = compute_regime_reliability_multiplier(
            [-30.0, -25.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert result == pytest.approx(0.2)

    def test_reliability_mult_positive_edge_keeps_full(self) -> None:
        result = compute_regime_reliability_multiplier(
            [150.0, 140.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert result == pytest.approx(1.0)

    def test_reliability_mult_linear_ramp_midpoint(self) -> None:
        result = compute_regime_reliability_multiplier(
            [-5.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        expected = 0.2 + 0.8 * (-5.0 - (-10.0)) / (0.0 - (-10.0))
        assert result == pytest.approx(expected)

    def test_reliability_mult_empty_returns_neutral(self) -> None:
        result = compute_regime_reliability_multiplier([])
        assert result == 1.0

    def test_reliability_mult_clamped_to_floor_bounds(self) -> None:
        low = compute_regime_reliability_multiplier(
            [-1000.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert low == pytest.approx(0.2)

        high = compute_regime_reliability_multiplier(
            [1000.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert high == pytest.approx(1.0)

    def test_reliability_mult_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            compute_regime_reliability_multiplier([], floor=0.0)
        with pytest.raises(ValueError, match="floor"):
            compute_regime_reliability_multiplier([], floor=1.5)
        with pytest.raises(ValueError, match="pos_edge_at_full_bps"):
            compute_regime_reliability_multiplier(
                [],
                floor=0.2,
                pos_edge_at_full_bps=-10.0,
                neg_edge_at_floor_bps=0.0,
            )


class TestBearEdgePerBarBps:
    def test_bear_edge_per_bar_bps(self) -> None:
        assert bear_edge_per_bar_bps(0.015, 100) == pytest.approx(1.5)
        assert bear_edge_per_bar_bps(-0.30, 200) == pytest.approx(-15.0)
        assert bear_edge_per_bar_bps(5.0, 0) == 0.0


# ── _parse_meta_group_ids ────────────────────────────────────────────


class TestParseMetaGroupIds:
    """All error paths are covered by defensive fallback (no pytest.raises needed)."""

    # Scenario 1: Happy Path
    def test_parse_meta_group_ids_splits_family_and_tf_from_canonical_format(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma:ema_12_72_4h")
        assert result == ("trend_ma", "4h")

    # Scenario 1b: Family literal containing hour pattern
    def test_parse_meta_group_ids_preserves_family_literal_containing_hour_pattern(
        self,
    ) -> None:
        result = _parse_meta_group_ids("macd_4h:base_variant_1h")
        assert result == ("macd_4h", "1h")

    # Scenario 2a: Variant without tf suffix
    def test_parse_meta_group_ids_defaults_tf_unknown_when_variant_has_no_suffix(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma:ema_12_72")
        assert result == ("trend_ma", "unknown")

    # Scenario 2b: Legacy no-colon format (regression guard)
    def test_parse_meta_group_ids_preserves_legacy_no_colon_format_correctly(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma_4h")
        assert result == ("trend_ma", "4h")

    # Scenario 2c: Empty string
    def test_parse_meta_group_ids_returns_unknown_pair_for_empty_string(
        self,
    ) -> None:
        result = _parse_meta_group_ids("")
        assert result == ("unknown", "unknown")


# ─── apply_bucket_conditional_weight ─────────────────────────────────


def test_apply_bucket_conditional_weight_clip_lower_bound() -> None:
    """1.2 edge가 낮으면 g가 g_min으로 clip된다."""
    sleeve_sigs = {
        ("BTCUSDT", "dual_momentum:trend_4h"): _sig(3.678, quality_weight=1.0),
        ("BTCUSDT", "ichimoku_trend:signal_4h"): _sig(-0.222, quality_weight=0.8),
    }
    bucket_edges = {
        (2, "dual_momentum", "4h"): 10.0,
        (2, "ichimoku_trend", "4h"): 65.0,
    }
    result = apply_bucket_conditional_weight(
        sleeve_sigs,
        bucket_edges,
        regime_now=2,
        edge_floor_bps=0.0,
        edge_ref_bps=50.0,
        g_min=0.5,
        g_max=1.5,
    )
    assert result[("BTCUSDT", "dual_momentum:trend_4h")].quality_weight == pytest.approx(1.0 * 0.5)
    assert result[("BTCUSDT", "ichimoku_trend:signal_4h")].quality_weight == pytest.approx(0.8 * 1.3)


def test_apply_bucket_conditional_weight_empty_input() -> None:
    """3.2 빈 dict: {} 반환."""
    assert apply_bucket_conditional_weight({}, {}, regime_now=0) == {}


def test_apply_bucket_conditional_weight_missing_bucket_key() -> None:
    """2.4 bucket_edges에 키 없음 → edge=0 → edge_floor=0이면 제외."""
    sleeve_sigs = {
        ("BTCUSDT", "dual_momentum:trend_4h"): _sig(3.678, quality_weight=1.0),
    }
    bucket_edges: dict[tuple[int, str, str], float] = {}
    result = apply_bucket_conditional_weight(
        sleeve_sigs,
        bucket_edges,
        regime_now=2,
        edge_floor_bps=0.0,
    )
    assert len(result) == 0


def test_apply_bucket_conditional_weight_missing_key_with_neg_floor() -> None:
    """2.4 bucket_edges에 키 없지만 edge_floor_bps < 0이면 통과(g=1.0)."""
    sleeve_sigs = {
        ("BTCUSDT", "dual_momentum:trend_4h"): _sig(3.678, quality_weight=1.0),
    }
    bucket_edges: dict[tuple[int, str, str], float] = {}
    result = apply_bucket_conditional_weight(
        sleeve_sigs,
        bucket_edges,
        regime_now=2,
        edge_floor_bps=-10.0,
        edge_ref_bps=50.0,
        g_min=0.5,
        g_max=1.5,
    )
    assert len(result) == 1
    g = max(0.5, min(1.5, (0.0 - (-10.0)) / 50.0))
    assert result[("BTCUSDT", "dual_momentum:trend_4h")].quality_weight == pytest.approx(1.0 * g)


def test_apply_bucket_conditional_weight_reduces_pooled_magnitude() -> None:
    """1.3 재가중 후 pooled_mu 크기가 감소(부호는 유지)."""
    sleeve_sigs = {
        ("BTCUSDT", "dual_momentum:trend_4h"): _sig(3.678, quality_weight=1.0),
        ("BTCUSDT", "ichimoku_trend:signal_4h"): _sig(-0.222, quality_weight=0.8),
    }
    bucket_edges = {
        (2, "dual_momentum", "4h"): 10.0,
        (2, "ichimoku_trend", "4h"): 65.0,
    }
    reweighted = apply_bucket_conditional_weight(
        sleeve_sigs,
        bucket_edges,
        regime_now=2,
        edge_floor_bps=0.0,
        edge_ref_bps=50.0,
        g_min=0.5,
        g_max=1.5,
    )
    cs = np.array([max(s.quality_weight, 0.0) for s in sleeve_sigs.values()], dtype=np.float64)
    mus = np.array([s.raw_mu for s in sleeve_sigs.values()], dtype=np.float64)
    pooled_mu_before = float((cs * mus).sum() / cs.sum())

    cs_r = np.array([max(s.quality_weight, 0.0) for s in reweighted.values()], dtype=np.float64)
    mus_r = np.array([s.raw_mu for s in reweighted.values()], dtype=np.float64)
    pooled_mu_after = float((cs_r * mus_r).sum() / cs_r.sum())

    assert pooled_mu_before > pooled_mu_after > 0


def test_apply_bucket_conditional_weight_preserves_conviction_cap() -> None:
    """2.6 재가중 후에도 conviction_cap(c_s = min(sum(c_i'), κ*max(c_i')))이 적용됨."""
    sleeve_sigs = {
        ("BTCUSDT", "dual_momentum:trend_4h"): _sig(3.0, quality_weight=1.0),
        ("BTCUSDT", "supertrend:signal_4h"): _sig(1.0, quality_weight=2.0),
    }
    bucket_edges = {
        (2, "dual_momentum", "4h"): 50.0,
        (2, "supertrend", "4h"): 50.0,
    }
    reweighted = apply_bucket_conditional_weight(
        sleeve_sigs,
        bucket_edges,
        regime_now=2,
        edge_floor_bps=0.0,
        edge_ref_bps=50.0,
        g_min=0.5,
        g_max=1.5,
    )
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _combine_sleeve_signals_to_symbol

    combined, _ = _combine_sleeve_signals_to_symbol(
        dict(reweighted),
        method="precision_weighted",
        conviction_cap_mult=1.5,
    )
    btc_sig = combined["BTCUSDT"]
    cs = np.array([max(s.quality_weight, 0.0) for s in reweighted.values()], dtype=np.float64)
    cap = min(float(cs.sum()), 1.5 * float(cs.max()))
    assert btc_sig.quality_weight == pytest.approx(cap)


def _simulate_guard(config: object) -> None:
    """_run_awf_simulation 진입점 guard 로직을 재현."""
    _l2_regime_weight = bool(getattr(config, "l2_regime_conditional_weight_enabled", False))
    _l2_intra_divergence = bool(getattr(config, "l2_intra_symbol_divergence_enabled", False))
    assert not (_l2_regime_weight and _l2_intra_divergence), (
        "l2_regime_conditional_weight_enabled and l2_intra_symbol_divergence_enabled are mutually exclusive"
    )


def test_mutual_exclusion_guard_raises_on_dual_enable() -> None:
    """2.3 l2_regime_conditional_weight_enabled + l2_intra_symbol_divergence_enabled 동시 True → AssertionError."""
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

    cfg = Layer2AllocationConfig(
        l2_regime_conditional_weight_enabled=True,
        l2_intra_symbol_divergence_enabled=True,
    )
    with pytest.raises(AssertionError, match="mutually exclusive"):
        _simulate_guard(cfg)


def test_mutual_exclusion_guard_passes_single_enable() -> None:
    """단일 flag만 True면 guard 통과."""
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

    cfg = Layer2AllocationConfig(
        l2_regime_conditional_weight_enabled=True,
        l2_intra_symbol_divergence_enabled=False,
    )
    _simulate_guard(cfg)  # no raise


class TestFilterSleevesByBucket:
    """Phase B: regime x side bucket key split."""

    def _make_sleeve_sigs(self) -> dict[tuple[str, str], SymbolSignal]:
        return {
            ("BTCUSDT", "dual_momentum:ema_12_72_4h"): _sig(raw_mu=-10.0),
            ("ETHUSDT", "dual_momentum:ema_12_72_4h"): _sig(raw_mu=8.0),
            ("BTCUSDT", "trend_donchian:dc_20_4h"): _sig(raw_mu=-5.0),
        }

    def _make_bucket_edges_with_side(self) -> dict[tuple[int, str, str, int], float]:
        return {
            (2, "dual_momentum", "4h", -1): 20.0,
            (2, "dual_momentum", "4h", 1): -15.0,
            (2, "trend_donchian", "4h", -1): 10.0,
        }

    def _make_bucket_edges_legacy(self) -> dict[tuple[int, str, str], float]:
        return {
            (2, "dual_momentum", "4h"): 12.0,
            (2, "trend_donchian", "4h"): 10.0,
        }

    def _make_sleeve_sigs_diff_families(self) -> dict[tuple[str, str], SymbolSignal]:
        return {
            ("BTCUSDT", "dual_momentum:ema_12_72_4h"): _sig(raw_mu=-10.0),
            ("BTCUSDT", "residual_reversion:rr_20_4h"): _sig(raw_mu=8.0),
        }

    def test_regime_bucket_side_split_separates_short_and_long_edges(self) -> None:
        sleeve_sigs = self._make_sleeve_sigs_diff_families()
        bucket_edges: dict[tuple[int, str, str, int], float] = {
            (2, "dual_momentum", "4h", -1): 20.0,
            (2, "dual_momentum", "4h", 1): -15.0,
            (2, "residual_reversion", "4h", -1): -5.0,
            (2, "residual_reversion", "4h", 1): 12.0,
        }

        short_result = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=2, side=-1, edge_floor_bps=5.0)
        long_result = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=2, side=1, edge_floor_bps=5.0)

        assert ("BTCUSDT", "dual_momentum:ema_12_72_4h") in short_result
        assert ("BTCUSDT", "residual_reversion:rr_20_4h") not in short_result
        assert len(short_result) == 1

        assert ("BTCUSDT", "dual_momentum:ema_12_72_4h") not in long_result
        assert ("BTCUSDT", "residual_reversion:rr_20_4h") in long_result
        assert len(long_result) == 1

    def test_regime_bucket_side_split_low_sample_falls_back_to_pooled(self) -> None:
        sleeve_sigs = self._make_sleeve_sigs()
        bucket_edges: dict[tuple[int, str, str, int], float] = {}

        result = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=2, side=-1, edge_floor_bps=5.0)
        assert len(result) == 0

    def test_regime_bucket_side_split_disabled_matches_legacy_output(self) -> None:
        sleeve_sigs = self._make_sleeve_sigs()
        bucket_edges = self._make_bucket_edges_legacy()

        result_without_side = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=2, side=0, edge_floor_bps=5.0)
        result_default = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=2, edge_floor_bps=5.0)

        assert result_without_side == result_default
        assert len(result_without_side) == 3
        for key in result_without_side:
            assert key in sleeve_sigs


class TestApplyBucketConditionalWeightSide:
    """Phase B: side-split coverage for apply_bucket_conditional_weight."""

    def test_apply_bucket_conditional_weight_side_key_construction(self) -> None:
        sleeve_sigs: dict[tuple[str, str], SymbolSignal] = {
            ("BTCUSDT", "dual_momentum:ema_12_72_4h"): _sig(raw_mu=-10.0),
        }
        bucket_edges: dict[tuple[int, str, str, int], float] = {
            (2, "dual_momentum", "4h", -1): 50.0,
        }

        result = apply_bucket_conditional_weight(
            sleeve_sigs, bucket_edges, regime_now=2, side=-1,
            edge_floor_bps=5.0, edge_ref_bps=50.0,
        )
        assert ("BTCUSDT", "dual_momentum:ema_12_72_4h") in result
        assert result[("BTCUSDT", "dual_momentum:ema_12_72_4h")].quality_weight == pytest.approx(0.9)

    def test_apply_bucket_conditional_weight_side_zero_default(self) -> None:
        sleeve_sigs: dict[tuple[str, str], SymbolSignal] = {
            ("BTCUSDT", "dual_momentum:ema_12_72_4h"): _sig(raw_mu=-10.0),
        }
        bucket_edges: dict[tuple[int, str, str], float] = {
            (2, "dual_momentum", "4h"): 50.0,
        }

        result = apply_bucket_conditional_weight(
            sleeve_sigs, bucket_edges, regime_now=2, side=0,
            edge_floor_bps=5.0, edge_ref_bps=50.0,
        )
        assert ("BTCUSDT", "dual_momentum:ema_12_72_4h") in result


# ── [SPEC alpha-funnel-regime-coverage Phase B] apply_regime_cell_policy side_split ──


def _policy(
    *, action: str, side: int, edge_multiplier: float = 1.0, hard_block_eligible: bool = False,
) -> RegimeCellPolicy:
    return RegimeCellPolicy(
        state=2,
        state_name="bear",
        family="trend",
        tf="4h",
        side=side,
        action=action,  # type: ignore[arg-type]
        reason="positive_cal_lift",  # type: ignore[arg-type]
        edge_multiplier=edge_multiplier,
        confidence=1.0,
        fit_edge_bps=10.0,
        pooled_fit_edge_bps=0.0,
        cal_edge_bps=10.0,
        pooled_cal_edge_bps=0.0,
        fit_lift_bps=10.0,
        cal_lift_bps=10.0,
        sign_consistent=True,
        hard_block_eligible=hard_block_eligible,
        n_fit=100,
        n_cal=100,
    )


class TestApplyRegimeCellPolicySideSplit:
    def test_side_split_enabled_routes_long_and_short_to_different_policies(self) -> None:
        sleeve_sigs = {
            ("BTCUSDT", "trend:ema_4h"): _sig(raw_mu=5.0),
            ("ETHUSDT", "trend:ema_4h"): _sig(raw_mu=5.0),
        }
        sleeve_edges = {
            ("BTCUSDT", "trend:ema_4h"): 30.0,   # long
            ("ETHUSDT", "trend:ema_4h"): -30.0,  # short
        }
        policy_map = {
            (2, "trend", "4h", 1): _policy(action="block", side=1, edge_multiplier=0.0, hard_block_eligible=True),
            (2, "trend", "4h", -1): _policy(action="allow", side=-1, edge_multiplier=1.0),
        }

        result = apply_regime_cell_policy(
            sleeve_sigs, sleeve_edges, policy_map, regime_now=2,
            mode="hybrid", side_split_enabled=True,
        )

        assert result.n_block == 1
        assert result.n_allow == 1
        assert ("BTCUSDT", "trend:ema_4h") not in result.sleeve_sigs
        assert ("ETHUSDT", "trend:ema_4h") in result.sleeve_sigs

    def test_side_split_disabled_ignores_side_and_matches_legacy_3key(self) -> None:
        sleeve_sigs = {("BTCUSDT", "trend:ema_4h"): _sig(raw_mu=5.0)}
        sleeve_edges = {("BTCUSDT", "trend:ema_4h"): 30.0}
        policy_map = {(2, "trend", "4h"): _policy(action="allow", side=0, edge_multiplier=1.0)}

        result = apply_regime_cell_policy(
            sleeve_sigs, sleeve_edges, policy_map, regime_now=2,
            mode="hybrid", side_split_enabled=False,
        )

        assert result.n_allow == 1
        assert ("BTCUSDT", "trend:ema_4h") in result.sleeve_sigs

    def test_side_split_shape_mismatch_causes_full_pooled_bypass_regression_guard(self) -> None:
        """[ADR_20260721_L2_REGIME_POLICY_SIDE_SPLIT_SHAPE_FIX] Documents the exact failure
        mode found in production: calling apply_regime_cell_policy with side_split_enabled=True
        against a policy_map built with legacy 3-key entries makes every lookup miss, silently
        bypassing all regime policy (100% pooled) instead of raising. This is why the flag
        used to consume must always come from the routing plan that built policy_map, never be
        re-derived independently (see awf_sim.py `_side_split_enabled` single-source fix)."""
        sleeve_sigs = {("BTCUSDT", "trend:ema_4h"): _sig(raw_mu=5.0)}
        sleeve_edges = {("BTCUSDT", "trend:ema_4h"): 30.0}
        legacy_3key_policy_map = {(2, "trend", "4h"): _policy(action="block", side=0, hard_block_eligible=True)}

        result = apply_regime_cell_policy(
            sleeve_sigs, sleeve_edges, legacy_3key_policy_map, regime_now=2,
            mode="hybrid", side_split_enabled=True,
        )

        assert result.n_pooled == 1
        assert result.n_block == 0
        assert ("BTCUSDT", "trend:ema_4h") in result.sleeve_sigs
