from __future__ import annotations

import dataclasses

import pytest

from src.domain.futures.strategy.config import (
    _DEFAULT_PER_TF_FAMILIES,
    DEFAULT_L1_TFS,
    DEPRIORITIZED_FAMILY_PRIOR,
    BlendConfig,
    BtcNeutralResidualReversalConfig,
    CandidateStrategyConfig,
    LiquidityParticipationBreakoutConfig,
    RegimeConfig,
    apply_tf_gate_overrides,
    resolve_purge_and_embargo_bars,
    with_max_holding_bars,
)
from src.domain.futures.strategy.timeframe_contracts import scale_bar_count

_REMOVED_FAMILIES = (
    "xs_momentum", "xs_flow", "xs_oi_skew", "funding_flow_carry",
    "lsr_oi_regime_filter", "supertrend", "ichimoku_trend",
    "carry_net_of_funding", "liquidity_participation_breakout",
    "btc_neutral_residual_reversal", "price_band_reversion",
    "funding_flow_exhaustion_sparse", "oi_lsr_unwind", "vol_contraction_breakout",
)


def test_candidate_strategy_config_auto_derives_purge_and_embargo_bars() -> None:
    cfg = CandidateStrategyConfig(purge_bars=None, embargo_bars=None, purge_safety_mult=1.2)
    resolved_cfg = with_max_holding_bars(cfg, max_holding_bars=36)
    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg, max_holding_bars=36)

    assert cfg.purge_bars == 44
    assert cfg.embargo_bars == 44
    assert resolved_cfg.purge_bars == 44
    assert resolved_cfg.embargo_bars == 44
    assert purge_bars == 44
    assert embargo_bars == 44
    assert cfg.min_ic_tstat == 0.8


def test_with_max_holding_bars_rederives_from_raw_optional_inputs() -> None:
    cfg = CandidateStrategyConfig(purge_bars=None, embargo_bars=None, purge_safety_mult=1.2)

    resolved_cfg = with_max_holding_bars(cfg, max_holding_bars=50)

    assert resolved_cfg.purge_bars == 60
    assert resolved_cfg.embargo_bars == 60


def test_candidate_strategy_config_validates_ensemble_backend_fields() -> None:
    with pytest.raises(ValueError, match="allocation_backend"):
        CandidateStrategyConfig(allocation_backend="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ensemble_shrinkage_k"):
        CandidateStrategyConfig(ensemble_shrinkage_k=0.0)


def test_candidate_strategy_config_defaults_exact_l1_boundary() -> None:
    cfg = CandidateStrategyConfig()

    assert cfg.l1_boundary_mode == "exact_label_interval"
    assert cfg.l1_boundary_buffer_bars == 0


def test_candidate_strategy_config_rejects_negative_l1_boundary_buffer() -> None:
    with pytest.raises(ValueError, match="l1_boundary_buffer_bars"):
        CandidateStrategyConfig(l1_boundary_buffer_bars=-1)


def test_l1_ens_prior_effective_n_default_zero() -> None:
    cfg = CandidateStrategyConfig()
    assert cfg.l1_ens_prior_effective_n == 0.0


def test_l1_ens_min_display_events_default_zero() -> None:
    cfg = CandidateStrategyConfig()
    assert cfg.l1_ens_min_display_events == 0


def test_l1_evidence_early_snapshots_default_zero() -> None:
    cfg = CandidateStrategyConfig()
    assert cfg.l1_evidence_early_snapshots == 0


def test_l1_evidence_early_snapshots_rejects_negative() -> None:
    with pytest.raises(ValueError, match="l1_evidence_early_snapshots"):
        CandidateStrategyConfig(l1_evidence_early_snapshots=-1)


def test_l1_ens_prior_effective_n_rejects_negative() -> None:
    with pytest.raises(ValueError, match="l1_ens_prior_effective_n"):
        CandidateStrategyConfig(l1_ens_prior_effective_n=-1.0)


def test_l1_pair_min_effective_obs_early_default() -> None:
    cfg = CandidateStrategyConfig()
    assert cfg.l1_pair_min_effective_obs_early == 2.0


def test_l1_pair_min_folds_early_default() -> None:
    cfg = CandidateStrategyConfig()
    assert cfg.l1_pair_min_folds_early == 1


def test_l1_pair_min_effective_obs_early_rejects_low() -> None:
    with pytest.raises(ValueError, match="l1_pair_min_effective_obs_early"):
        CandidateStrategyConfig(l1_pair_min_effective_obs_early=0.5)


# ─── l0_signal_yield_improvement ─────────────────────────────────────────────


def test_default_per_tf_families_2h_includes_trend_pullback_continuation() -> None:
    assert "trend_pullback_continuation" in _DEFAULT_PER_TF_FAMILIES["2h"]


def test_family_prior_score_deprioritized_families_all_negative() -> None:
    assert all(v < 0 for v in DEPRIORITIZED_FAMILY_PRIOR.values())


# ─── LiquidityParticipationBreakoutConfig ──────────────────────────────


def test_lpb_config_defaults() -> None:
    cfg = LiquidityParticipationBreakoutConfig()
    assert cfg.channel_bars == (40, 60)
    assert cfg.min_breakout_impulse_atr == 0.25
    assert cfg.score_impulse_atr == 1.00
    assert cfg.min_volume_zscore == 0.50
    # max_event_cost_bps / min_adv_usdt removed [LIMIT-05]
    assert not hasattr(cfg, "max_event_cost_bps")
    assert not hasattr(cfg, "min_adv_usdt")


def test_lpb_config_s3_01_invalid_channel_bars_single_element() -> None:
    with pytest.raises(ValueError, match="channel_bars"):
        LiquidityParticipationBreakoutConfig(channel_bars=(1,))


def test_lpb_config_s3_01_invalid_zero_score_scale() -> None:
    with pytest.raises(ValueError, match="score_impulse_atr"):
        LiquidityParticipationBreakoutConfig(score_impulse_atr=0.0)


def test_lpb_config_empty_channel_bars() -> None:
    with pytest.raises(ValueError, match="channel_bars"):
        LiquidityParticipationBreakoutConfig(channel_bars=())


# ─── BtcNeutralResidualReversalConfig ──────────────────────────────────


def test_bnrr_config_defaults() -> None:
    cfg = BtcNeutralResidualReversalConfig()
    assert cfg.lookback_bars == (24, 48)
    assert cfg.tail_fraction == 0.20
    # max_event_cost_bps / min_adv_usdt removed [LIMIT-05]
    assert not hasattr(cfg, "max_event_cost_bps")
    assert not hasattr(cfg, "min_adv_usdt")
    assert cfg.min_cross_section == 30
    assert cfg.max_abs_btc_beta == 0.80


def test_bnrr_config_s3_02_invalid_tail_fraction_ge_05() -> None:
    with pytest.raises(ValueError, match="tail_fraction"):
        BtcNeutralResidualReversalConfig(tail_fraction=0.5)


def test_bnrr_config_s3_02_invalid_tail_fraction_above_05() -> None:
    with pytest.raises(ValueError, match="tail_fraction"):
        BtcNeutralResidualReversalConfig(tail_fraction=0.7)


def test_bnrr_config_s3_02_invalid_min_cross_section_below_2() -> None:
    with pytest.raises(ValueError, match="min_cross_section"):
        BtcNeutralResidualReversalConfig(min_cross_section=1)


def test_bnrr_config_empty_lookback_bars() -> None:
    with pytest.raises(ValueError, match="lookback_bars"):
        BtcNeutralResidualReversalConfig(lookback_bars=())


# ─── CandidateStrategyConfig new fields ────────────────────────────────


def test_candidate_config_includes_core_families() -> None:
    cfg = CandidateStrategyConfig()
    assert "trend_ma" in cfg.candidate_families
    assert "mtf_fusion" in cfg.candidate_families


def test_candidate_config_lpb_defaults_injected() -> None:
    cfg = CandidateStrategyConfig()
    lpb = cfg.liquidity_participation_breakout
    assert lpb.channel_bars == (40, 60)
    assert lpb.min_breakout_impulse_atr == 0.25


def test_candidate_config_bnrr_defaults_injected() -> None:
    cfg = CandidateStrategyConfig()
    bnrr = cfg.btc_neutral_residual_reversal
    assert bnrr.lookback_bars == (24, 48)
    assert bnrr.tail_fraction == 0.20


# ─── Fix A: TF Portfolio Restructure ──────────────────────────────────


def test_default_l1_tfs_includes_1h() -> None:
    assert "1h" in DEFAULT_L1_TFS
    assert "1d" in DEFAULT_L1_TFS
    assert DEFAULT_L1_TFS.index("1h") < DEFAULT_L1_TFS.index("2h")


def test_1d_family_pool_excludes_mtf_fusion() -> None:
    assert "1d" in _DEFAULT_PER_TF_FAMILIES
    assert "mtf_fusion" not in _DEFAULT_PER_TF_FAMILIES["1d"]
    assert "trend_ma" in _DEFAULT_PER_TF_FAMILIES["1d"]


# ─── Fix B: Durable-Zero Family Pruning ───────────────────────────────


def test_durable_zero_families_removed_from_candidate_families() -> None:
    cfg = CandidateStrategyConfig()
    for fam in _REMOVED_FAMILIES:
        assert fam not in cfg.candidate_families


def test_residual_reversion_survives_pruning() -> None:
    cfg = CandidateStrategyConfig()
    assert "residual_reversion" in cfg.candidate_families


@pytest.mark.parametrize("tf", ["1h", "2h", "4h", "6h", "8h", "12h", "1d"])
def test_no_removed_family_in_any_tf_pool(tf: str) -> None:
    pool = _DEFAULT_PER_TF_FAMILIES[tf]
    for fam in _REMOVED_FAMILIES:
        assert fam not in pool, f"{fam} still present in {tf}"


def test_deprioritized_family_prior_no_longer_lists_removed_families() -> None:
    assert "supertrend" not in DEPRIORITIZED_FAMILY_PRIOR
    assert "funding_flow_carry" not in DEPRIORITIZED_FAMILY_PRIOR


# ── Fix A: TF-relative field metadata governance ────────────────────────────

_TF_SCALE_NAME_PATTERNS = ("_bars", "_window", "_span", "_hours", "_days", "_per_fold", "_events_per_fold")
_TF_SCALE_EXEMPT_FIELDS = frozenset({
    "wf_n_folds", "train_months", "valid_months", "test_months",
    "l1_evidence_max_folds", "l1_outer_warmup_blocks", "min_wf_fold_pass_ratio",
})


@pytest.mark.parametrize(
    "dc_cls",
    [
        CandidateStrategyConfig,
        RegimeConfig,
        BlendConfig,
        LiquidityParticipationBreakoutConfig,
        BtcNeutralResidualReversalConfig,
    ],
)
def test_all_tf_relative_fields_have_scale_metadata(dc_cls: type) -> None:
    unclassified = [
        f.name for f in dataclasses.fields(dc_cls)
        if any(p in f.name for p in _TF_SCALE_NAME_PATTERNS)
        and f.name not in _TF_SCALE_EXEMPT_FIELDS
        and "tf_scale_base" not in f.metadata
    ]
    assert unclassified == [], f"미분류 TF-상대 필드 발견: {unclassified}"


# ── Fix B: apply_tf_gate_overrides scaling ──────────────────────────────────


def test_apply_tf_gate_overrides_scales_max_holding_bars_for_1d() -> None:
    cfg = CandidateStrategyConfig(max_holding_bars=36)
    resolved = apply_tf_gate_overrides(cfg, "1d")
    assert resolved.max_holding_bars == scale_bar_count(36, "1d", base_tf="4h")
    assert resolved.max_holding_bars < 36


def test_apply_tf_gate_overrides_purge_bars_rederived_after_scaling() -> None:
    cfg = CandidateStrategyConfig(max_holding_bars=36, purge_bars=None, embargo_bars=None)
    resolved = apply_tf_gate_overrides(cfg, "1d")
    expected_max_holding = scale_bar_count(36, "1d", base_tf="4h")
    assert resolved.max_holding_bars == expected_max_holding
    assert resolved.purge_bars is not None
    assert resolved.purge_bars >= expected_max_holding


def test_apply_tf_gate_overrides_ignores_none_metadata_fields() -> None:
    cfg = CandidateStrategyConfig(l1_bootstrap_block_bars=6)
    resolved = apply_tf_gate_overrides(cfg, "1d")
    assert resolved.l1_bootstrap_block_bars == 6


def test_apply_tf_gate_overrides_roundtrip_same_tf() -> None:
    cfg = CandidateStrategyConfig(max_holding_bars=36, label_horizon_bars=12)
    resolved = apply_tf_gate_overrides(cfg, "4h")
    assert resolved.max_holding_bars == 36
    assert resolved.label_horizon_bars == 12
