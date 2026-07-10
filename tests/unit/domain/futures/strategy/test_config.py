from __future__ import annotations

import pytest

from src.domain.futures.strategy.config import (
    _DEFAULT_PER_TF_FAMILIES,
    DEPRIORITIZED_FAMILY_PRIOR,
    BtcNeutralResidualReversalConfig,
    CandidateStrategyConfig,
    LiquidityParticipationBreakoutConfig,
    resolve_purge_and_embargo_bars,
    with_max_holding_bars,
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


def test_default_per_tf_families_1h_2h_includes_trend_pullback_continuation() -> None:
    assert "trend_pullback_continuation" in _DEFAULT_PER_TF_FAMILIES["1h"]
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


def test_candidate_config_includes_new_families() -> None:
    cfg = CandidateStrategyConfig()
    assert "liquidity_participation_breakout" in cfg.candidate_families
    assert "btc_neutral_residual_reversal" in cfg.candidate_families


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
