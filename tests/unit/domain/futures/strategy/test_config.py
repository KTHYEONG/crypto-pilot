from __future__ import annotations

import pytest

from src.domain.futures.strategy.config import (
    CandidateStrategyConfig,
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
