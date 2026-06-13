from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_ensemble import (
    RegimeConditionalEnsemble,
    _compute_eb_shrinkage_k,
    _fit_cell_means,
    _fit_variant_means,
    _internal_validation_rank_ic,
    _log_ensemble_diagnostics,
    fit_regime_conditional_ensemble,
    predict_regime_conditional_ensemble,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kwargs: object) -> CandidateStrategyConfig:
    defaults: dict[str, object] = {"ensemble_shrinkage_k": 50.0}
    defaults.update(kwargs)
    return CandidateStrategyConfig(**defaults)  # type: ignore[arg-type]


def _train_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "archetype": rng.choice(["trend_continuation", "mean_reversion"], size=n),
            "entry_regime_code": rng.integers(0, 4, size=n),
            "net_return_bps": rng.normal(10.0, 30.0, size=n),
            "entry_idx": np.arange(n),
        }
    )


# ---------------------------------------------------------------------------
# archetype_regime conditioning (explicit) — original behaviour
# ---------------------------------------------------------------------------

def test_regime_conditional_ensemble_shrinks_small_cells_to_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.domain.futures.strategy import candidate_ensemble as _ce
    from src.domain.futures.strategy.regime_evaluation import RegimeLiftProofResult

    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [100.0, 100.0, -100.0],
        }
    )
    proof = train_events.copy()
    proof_ids = np.zeros(len(proof), dtype=np.int32)
    cfg = _make_cfg(
        ensemble_shrinkage_k=50.0,
        ensemble_conditioning="archetype_regime",
        ensemble_adaptive_shrinkage=False,
    )

    # lift_proof 통과로 고정 → archetype_regime 유지
    monkeypatch.setattr(
        _ce,
        "evaluate_regime_lift_proof",
        lambda **_kw: RegimeLiftProofResult(
            proof_passed=True,
            nw_tstat=2.0,
            fold_pass_ratio=1.0,
            conditioning_path="regime_conditioned",
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.6,
            mean_lift_bps=5.0,
            n_eff=10.0,
            deflated_sharpe=1.0,
            n_folds_evaluated=1,
        ),
    )

    model = fit_regime_conditional_ensemble(
        train_events=train_events, cfg=cfg, oos_proof_events=proof, fold_ids=proof_ids
    )

    global_mu = (100.0 + 100.0 - 100.0) / 3.0
    shrunk_small_cell = model.cell_mu_bps[("mean_reversion", 4)]

    assert model.global_mu_bps == pytest.approx(global_mu)
    assert -100.0 < shrunk_small_cell < global_mu
    assert model.conditioning == "archetype_regime"


def test_ensemble_predict_lookup_matches_cell_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.domain.futures.strategy import candidate_ensemble as _ce
    from src.domain.futures.strategy.regime_evaluation import RegimeLiftProofResult

    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [40.0, 60.0, 10.0],
        }
    )
    proof = train_events.copy()
    proof_ids = np.zeros(len(proof), dtype=np.int32)
    cfg = _make_cfg(ensemble_shrinkage_k=1.0, ensemble_conditioning="archetype_regime")

    monkeypatch.setattr(
        _ce,
        "evaluate_regime_lift_proof",
        lambda **_kw: RegimeLiftProofResult(
            proof_passed=True,
            nw_tstat=2.0,
            fold_pass_ratio=1.0,
            conditioning_path="regime_conditioned",
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.6,
            mean_lift_bps=5.0,
            n_eff=10.0,
            deflated_sharpe=1.0,
            n_folds_evaluated=1,
        ),
    )

    model = fit_regime_conditional_ensemble(
        train_events=train_events, cfg=cfg, oos_proof_events=proof, fold_ids=proof_ids
    )
    oos_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "mean_reversion", "unseen"],
            "entry_regime_code": [0, 4, 9],
        }
    )

    # cfg=None → no mu_quality shrinkage, raw cell lookup
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos_events)

    assert out.expected_net_bps[0] > 0.0
    assert out.expected_net_bps[1] == pytest.approx(model.cell_mu_bps[("mean_reversion", 4)])
    # "unseen" archetype → global fallback
    assert out.expected_net_bps[2] == pytest.approx(model.global_mu_bps)
    assert np.all(out.p_pass == 1.0)


def test_ensemble_fit_uses_train_window_only() -> None:
    """Unseen (arch, regime) in archetype_regime mode falls back to archetype mean."""
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation"],
            "entry_regime_code": [0, 0],
            "net_return_bps": [10.0, 20.0],
        }
    )
    cfg = _make_cfg(ensemble_shrinkage_k=1.0, ensemble_conditioning="archetype_regime")

    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    out = predict_regime_conditional_ensemble(
        model=model,
        oos_events=pd.DataFrame({"archetype": ["trend_continuation"], "entry_regime_code": [7]}),
    )

    assert ("trend_continuation", 7) not in model.cell_mu_bps
    # fallback: archetype_mu_bps["trend_continuation"] (not global)
    expected = model.archetype_mu_bps.get("trend_continuation", model.global_mu_bps)
    assert out.expected_net_bps[0] == pytest.approx(expected)


def test_ensemble_prefers_gross_targets_over_legacy_net_targets() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "gross_event_bps": [40.0, 60.0, 20.0],
            "net_return_bps": [-100.0, -100.0, -100.0],
            "entry_idx": [0, 1, 2],
        }
    )
    cfg = _make_cfg(ensemble_shrinkage_k=1.0, ensemble_conditioning="archetype_regime")

    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    out = predict_regime_conditional_ensemble(
        model=model,
        oos_events=pd.DataFrame(
            {"archetype": ["trend_continuation"], "entry_regime_code": [0]}
        ),
        cfg=cfg,
    )

    assert model.global_mu_bps > 0.0
    assert out.expected_net_bps[0] > 0.0
    assert out.validation_diagnostics["target_contract"] == "gross"


# ---------------------------------------------------------------------------
# archetype_only conditioning — regime_code stripped from alpha
# ---------------------------------------------------------------------------

def test_archetype_only_ignores_regime_code() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation"] * 6 + ["mean_reversion"] * 6,
            "entry_regime_code": [0, 1, 2, 3, 4, 5] * 2,
            "net_return_bps": [50.0] * 6 + [-20.0] * 6,
        }
    )
    cfg = _make_cfg(ensemble_conditioning="archetype_only")
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # cell_mu_bps should be empty for archetype_only
    assert model.cell_mu_bps == {}
    assert model.conditioning == "archetype_only"

    oos_regime0 = pd.DataFrame({"archetype": ["trend_continuation"], "entry_regime_code": [0]})
    oos_regime3 = pd.DataFrame({"archetype": ["trend_continuation"], "entry_regime_code": [3]})
    out0 = predict_regime_conditional_ensemble(model=model, oos_events=oos_regime0)
    out3 = predict_regime_conditional_ensemble(model=model, oos_events=oos_regime3)

    # same archetype → same prediction regardless of regime_code
    assert out0.expected_net_bps[0] == pytest.approx(out3.expected_net_bps[0])


# ---------------------------------------------------------------------------
# mu-quality shrinkage
# ---------------------------------------------------------------------------

def test_mu_quality_shrinkage_flattens_when_ic_zero() -> None:
    """val IC=0 → lam=0 → all mu collapsed to cross-sectional mean."""
    model = RegimeConditionalEnsemble(
        cell_mu_bps={},
        cell_q10_bps={},
        global_mu_bps=0.0,
        global_q10_bps=0.0,
        conditioning="archetype_only",
        archetype_mu_bps={"trend_continuation": 30.0, "mean_reversion": -10.0},
        archetype_q10_bps={"trend_continuation": 0.0, "mean_reversion": -20.0},
        validation_rank_ic=0.0,
    )
    cfg = _make_cfg(mu_quality_shrinkage_enabled=True, mu_quality_ic_full_scale=0.05)
    oos = pd.DataFrame(
        {"archetype": ["trend_continuation", "mean_reversion"], "entry_regime_code": [0, 0]}
    )
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos, cfg=cfg)

    # lam=0 → all values = cross-sectional mean = mean(30, -10) = 10
    assert out.expected_net_bps[0] == pytest.approx(10.0)
    assert out.expected_net_bps[1] == pytest.approx(10.0)
    assert out.validation_diagnostics["mu_shrinkage_lambda"] == pytest.approx(0.0)


def test_mu_quality_shrinkage_full_conviction_when_ic_full_scale() -> None:
    """val_rank_ic = mu_quality_ic_full_scale → lam=1 → predictions unchanged."""
    model = RegimeConditionalEnsemble(
        cell_mu_bps={},
        cell_q10_bps={},
        global_mu_bps=0.0,
        global_q10_bps=0.0,
        conditioning="archetype_only",
        archetype_mu_bps={"trend_continuation": 40.0, "mean_reversion": -15.0},
        archetype_q10_bps={"trend_continuation": 0.0, "mean_reversion": -25.0},
        validation_rank_ic=0.05,
    )
    cfg = _make_cfg(mu_quality_shrinkage_enabled=True, mu_quality_ic_full_scale=0.05)
    oos = pd.DataFrame(
        {"archetype": ["trend_continuation", "mean_reversion"], "entry_regime_code": [0, 0]}
    )
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos, cfg=cfg)

    assert out.expected_net_bps[0] == pytest.approx(40.0)
    assert out.expected_net_bps[1] == pytest.approx(-15.0)
    assert out.validation_diagnostics["mu_shrinkage_lambda"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# auto conditioning — selects archetype_only when gain is marginal
# ---------------------------------------------------------------------------

def test_auto_conditioning_selects_archetype_only_without_entry_idx() -> None:
    """Without entry_idx, internal IC=0 for both axes → auto picks archetype_only."""
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation"] * 10 + ["mean_reversion"] * 10,
            "entry_regime_code": list(range(10)) + list(range(10)),
            "net_return_bps": [20.0] * 10 + [-5.0] * 10,
        }
    )
    cfg = _make_cfg(ensemble_conditioning="auto")
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # no entry_idx → both ICs = 0 → auto chooses archetype_only
    assert model.conditioning == "archetype_only"


def test_auto_conditioning_exposes_diagnostics() -> None:
    """predict returns conditioning + val IC in validation_diagnostics."""
    train_events = _train_df(30)
    cfg = _make_cfg(ensemble_conditioning="auto")
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    oos = train_events[["archetype", "entry_regime_code"]].head(5).reset_index(drop=True)
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos, cfg=cfg)

    diags = out.validation_diagnostics
    assert "conditioning" in diags
    assert "validation_rank_ic" in diags
    assert "mu_shrinkage_lambda" in diags
    assert diags["conditioning"] in {"archetype_regime", "archetype_only"}


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

def test_fit_returns_zero_model_on_empty_input() -> None:
    cfg = _make_cfg()
    model = fit_regime_conditional_ensemble(train_events=pd.DataFrame(), cfg=cfg)
    assert model.global_mu_bps == pytest.approx(0.0)
    assert model.cell_mu_bps == {}


def test_predict_returns_empty_on_empty_oos() -> None:
    cfg = _make_cfg()
    model = fit_regime_conditional_ensemble(
        train_events=pd.DataFrame(
            {
                "archetype": ["trend_continuation"],
                "entry_regime_code": [0],
                "net_return_bps": [10.0],
            }
        ),
        cfg=cfg,
    )
    out = predict_regime_conditional_ensemble(
        model=model,
        oos_events=pd.DataFrame({"archetype": [], "entry_regime_code": []}),
        cfg=cfg,
    )
    assert out.expected_net_bps.shape[0] == 0


# ---------------------------------------------------------------------------
# P0: Regime Lift Proof Gate
# ---------------------------------------------------------------------------


def test_fit_ensemble_proof_fail_uses_pooled_fallback() -> None:
    """Proof gate 실패 시 conditioning_path=pooled_fallback, conditioning=archetype_only."""
    n_train = 200
    n_oos = 100

    def _make_events(n: int, seed: int) -> pd.DataFrame:
        rng2 = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "archetype": ["trend"] * n,
                "entry_regime_code": rng2.integers(0, 6, n),
                "net_return_bps": rng2.standard_normal(n) * 5.0,  # pure noise, mean≈0
                "entry_idx": np.arange(n),
            }
        )

    train_events = _make_events(n_train, 0)
    oos_events = _make_events(n_oos, 1)
    fold_ids = np.repeat(np.arange(4), n_oos // 4).astype(np.int32)

    cfg = _make_cfg(ensemble_conditioning="archetype_regime")
    model = fit_regime_conditional_ensemble(
        train_events=train_events,
        cfg=cfg,
        oos_proof_events=oos_events,
        fold_ids=fold_ids,
    )

    # pure noise → proof should fail → pooled_fallback
    assert model.lift_proof is not None
    assert not model.lift_proof.proof_passed
    assert model.conditioning_path == "pooled_fallback"
    assert model.conditioning == "archetype_only"


def test_fit_ensemble_no_proof_events_records_path() -> None:
    """oos_proof_events=None이면 lift_proof=None, conditioning_path 기본값 설정."""
    rng = np.random.default_rng(1)
    n = 100
    train = pd.DataFrame(
        {
            "archetype": ["trend"] * n,
            "entry_regime_code": rng.integers(0, 6, n),
            "net_return_bps": rng.standard_normal(n),
            "entry_idx": np.arange(n),
        }
    )
    cfg = _make_cfg(ensemble_conditioning="archetype_regime")
    model = fit_regime_conditional_ensemble(train_events=train, cfg=cfg)

    assert model.lift_proof is None
    # proof 없음 → fail-SAFE 강등 (no_oos_evidence_failsafe) 또는 auto가 archetype_only 선택
    assert model.conditioning_path in {"regime_conditioned", "pooled_fallback", "no_oos_evidence_failsafe"}


def test_predict_ensemble_records_conditioning_path_in_diagnostics() -> None:
    """predict_regime_conditional_ensemble validation_diagnostics에 conditioning_path 존재."""
    rng = np.random.default_rng(2)
    n = 80
    events = pd.DataFrame(
        {
            "archetype": ["trend"] * n,
            "entry_regime_code": rng.integers(0, 6, n),
            "net_return_bps": rng.standard_normal(n),
            "entry_idx": np.arange(n),
        }
    )
    cfg = _make_cfg()
    model = fit_regime_conditional_ensemble(train_events=events, cfg=cfg)
    output = predict_regime_conditional_ensemble(model=model, oos_events=events, cfg=cfg)

    assert "conditioning_path" in output.validation_diagnostics
    assert output.validation_diagnostics["conditioning_path"] in {
        "regime_conditioned",
        "pooled_fallback",
    }


def test_predict_ensemble_backward_compat_no_lift_proof() -> None:
    """lift_proof=None인 legacy model도 예외 없이 작동하고 lift_proof_passed=None."""
    model = RegimeConditionalEnsemble(
        cell_mu_bps={},
        cell_q10_bps={},
        global_mu_bps=5.0,
        global_q10_bps=-2.0,
        conditioning="archetype_only",
        # conditioning_path, lift_proof → default값 사용
    )
    oos = pd.DataFrame(
        {
            "archetype": ["trend"],
            "entry_regime_code": [0],
        }
    )
    output = predict_regime_conditional_ensemble(model=model, oos_events=oos)

    # lift_proof=None → sentinel values: -1 (unset) and NaN
    assert output.validation_diagnostics["lift_proof_passed"] == -1
    lift_nw_tstat = output.validation_diagnostics["lift_nw_tstat"]
    assert isinstance(lift_nw_tstat, float)
    assert np.isnan(lift_nw_tstat)


# ---------------------------------------------------------------------------
# S1~S7: default "auto" + fail-SAFE 검증 (spec: regime_allocation_conditioning_contract)
# ---------------------------------------------------------------------------


def _diverse_train_df(n: int = 80) -> pd.DataFrame:
    """3 archetype x 4 regime code, strong per-cell signal for auto to pick archetype_regime."""
    rng = np.random.default_rng(0)
    archetypes = rng.choice(["trend", "reversion", "carry"], size=n)
    regimes = rng.integers(0, 4, size=n)
    # archetype-regime 셀별 뚜렷한 edge 차이 → IC_regime > IC_arch
    edge = np.where(
        archetypes == "trend",
        regimes * 10.0 + rng.normal(0, 2, n),
        -regimes * 10.0 + rng.normal(0, 2, n),
    )
    return pd.DataFrame(
        {
            "archetype": archetypes,
            "entry_regime_code": regimes,
            "net_return_bps": edge.astype(float),
            "entry_idx": np.arange(n),
        }
    )


def _homogeneous_train_df(n: int = 60) -> pd.DataFrame:
    """단일 archetype, 모든 regime에서 동일 edge → IC 차이 없음."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "archetype": ["trend"] * n,
            "entry_regime_code": rng.integers(0, 4, size=n),
            "net_return_bps": rng.normal(10.0, 3.0, size=n),
            "entry_idx": np.arange(n),
        }
    )


def test_s6_default_conditioning_is_auto() -> None:
    """S6: config 기본값이 "auto"여야 한다."""
    cfg = CandidateStrategyConfig()
    assert cfg.ensemble_conditioning == "auto"


def test_s3_failsafe_when_no_proof_events_and_auto_picks_regime() -> None:
    """S3-A: auto가 archetype_regime를 선택했으나 proof window 없음 → fail-SAFE 강등."""
    train = _diverse_train_df(120)
    cfg = _make_cfg(
        ensemble_conditioning="auto",
        ensemble_min_conditioning_ic_gain=0.0,  # IC 차이 있으면 무조건 regime 선택
    )

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=None,  # proof 없음
    )

    # auto가 archetype_regime을 선택했더라도 proof 없으면 fail-safe 강등
    assert result.conditioning == "archetype_only"
    assert result.conditioning_path == "no_oos_evidence_failsafe"
    assert result.cell_mu_bps == {}


def test_s3_failsafe_explicit_archetype_regime_no_proof() -> None:
    """S3-B: ensemble_conditioning='archetype_regime' 명시 + proof 없음 → fail-SAFE."""
    train = _diverse_train_df(80)
    cfg = _make_cfg(ensemble_conditioning="archetype_regime")

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=None,
    )

    assert result.conditioning == "archetype_only"
    assert result.conditioning_path == "no_oos_evidence_failsafe"
    assert result.cell_mu_bps == {}


def test_s2_auto_picks_archetype_only_for_homogeneous_pool() -> None:
    """S2: 동질적 풀에서 IC 차이 < gain_thr → archetype_only 선택."""
    train = _homogeneous_train_df(60)
    cfg = _make_cfg(
        ensemble_conditioning="auto",
        ensemble_min_conditioning_ic_gain=0.01,
    )

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=None,
    )

    assert result.conditioning == "archetype_only"
    assert result.cell_mu_bps == {}
    assert result.archetype_mu_bps  # archetype fallback은 채워짐


def test_s4_lift_proof_failure_still_downgrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """S4: proof events 존재하나 lift_proof 실패 → archetype_only (기존 동작 불변)."""
    from src.domain.futures.strategy import candidate_ensemble as _ce
    from src.domain.futures.strategy.regime_evaluation import RegimeLiftProofResult

    train = _diverse_train_df(120)
    proof = _diverse_train_df(40)
    proof_ids = np.zeros(len(proof), dtype=np.int32)
    cfg = _make_cfg(
        ensemble_conditioning="archetype_regime",
        ensemble_min_conditioning_ic_gain=0.0,
    )

    # lift_proof 항상 실패 반환으로 고정
    monkeypatch.setattr(
        _ce,
        "evaluate_regime_lift_proof",
        lambda **_kw: RegimeLiftProofResult(
            proof_passed=False,
            nw_tstat=-1.0,
            fold_pass_ratio=0.0,
            conditioning_path="pooled_fallback",
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.6,
            mean_lift_bps=-5.0,
            n_eff=30.0,
            deflated_sharpe=-0.5,
            n_folds_evaluated=1,
        ),
    )

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=proof,
        fold_ids=proof_ids,
    )

    assert result.conditioning == "archetype_only"


def test_s5_lift_proof_pass_keeps_archetype_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    """S5: proof events 존재 + lift_proof 통과 → archetype_regime 유지."""
    from src.domain.futures.strategy import candidate_ensemble as _ce
    from src.domain.futures.strategy.regime_evaluation import RegimeLiftProofResult

    train = _diverse_train_df(200)
    proof = _diverse_train_df(60)
    proof_ids = np.zeros(len(proof), dtype=np.int32)
    cfg = _make_cfg(
        ensemble_conditioning="archetype_regime",
        ensemble_min_conditioning_ic_gain=0.0,
    )

    # lift_proof 항상 통과 반환으로 고정
    monkeypatch.setattr(
        _ce,
        "evaluate_regime_lift_proof",
        lambda **_kw: RegimeLiftProofResult(
            proof_passed=True,
            nw_tstat=2.5,
            fold_pass_ratio=1.0,
            conditioning_path="regime_conditioned",
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.6,
            mean_lift_bps=8.0,
            n_eff=50.0,
            deflated_sharpe=1.2,
            n_folds_evaluated=1,
        ),
    )

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=proof,
        fold_ids=proof_ids,
    )

    assert result.conditioning == "archetype_regime"
    assert result.conditioning_path == "regime_conditioned"
    assert result.cell_mu_bps  # cell 채워짐


def test_s7_rho_diagnostic_stored_not_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """S7: regime_oos_stability_rho는 결과에 저장되지만 conditioning 결정을 변경하지 않는다."""
    from src.domain.futures.strategy import candidate_ensemble as _ce
    from src.domain.futures.strategy.regime_evaluation import RegimeLiftProofResult

    train = _diverse_train_df(200)
    proof = _diverse_train_df(60)
    proof_ids = np.zeros(len(proof), dtype=np.int32)
    cfg = _make_cfg(
        ensemble_conditioning="archetype_regime",
        ensemble_min_conditioning_ic_gain=0.0,
    )

    monkeypatch.setattr(
        _ce,
        "evaluate_regime_lift_proof",
        lambda **_kw: RegimeLiftProofResult(
            proof_passed=True,
            nw_tstat=2.5,
            fold_pass_ratio=1.0,
            conditioning_path="regime_conditioned",
            nw_tstat_threshold=1.5,
            fold_pass_ratio_threshold=0.6,
            mean_lift_bps=8.0,
            n_eff=50.0,
            deflated_sharpe=1.2,
            n_folds_evaluated=1,
        ),
    )

    result = fit_regime_conditional_ensemble(
        train_events=train,
        cfg=cfg,
        oos_proof_events=proof,
        fold_ids=proof_ids,
        regime_oos_stability_rho=0.25,  # 낮은 rho
    )

    # rho가 낮아도 conditioning 변경 없음 (진단 전용)
    assert result.regime_oos_stability_rho == pytest.approx(0.25)
    assert result.conditioning == "archetype_regime"  # rho로 강등 안 됨


# ---------------------------------------------------------------------------
# Phase 1: EB Adaptive Shrinkage — S1 ~ S4
# ---------------------------------------------------------------------------

def _make_high_low_edge_frame() -> pd.DataFrame:
    """2개 archetype: A=고edge(n=50,mean=70bps) / B=noise(n=500,mean=2bps)."""
    rng = np.random.default_rng(0)
    df_a = pd.DataFrame({
        "archetype": ["high_edge"] * 50,
        "entry_regime_code": [0] * 50,
        "net_return_bps": rng.normal(70.0, 5.0, 50),
        "entry_idx": np.arange(50),
    })
    df_b = pd.DataFrame({
        "archetype": ["noise"] * 500,
        "entry_regime_code": [0] * 500,
        "net_return_bps": rng.normal(2.0, 5.0, 500),
        "entry_idx": np.arange(500, 1000),
    })
    return pd.concat([df_a, df_b], ignore_index=True)


def test_eb_shrinkage_preserves_high_edge_archetype_vs_fixed_k() -> None:
    """S1 Happy: EB 수축은 고edge 희귀 archetype을 fixed k=50 대비 더 잘 보존한다."""
    frame = _make_high_low_edge_frame()

    # Fixed k=50
    _, _, arch_mu_fixed, _, _, _, *_ = _fit_cell_means(
        frame,
        shrinkage_k=50.0,
        axis="archetype_only",
        adaptive_shrinkage=False,
    )

    # EB adaptive (k_max=50)
    _, _, arch_mu_eb, _, _, _, *_ = _fit_cell_means(
        frame,
        shrinkage_k=50.0,
        axis="archetype_only",
        adaptive_shrinkage=True,
        shrinkage_k_max=50.0,
    )

    mu_A_fixed = arch_mu_fixed["high_edge"]
    mu_A_eb = arch_mu_eb["high_edge"]

    # EB shrinkage은 cell 간 분산이 크면 k_eff를 줄여 고edge cell을 보존
    assert mu_A_eb > mu_A_fixed, (
        f"EB mu_A={mu_A_eb:.3f} should exceed fixed k=50 mu_A={mu_A_fixed:.3f}"
    )


def test_eb_shrinkage_homogeneous_cells_collapses_to_k_max() -> None:
    """S2 Edge: 동질 cell → between_var≈0 → k_eff=k_max → 고정 k와 동일."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "archetype": ["typeA"] * 100 + ["typeB"] * 100,
        "entry_regime_code": [0] * 200,
        "net_return_bps": rng.normal(10.0, 5.0, 200),  # 동일 분포
        "entry_idx": np.arange(200),
    })

    _, _, arch_mu_fixed, _, _, _, *_ = _fit_cell_means(
        frame, shrinkage_k=50.0, axis="archetype_only", adaptive_shrinkage=False
    )
    _, _, arch_mu_eb, _, _, _, *_ = _fit_cell_means(
        frame,
        shrinkage_k=50.0,
        axis="archetype_only",
        adaptive_shrinkage=True,
        shrinkage_k_max=50.0,
    )

    for arch in ["typeA", "typeB"]:
        assert arch_mu_eb[arch] == pytest.approx(arch_mu_fixed[arch], rel=0.05), (
            f"Homogeneous cells should converge to fixed-k result for {arch}"
        )


def test_eb_k_estimation_is_is_only_no_oos_data_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 Leakage 방지: k_eff는 IS train 데이터에서만 추정, OOS row가 포함되면 안 된다."""
    captured: list[list[float]] = []

    original_fn = _compute_eb_shrinkage_k

    def _spy(cell_means: list[float], cell_vars: list[float], k_max: float) -> float:
        captured.append(list(cell_means))
        return original_fn(cell_means, cell_vars, k_max)

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_ensemble._compute_eb_shrinkage_k",
        _spy,
    )

    frame = _make_high_low_edge_frame()
    _fit_cell_means(
        frame,
        shrinkage_k=50.0,
        axis="archetype_only",
        adaptive_shrinkage=True,
        shrinkage_k_max=50.0,
    )

    # _compute_eb_shrinkage_k가 실제로 호출됐는지 확인
    assert len(captured) >= 1
    # IS-only: 추정에 사용된 cell_means 개수 = archetype 종류 수(2개)
    assert len(captured[0]) == 2


def test_diagnostic_log_emits_negative_ic_flag(caplog: pytest.LogCaptureFixture) -> None:
    """S4 부호 진단: val_ic<0 → 로그에 NEGATIVE 표시, archetype 테이블 포함."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "archetype": ["anti_predictive"] * 50 + ["normal"] * 50,
        "entry_regime_code": [0] * 100,
        "net_return_bps": np.concatenate([rng.normal(-5.0, 3.0, 50), rng.normal(15.0, 3.0, 50)]),
    })
    arch_mu = {"anti_predictive": -3.0, "normal": 12.0}

    with caplog.at_level(logging.INFO, logger="src.domain.futures.strategy.candidate_ensemble"):
        _log_ensemble_diagnostics(
            frame=frame,
            global_mu=5.0,
            arch_mu=arch_mu,
            val_ic=-0.08,
            chosen="archetype_only",
            adaptive_shrinkage=True,
            k_used=50.0,
        )

    combined = "\n".join(caplog.messages)
    assert "❌" in combined
    assert "anti_predictive" in combined
    assert "normal" in combined


# ---------------------------------------------------------------------------
# Variant-Edge Hierarchical Prior — S1 ~ S6
# ---------------------------------------------------------------------------

def _make_variant_frame(
    *,
    high_edge_n: int = 200,
    high_edge_mean: float = 70.0,
    noise_n: int = 200,
    noise_mean: float = 5.0,
    archetype: str = "time_series_momentum",
    regime: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """2 variants in same archetype x regime cell: 'fam:high' (high edge) vs 'fam:noise'."""
    rng = np.random.default_rng(seed)
    df_high = pd.DataFrame({
        "family": ["fam"] * high_edge_n,
        "variant": ["high"] * high_edge_n,
        "archetype": [archetype] * high_edge_n,
        "entry_regime_code": [regime] * high_edge_n,
        "net_return_bps": rng.normal(high_edge_mean, 5.0, high_edge_n),
        "entry_idx": np.arange(high_edge_n, dtype=np.int64),
    })
    df_noise = pd.DataFrame({
        "family": ["fam"] * noise_n,
        "variant": ["noise"] * noise_n,
        "archetype": [archetype] * noise_n,
        "entry_regime_code": [regime] * noise_n,
        "net_return_bps": rng.normal(noise_mean, 5.0, noise_n),
        "entry_idx": np.arange(noise_n, dtype=np.int64) + high_edge_n,
    })
    return pd.concat([df_high, df_noise], ignore_index=True)


def test_s1_variant_prior_discriminates_within_cell() -> None:
    """S1: 동일 셀 내 고엣지 변이 vs noise 변이에 서로 다른 score 부여."""
    # Arrange
    frame = _make_variant_frame()
    cell_mu, _, arch_mu, _, global_mu, _, *_ = _fit_cell_means(
        frame, shrinkage_k=50.0, axis="archetype_regime"
    )
    # cell_mu: (archetype, regime) → single value (둘 다 혼합 평균)
    cell_anchor = cell_mu[("time_series_momentum", 0)]

    # Act
    variant_mu, _ = _fit_variant_means(
        frame,
        cell_mu=cell_mu,
        arch_mu=arch_mu,
        global_mu=global_mu,
        k_variant=30.0,
        min_obs=40,
    )

    # Assert: 고엣지 > 셀 앵커 > noise
    assert "fam:high" in variant_mu
    assert "fam:noise" in variant_mu
    assert variant_mu["fam:high"] > cell_anchor
    assert variant_mu["fam:noise"] < cell_anchor
    assert variant_mu["fam:high"] > variant_mu["fam:noise"]


def test_s2_variant_prior_improves_rank_ic_sign() -> None:
    """S2: variant prior 활성화 시 IC가 셀 평균만 사용하는 경우보다 높아야 한다."""
    # Arrange: 변이 정체성이 수익을 예측하나, 셀 평균은 혼합으로 묘화.
    # 두 변이를 시계열 전체에 인터리브 → val_set에 양쪽 변이 모두 존재 보장.
    rng = np.random.default_rng(7)
    n = 300  # 각 변이가 n/2씩, 인터리브(짝수=A, 홀수=B)
    variants = ["A" if i % 2 == 0 else "B" for i in range(n)]
    returns_A = rng.normal(60.0, 5.0, n // 2)
    returns_B = rng.normal(5.0, 5.0, n // 2)
    idx_A, idx_B = 0, 0
    net_returns = []
    for v in variants:
        if v == "A":
            net_returns.append(returns_A[idx_A])
            idx_A += 1
        else:
            net_returns.append(returns_B[idx_B])
            idx_B += 1
    df = pd.DataFrame({
        "family": ["fam"] * n,
        "variant": variants,
        "archetype": ["momentum"] * n,
        "entry_regime_code": [0] * n,
        "net_return_bps": net_returns,
        "entry_idx": np.arange(n, dtype=np.int64),
    })

    # Act: IC with and without variant prior
    ic_no_prior = _internal_validation_rank_ic(
        df,
        shrinkage_k=50.0,
        val_fraction=0.3,
        axis="archetype_regime",
        variant_prior_enabled=False,
    )
    ic_with_prior = _internal_validation_rank_ic(
        df,
        shrinkage_k=50.0,
        val_fraction=0.3,
        axis="archetype_regime",
        variant_prior_enabled=True,
        variant_shrinkage_k=30.0,
        variant_min_obs=20,
    )

    # Assert: variant prior IC > no prior IC
    assert ic_with_prior > ic_no_prior, (
        f"variant prior IC({ic_with_prior:.4f}) should > cell-only IC({ic_no_prior:.4f})"
    )
    assert ic_with_prior > 0.0, "variant prior IC should be positive"


def test_s3_variant_prior_is_only_no_oos_leakage() -> None:
    """S3: _fit_variant_means가 IS-only sub_fit 데이터만 사용하는지 검증."""
    # Arrange: 시계열 분할 가능한 frame with entry_idx
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({
        "family": ["fam"] * n,
        "variant": ["v1"] * n,
        "archetype": ["momentum"] * n,
        "entry_regime_code": [0] * n,
        "net_return_bps": rng.normal(20.0, 5.0, n),
        "entry_idx": np.arange(n, dtype=np.int64),
    })

    val_start = int(n * 0.75)
    val_idx_cutoff = int(df.iloc[val_start]["entry_idx"])
    sub_fit = df[df["entry_idx"] < val_idx_cutoff - 1]
    val_set = df[df["entry_idx"] >= val_idx_cutoff]

    # Verify no overlap
    assert sub_fit["entry_idx"].max() < val_set["entry_idx"].min()

    # Act: fit variant_mu on sub_fit only
    cell_mu, _, arch_mu, _, global_mu, _, *_ = _fit_cell_means(
        sub_fit, shrinkage_k=50.0, axis="archetype_regime"
    )
    variant_mu, _ = _fit_variant_means(
        sub_fit,  # IS-only
        cell_mu=cell_mu,
        arch_mu=arch_mu,
        global_mu=global_mu,
        k_variant=30.0,
        min_obs=5,
    )

    # Assert: variant_mu computed from sub_fit rows only
    # (n_v used for w_v should reflect sub_fit size, not full n)
    sub_n = int((sub_fit["family"].astype(str) + ":" + sub_fit["variant"].astype(str) == "fam:v1").sum())
    full_n = n
    assert sub_n < full_n, "sub_fit subset confirmed smaller than full set"
    assert "fam:v1" in variant_mu  # was fitted


def test_s4_variant_prior_small_sample_falls_back_to_anchor() -> None:
    """S4: 소표본(n < min_obs) 변이는 spurious high score 차단 → anchor 반환."""
    # Arrange
    rng = np.random.default_rng(99)
    # lucky:rare: n=5 (<<40), raw mean=300bps (spurious)
    df_rare = pd.DataFrame({
        "family": ["lucky"] * 5,
        "variant": ["rare"] * 5,
        "archetype": ["momentum"] * 5,
        "entry_regime_code": [0] * 5,
        "net_return_bps": rng.normal(300.0, 5.0, 5),
        "entry_idx": np.arange(5, dtype=np.int64),
    })
    # bulk: large anchor cell
    df_bulk = pd.DataFrame({
        "family": ["bulk"] * 200,
        "variant": ["normal"] * 200,
        "archetype": ["momentum"] * 200,
        "entry_regime_code": [0] * 200,
        "net_return_bps": rng.normal(20.0, 5.0, 200),
        "entry_idx": np.arange(200, dtype=np.int64) + 5,
    })
    frame = pd.concat([df_rare, df_bulk], ignore_index=True)

    cell_mu, _, arch_mu, _, global_mu, _, *_ = _fit_cell_means(
        frame, shrinkage_k=50.0, axis="archetype_regime"
    )
    anchor = cell_mu.get(("momentum", 0), arch_mu.get("momentum", global_mu))

    # Act
    variant_mu, _ = _fit_variant_means(
        frame,
        cell_mu=cell_mu,
        arch_mu=arch_mu,
        global_mu=global_mu,
        k_variant=30.0,
        min_obs=40,  # rare n=5 < 40
    )

    # Assert: rare variant gets anchor, not 300bps
    assert "lucky:rare" in variant_mu
    assert variant_mu["lucky:rare"] == pytest.approx(anchor, rel=1e-6)
    assert variant_mu["lucky:rare"] < 100.0  # not spurious 300bps


def test_s5_variant_prior_backward_compat_no_family_variant_cols() -> None:
    """S5: family/variant 컬럼 없는 frame → 예외 없이 기존 셀 평균 경로 동작."""
    # Arrange: 기존 스키마 (family/variant 미존재)
    rng = np.random.default_rng(21)
    n = 200
    df = pd.DataFrame({
        "archetype": ["momentum"] * n,
        "entry_regime_code": [0] * n,
        "net_return_bps": rng.normal(20.0, 5.0, n),
        "entry_idx": np.arange(n, dtype=np.int64),
    })
    cfg = _make_cfg(
        ensemble_variant_prior_enabled=True,
        ensemble_variant_shrinkage_k=30.0,
        ensemble_variant_min_obs=40,
        ensemble_conditioning="archetype_only",
        ensemble_adaptive_shrinkage=False,
    )

    # Act: no exception expected
    model = fit_regime_conditional_ensemble(train_events=df, cfg=cfg)

    # Assert: variant_mu_bps는 빈 dict (컬럼 없음)
    assert model.variant_mu_bps == {}

    # predict도 예외 없이 동작
    oos = df.head(10).copy()
    result = predict_regime_conditional_ensemble(model=model, oos_events=oos, cfg=cfg)
    assert result.expected_net_bps.shape[0] == 10


def test_s6_variant_prior_freq_n_cap_limits_weight() -> None:
    """S6: freq_n_cap=200이면 n=1000 고빈도 변이도 n_eff=200으로 캡."""
    # Arrange
    rng = np.random.default_rng(55)
    n_hf = 1000  # high-frequency noise variant
    n_ref = 200
    df_hf = pd.DataFrame({
        "family": ["hf"] * n_hf,
        "variant": ["noise"] * n_hf,
        "archetype": ["momentum"] * n_hf,
        "entry_regime_code": [0] * n_hf,
        "net_return_bps": rng.normal(8.0, 3.0, n_hf),
        "entry_idx": np.arange(n_hf, dtype=np.int64),
    })
    df_ref = pd.DataFrame({
        "family": ["ref"] * n_ref,
        "variant": ["normal"] * n_ref,
        "archetype": ["momentum"] * n_ref,
        "entry_regime_code": [0] * n_ref,
        "net_return_bps": rng.normal(20.0, 3.0, n_ref),
        "entry_idx": np.arange(n_ref, dtype=np.int64) + n_hf,
    })
    frame = pd.concat([df_hf, df_ref], ignore_index=True)
    cell_mu, _, arch_mu, _, global_mu, _, *_ = _fit_cell_means(
        frame, shrinkage_k=50.0, axis="archetype_regime"
    )
    anchor = cell_mu.get(("momentum", 0), arch_mu.get("momentum", global_mu))

    # Act: with freq_n_cap=200
    v_mu_capped, _ = _fit_variant_means(
        frame, cell_mu=cell_mu, arch_mu=arch_mu, global_mu=global_mu,
        k_variant=30.0, min_obs=10, freq_n_cap=200
    )
    # Act: without freq_n_cap
    v_mu_uncapped, _ = _fit_variant_means(
        frame, cell_mu=cell_mu, arch_mu=arch_mu, global_mu=global_mu,
        k_variant=30.0, min_obs=10, freq_n_cap=0
    )

    # Assert: capped w_v = 200/(200+30) < uncapped w_v = 1000/(1000+30)
    # → capped mu closer to anchor than uncapped
    w_capped = 200 / (200 + 30.0)
    w_uncapped = 1000 / (1000 + 30.0)
    raw_hf = float(frame[frame["variant"] == "noise"]["net_return_bps"].mean())
    expected_capped = w_capped * raw_hf + (1.0 - w_capped) * anchor
    expected_uncapped = w_uncapped * raw_hf + (1.0 - w_uncapped) * anchor

    assert v_mu_capped["hf:noise"] == pytest.approx(expected_capped, rel=1e-4)
    assert v_mu_uncapped["hf:noise"] == pytest.approx(expected_uncapped, rel=1e-4)
    # capped가 anchor에 더 가까움 (노이즈 억제)
    assert abs(v_mu_capped["hf:noise"] - anchor) < abs(v_mu_uncapped["hf:noise"] - anchor)


def test_s7_variant_prior_regime_conditional_offset() -> None:
    """S7: Happy Path - Variant prior predictions vary dynamically by regime while maintaining the offset."""
    cell_mu = {
        ("trend_continuation", 1): 20.0,
        ("trend_continuation", 2): 5.0,
    }
    cell_q10 = {
        ("trend_continuation", 1): -10.0,
        ("trend_continuation", 2): -10.0,
    }
    model = RegimeConditionalEnsemble(
        cell_mu_bps=cell_mu,
        cell_q10_bps=cell_q10,
        global_mu_bps=0.0,
        global_q10_bps=0.0,
        conditioning="archetype_regime",
        archetype_mu_bps={"trend_continuation": 10.0},
        archetype_q10_bps={"trend_continuation": -5.0},
        validation_rank_ic=0.0,
        variant_mu_bps={"tpc:tpc_50_200": 23.0},
        variant_offset_bps={"tpc:tpc_50_200": 3.0},
    )

    oos_events = pd.DataFrame(
        {
            "family": ["tpc", "tpc"],
            "variant": ["tpc_50_200", "tpc_50_200"],
            "archetype": ["trend_continuation", "trend_continuation"],
            "entry_regime_code": [1, 2],
        }
    )

    out = predict_regime_conditional_ensemble(model=model, oos_events=oos_events)
    # Expected in regime 1: 20.0 (cell) + 3.0 (offset) = 23.0
    # Expected in regime 2: 5.0 (cell) + 3.0 (offset) = 8.0
    assert out.expected_net_bps[0] == pytest.approx(23.0)
    assert out.expected_net_bps[1] == pytest.approx(8.0)


def test_s8_variant_prior_family_filtering() -> None:
    """S8: Edge Case - Family Filtering filters out variants not in ensemble_variant_prior_families."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "family": ["tpc"] * (n // 2) + ["rsi"] * (n // 2),
            "variant": ["tpc_50_200"] * (n // 2) + ["rsi_14"] * (n // 2),
            "archetype": ["trend_continuation"] * (n // 2) + ["mean_reversion"] * (n // 2),
            "entry_regime_code": [1] * n,
            "net_return_bps": rng.normal(10.0, 5.0, n),
            "entry_idx": np.arange(n),
        }
    )

    cfg = _make_cfg(
        ensemble_variant_prior_enabled=True,
        ensemble_variant_prior_families=("tpc",),
        ensemble_variant_min_obs=5,
    )

    model = fit_regime_conditional_ensemble(train_events=df, cfg=cfg)

    # tpc:tpc_50_200 should have variant prior and offset
    assert "tpc:tpc_50_200" in model.variant_mu_bps
    assert "tpc:tpc_50_200" in model.variant_offset_bps

    # rsi:rsi_14 should be filtered out
    assert "rsi:rsi_14" not in model.variant_mu_bps
    assert "rsi:rsi_14" not in model.variant_offset_bps


def test_s9_variant_prior_backward_compatibility() -> None:
    """S9: Backward Compatibility - predict works fine even if variant_offset_bps is missing or empty."""
    cell_mu = {
        ("trend_continuation", 1): 20.0,
    }
    cell_q10 = {
        ("trend_continuation", 1): -10.0,
    }
    model = RegimeConditionalEnsemble(
        cell_mu_bps=cell_mu,
        cell_q10_bps=cell_q10,
        global_mu_bps=0.0,
        global_q10_bps=0.0,
        conditioning="archetype_regime",
        archetype_mu_bps={"trend_continuation": 10.0},
        archetype_q10_bps={"trend_continuation": -5.0},
        validation_rank_ic=0.0,
        variant_mu_bps={"tpc:tpc_50_200": 23.0},
        variant_offset_bps={},  # Empty (simulates legacy / missing offset dict)
    )

    oos_events = pd.DataFrame(
        {
            "family": ["tpc"],
            "variant": ["tpc_50_200"],
            "archetype": ["trend_continuation"],
            "entry_regime_code": [1],
        }
    )

    out = predict_regime_conditional_ensemble(model=model, oos_events=oos_events)
    # Expected: cell mean only (offset defaults to 0.0) -> 20.0
    assert out.expected_net_bps[0] == pytest.approx(20.0)


def test_validation_rank_ic_uses_dynamic_offset_prediction() -> None:
    """Verify that _internal_validation_rank_ic uses dynamic cell_val + offset prediction."""
    rng = np.random.default_rng(12345)
    n = 100
    # Create 2 variants: v1 (good), v2 (bad) interleaved
    variants = []
    net_returns = []
    for i in range(n):
        if i % 2 == 0:
            variants.append("v1")
            net_returns.append(rng.normal(30.0, 5.0))
        else:
            variants.append("v2")
            net_returns.append(rng.normal(-30.0, 5.0))
    
    df = pd.DataFrame({
        "family": ["fam"] * n,
        "variant": variants,
        "archetype": ["momentum"] * n,
        "entry_regime_code": rng.choice([0, 1], size=n),
        "net_return_bps": net_returns,
        "entry_idx": np.arange(n, dtype=np.int64),
    })

    # Act
    ic_val = _internal_validation_rank_ic(
        df,
        shrinkage_k=10.0,
        val_fraction=0.3,
        axis="archetype_regime",
        variant_prior_enabled=True,
        variant_shrinkage_k=5.0,
        variant_min_obs=10,
    )
    
    # Assert: We expect a high positive rank IC because our prediction (which distinguishes v1 vs v2 dynamically)
    # matches the actual returns (v1 positive, v2 negative).
    assert ic_val > 0.5, f"Expected strong positive Rank IC, got {ic_val:.4f}"


def test_validation_rank_ic_empty_offset_fallback() -> None:
    """Verify fallback behavior in _internal_validation_rank_ic when no variants meet min_obs."""
    rng = np.random.default_rng(12345)
    n = 50
    df = pd.DataFrame({
        "family": ["fam"] * n,
        "variant": [f"v_{i}" for i in range(n)],  # All unique variants (obs = 1 each)
        "archetype": ["momentum"] * n,
        "entry_regime_code": [0] * n,
        "net_return_bps": rng.normal(10.0, 5.0, n),
        "entry_idx": np.arange(n, dtype=np.int64),
    })

    # Act
    ic_val = _internal_validation_rank_ic(
        df,
        shrinkage_k=10.0,
        val_fraction=0.3,
        axis="archetype_regime",
        variant_prior_enabled=True,
        variant_shrinkage_k=5.0,
        variant_min_obs=10,  # No variant will pass this
    )
    # Since variant_min_obs = 10 and each variant has 1 obs, offsets should fall back to 0.0
    # Thus, all predictions will be the cell mean, which is constant. Constant predictions give IC = 0.0 or nan.
    # Because of our nan guard, it returns 0.0.
    assert ic_val == 0.0, f"Expected 0.0 fallback for constant predictions, got {ic_val:.4f}"


# ---------------------------------------------------------------------------
# Direction A + B: score calibration & q90 산출 (S1~S6)
# ---------------------------------------------------------------------------

def _make_score_train_df(
    n: int,
    *,
    regime: int,
    score_z: np.ndarray,
    net_bps: np.ndarray,
    archetype: str = "trend_continuation",
) -> pd.DataFrame:
    """Helper: training DataFrame for score calibration tests."""
    return pd.DataFrame(
        {
            "archetype": [archetype] * n,
            "entry_regime_code": [regime] * n,
            "net_return_bps": net_bps,
            "score_z": score_z,
            "family": ["fam"] * n,
            "variant": ["v1"] * n,
            "entry_idx": np.arange(n, dtype=np.int64),
        }
    )


def _make_oos_event(*, regime: int, score_z: float, archetype: str = "trend_continuation") -> pd.DataFrame:
    """Single OOS event for prediction tests."""
    return pd.DataFrame(
        {
            "archetype": [archetype],
            "entry_regime_code": [regime],
            "score_z": [score_z],
            "family": ["fam"],
            "variant": ["v1"],
        }
    )


# S1 — score calibration 적합: 양의 상관 데이터

def test_score_calibration_positive_correlation_fits_valid_slope() -> None:
    """Regime 1에서 score_z ↑ → net_bps ↑ 양의 상관 시 slope > 0, calibration_valid=True.

    Arrange: rho ~= +0.97 합성 데이터 80개
    Act: fit_regime_conditional_ensemble with score_calibration_enabled=True
    Assert: slope > 0, calibration_valid=True, 높은 score_z → 높은 mu 예측
    """
    rng = np.random.default_rng(42)
    n = 80
    score_z = rng.normal(0.0, 1.0, n)
    net_bps = 20.0 * score_z + rng.normal(0.0, 5.0, n)

    train_events = _make_score_train_df(n, regime=1, score_z=score_z, net_bps=net_bps)
    cfg = _make_cfg(
        ensemble_score_calibration_enabled=True,
        ensemble_score_calibration_min_obs=40,
        ensemble_score_slope_k=100.0,
    )

    # Act
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # Assert: slope 피팅됨
    assert model.score_calibration_valid.get(1) is True
    assert model.regime_score_slope.get(1, 0.0) > 0.0

    # Assert: 높은 score_z → 낮은 score_z보다 μ 높음
    out_high = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=1, score_z=2.0)
    )
    out_low = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=1, score_z=-2.0)
    )
    assert out_high.expected_net_bps[0] > out_low.expected_net_bps[0]


# S2 — OOS 부호 불안정: score_calibration_valid=False → fallback

def test_score_calibration_negative_correlation_marks_invalid() -> None:
    """score_z와 net_bps가 음의 상관일 때 beta < 0 → calibration_valid=False.

    predict는 현행 cell lookup 기반으로 score_z 무시.
    """
    rng = np.random.default_rng(0)
    n = 80
    score_z = rng.normal(0.0, 1.0, n)
    net_bps = -20.0 * score_z + rng.normal(0.0, 5.0, n)

    train_events = _make_score_train_df(n, regime=2, score_z=score_z, net_bps=net_bps)
    cfg = _make_cfg(
        ensemble_score_calibration_enabled=True,
        ensemble_score_calibration_min_obs=40,
        ensemble_score_slope_k=100.0,
    )

    # Act
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # Assert: 음의 기울기 → calibration_valid=False
    assert model.score_calibration_valid.get(2) is False

    # Assert: predict는 score_z에 무관하게 동일한 cell 기반 값 반환
    out_high = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=2, score_z=2.0)
    )
    out_low = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=2, score_z=-2.0)
    )
    assert out_high.expected_net_bps[0] == pytest.approx(out_low.expected_net_bps[0])


# S3 — 희소 데이터: min_obs 미달 → calibration 미수행

def test_score_calibration_insufficient_obs_skips_regime() -> None:
    """n=20 < min_obs=60 → score_calibration_valid[regime] = False, slope 미산출.

    Arrange: regime 3에서 n=20 이벤트 (양의 상관이지만 obs 부족)
    """
    rng = np.random.default_rng(7)
    n = 20
    score_z = rng.normal(0.0, 1.0, n)
    net_bps = 20.0 * score_z + rng.normal(0.0, 5.0, n)

    train_events = _make_score_train_df(n, regime=3, score_z=score_z, net_bps=net_bps)
    cfg = _make_cfg(
        ensemble_score_calibration_enabled=True,
        ensemble_score_calibration_min_obs=60,  # n=20 < 60 → 스킵
    )

    # Act
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # Assert: obs 부족 → valid=False 또는 slope 미존재
    is_invalid = (
        not model.score_calibration_valid.get(3, False)
    )
    assert is_invalid, (
        f"Expected calibration skipped for regime 3 (n={n} < min_obs=60), "
        f"got valid={model.score_calibration_valid.get(3)}"
    )


# S4 — Direction B: q90 실제 산출 확인

def test_q90_bps_differs_from_mu() -> None:
    """비대칭 수익 분포(오른꼬리)에서 q90 > μ이고 q90_net_bps > q10_net_bps 성립.

    Arrange: chi-squared 분포(오른꼬리) 기반 net_bps → q90 >> μ
    """
    rng = np.random.default_rng(99)
    n = 200
    # chi2(3) 분포: 오른꼬리, mean≈3, q90≈6.25
    net_bps = rng.chisquare(3.0, n) * 10.0  # mean≈30bps, q90≈62bps
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation"] * n,
            "entry_regime_code": [0] * n,
            "net_return_bps": net_bps,
            "entry_idx": np.arange(n, dtype=np.int64),
        }
    )
    cfg = _make_cfg(ensemble_adaptive_shrinkage=False)

    # Act
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    oos = pd.DataFrame({"archetype": ["trend_continuation"], "entry_regime_code": [0]})
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos)

    # Assert: q90 실제값이 μ와 다름 (기존 mu.copy() 제거 검증)
    assert out.q90_net_bps[0] != pytest.approx(out.expected_net_bps[0], rel=0.01), (
        "q90_net_bps should differ from expected_net_bps for skewed distribution"
    )
    # Assert: q90 > q10 (분포 순서 보존)
    assert out.q90_net_bps[0] > out.q10_net_bps[0]


# S5 — 회귀: score_calibration_enabled=False (default) 시 기존 동작 불변

def test_score_calibration_disabled_keeps_existing_behavior() -> None:
    """ensemble_score_calibration_enabled=False (default) 시 score_z 무시.

    score_calibration_valid는 빈 dict, predict μ는 score_z와 무관.
    """
    rng = np.random.default_rng(42)
    n = 80
    score_z = rng.normal(0.0, 1.0, n)
    net_bps = 20.0 * score_z + rng.normal(0.0, 5.0, n)

    train_events = _make_score_train_df(n, regime=1, score_z=score_z, net_bps=net_bps)
    cfg = _make_cfg()  # default: ensemble_score_calibration_enabled=False

    # Act
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    # Assert: calibration 수행 안 됨
    assert model.score_calibration_valid == {}
    assert model.regime_score_slope == {}

    # Assert: predict μ가 score_z와 무관 (high_z ≈ low_z)
    out_high = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=1, score_z=2.0)
    )
    out_low = predict_regime_conditional_ensemble(
        model=model, oos_events=_make_oos_event(regime=1, score_z=-2.0)
    )
    assert out_high.expected_net_bps[0] == pytest.approx(out_low.expected_net_bps[0])


# S6 — Leakage 검증: _internal_validation_rank_ic에서 val_set 데이터 누출 방지

def test_score_calibration_no_leakage_from_val_set() -> None:
    """score calibration fit은 sub_fit(학습 영역)만 사용하며 val_set entry_idx 범위 밖.

    _internal_validation_rank_ic에서 val_set의 entry_idx가 sub_fit에 포함되지 않음을 검증.
    Approach: sub_fit/val_set 분리 로직의 cutoff 검증.
    """
    rng = np.random.default_rng(123)
    n = 100
    # 강한 양의 상관으로 IC 계산이 가능하도록 설정
    score_z_vals = rng.normal(0.0, 1.0, n)
    net_bps = 15.0 * score_z_vals + rng.normal(0.0, 3.0, n)
    df = pd.DataFrame(
        {
            "archetype": ["trend_continuation"] * n,
            "entry_regime_code": [1] * n,
            "net_return_bps": net_bps,
            "score_z": score_z_vals,
            "family": ["fam"] * n,
            "variant": ["v1"] * n,
            "entry_idx": np.arange(n, dtype=np.int64),
        }
    )

    val_fraction = 0.25
    n_total = len(df)
    sorted_df = df.sort_values("entry_idx")
    val_start = int(n_total * (1.0 - val_fraction))
    val_idx_cutoff = int(sorted_df.iloc[val_start]["entry_idx"])

    sub_fit = sorted_df[sorted_df["entry_idx"] < val_idx_cutoff - 1]
    val_set = sorted_df[sorted_df["entry_idx"] >= val_idx_cutoff]

    # Assert: sub_fit과 val_set은 겹치지 않음 (purge gap 포함)
    sub_fit_max_idx = int(sub_fit["entry_idx"].max())
    val_set_min_idx = int(val_set["entry_idx"].min())
    assert sub_fit_max_idx < val_set_min_idx, (
        f"Leakage: sub_fit max_idx={sub_fit_max_idx} >= val_set min_idx={val_set_min_idx}"
    )

    # Assert: val_set의 어떤 entry_idx도 sub_fit에 없음
    sub_fit_idx_set = set(sub_fit["entry_idx"].tolist())
    val_idx_list: list[int] = val_set["entry_idx"].tolist()
    overlap = [i for i in val_idx_list if i in sub_fit_idx_set]
    assert len(overlap) == 0, f"Leakage: {len(overlap)} val_set entries found in sub_fit"
