from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_ensemble import (
    RegimeConditionalEnsemble,
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

def test_regime_conditional_ensemble_shrinks_small_cells_to_global() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [100.0, 100.0, -100.0],
        }
    )
    cfg = _make_cfg(ensemble_shrinkage_k=50.0, ensemble_conditioning="archetype_regime")

    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    global_mu = (100.0 + 100.0 - 100.0) / 3.0
    shrunk_small_cell = model.cell_mu_bps[("mean_reversion", 4)]

    assert model.global_mu_bps == pytest.approx(global_mu)
    assert -100.0 < shrunk_small_cell < global_mu
    assert model.conditioning == "archetype_regime"


def test_ensemble_predict_lookup_matches_cell_estimate() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [40.0, 60.0, 10.0],
        }
    )
    cfg = _make_cfg(ensemble_shrinkage_k=1.0, ensemble_conditioning="archetype_regime")
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    oos_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "mean_reversion", "unseen"],
            "entry_regime_code": [0, 4, 9],
        }
    )

    # cfg=None → no mu_quality shrinkage, raw cell lookup
    out = predict_regime_conditional_ensemble(model=model, oos_events=oos_events)

    assert out.expected_net_bps[0] == pytest.approx(model.cell_mu_bps[("trend_continuation", 0)])
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
    assert model.conditioning_path in {"regime_conditioned", "pooled_fallback"}


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
    assert np.isnan(float(output.validation_diagnostics["lift_nw_tstat"]))
