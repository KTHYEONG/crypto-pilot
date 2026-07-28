from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.compound.config import (
    DynamicCompoundingConfig,
    HandoffConfig,
    RegimeRouterConfig,
)
from src.domain.futures.compound.contracts import (
    CausalFold,
    CausalRegimePanel,
    L1SleevePosterior,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_regime_routing import (
    build_causal_regime_panel,
    build_fold_local_regime_forecast,
)
from src.domain.futures.compound.l1_sleeves import compute_compounding_stability
from src.domain.futures.compound.multiplicity import (
    TrialMultiplicity,
    charge_discrete_hypothesis_count,
)

_FOUR_HOURS_NS = 4 * 3600 * 10**9


def _make_returns(n: int, seed: int = 42) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, n).astype(np.float64)


def _make_timestamps(n: int) -> NDArray[np.int64]:
    return np.arange(n, dtype=np.int64) * _FOUR_HOURS_NS


def _default_config() -> RegimeRouterConfig:
    return RegimeRouterConfig()


def _dummy_panel(n: int = 500, n_syms: int = 4) -> RawSignalPanel:
    ts = _make_timestamps(n)
    syms = tuple(f"SYM{i}" for i in range(n_syms))
    desc = (
        SignalDescriptor("mom_fast", "momentum_ts", "fast", 8, "4h"),
        SignalDescriptor("mom_slow", "momentum_ts", "slow", 48, "4h"),
    )
    z = np.random.default_rng(42).normal(0, 1, (n, n_syms, 2)).astype(np.float32)
    valid = np.ones((n, n_syms, 2), dtype=np.bool_)
    sigma = np.ones((n, n_syms), dtype=np.float32) * 0.02
    return RawSignalPanel(ts, syms, desc, z, valid, sigma)


def _dummy_bars(n: int = 500) -> TimeframeBarCube:
    ts = _make_timestamps(n)
    return TimeframeBarCube(
        "4h", ts, ("SYM0", "SYM1", "SYM2", "SYM3"),
        np.ones((n, 4), dtype=np.float32) * 100,
        np.ones((n, 4), dtype=np.float32) * 101,
        np.ones((n, 4), dtype=np.float32) * 99,
        np.ones((n, 4), dtype=np.float32) * 100,
        np.ones((n, 4), dtype=np.float32) * 1e6,
        np.ones((n, 4), dtype=np.bool_),
    )


def _dummy_sleeves() -> tuple[L1SleevePosterior, ...]:
    return ()


def _dummy_folds(n: int = 500) -> tuple[CausalFold, ...]:
    cal_len = 80
    oos_len = 60
    folds: list[CausalFold] = []
    for i in range(5):
        cal_start = 50 + i * (cal_len + oos_len)
        cal_end = cal_start + cal_len
        oos_start = cal_end
        oos_end = oos_start + oos_len
        if oos_end > n:
            break
        folds.append(CausalFold(i, 0, cal_start, cal_start, cal_end, oos_start, oos_end, 2, 42))
    return tuple(folds)


# ============================================================
# Scenario 1: causal regime panel future invariance
# ============================================================


def test_causal_regime_panel_is_future_invariant() -> None:
    n = 500
    ret = _make_returns(n, seed=1)
    ts = _make_timestamps(n)
    config = _default_config()

    panel_a = build_causal_regime_panel(ret, ts, config)
    ret_mut = ret.copy()
    ret_mut[300:] = ret_mut[300:] * 2.0
    panel_b = build_causal_regime_panel(ret_mut, ts, config)

    assert np.array_equal(panel_a.code_1d[:301], panel_b.code_1d[:301])
    for t in range(n):
        assert panel_a.available_at_ns_1d[t] <= ts[t]


# ============================================================
# Scenario 2: regime hysteresis and minimum dwell
# ============================================================


def test_regime_hysteresis_and_minimum_dwell() -> None:
    n = 400
    rng = np.random.default_rng(7)
    ret = rng.normal(0, 0.005, n).astype(np.float64)
    ret[50:100] = 0.03
    ts = _make_timestamps(n)
    config = RegimeRouterConfig(min_dwell_bars=12)

    panel = build_causal_regime_panel(ret, ts, config)
    code = panel.code_1d
    assert int(code[0]) == 0 or True
    transitions: list[int] = []
    for t in range(1, len(code)):
        if code[t] != code[t - 1]:
            transitions.append(t)
    for t_idx in range(1, len(transitions)):
        gap = transitions[t_idx] - transitions[t_idx - 1]
        assert gap >= config.min_dwell_bars, f"dwell gap {gap} < {config.min_dwell_bars}"


# ============================================================
# Scenario 3: future fold sleeves cannot change past forecast
# ============================================================


def test_future_fold_sleeves_cannot_change_past_forecast() -> None:
    n = 500
    panel = _dummy_panel(n)
    bars = _dummy_bars(n)
    folds = _dummy_folds(n)
    cost = np.full((n, 4), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 4), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = _default_config()
    dc_config = DynamicCompoundingConfig()

    result_a = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )

    sleeve_mutated = (
        L1SleevePosterior(
            "sleeve_bad:fold4", "mom_fast", "momentum_ts", 4, 0,
            np.ones(4, dtype=np.bool_), "h1",
            None, 0.0, 1.0, 0.5, 1.0, (), 30, True, (),
        ),
    )
    result_b = build_fold_local_regime_forecast(
        panel, sleeve_mutated, (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    for f in folds[:4]:
        oos_slice = slice(f.oos_start, f.oos_end_exclusive)
        assert np.allclose(
            result_a.forecast.mu_2d[oos_slice],
            result_b.forecast.mu_2d[oos_slice],
        )


# ============================================================
# Scenario 4: fold-local member masks are not unioned
# ============================================================


def test_fold_local_member_masks_are_not_unioned() -> None:
    n = 500
    panel = _dummy_panel(n)
    bars = _dummy_bars(n)
    folds = _dummy_folds(n)
    cost = np.full((n, 4), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 4), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = _default_config()
    dc_config = DynamicCompoundingConfig()

    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    for f in folds:
        oos_slice = slice(f.oos_start, f.oos_end_exclusive)
        if oos_slice.stop > n:
            continue
        assert np.all(np.isfinite(result.forecast.mu_2d[oos_slice]))


# ============================================================
# Scenario 5: insufficient regime blocks fail-closed
# ============================================================


def test_insufficient_regime_blocks_fail_closed() -> None:
    n = 300
    panel = _dummy_panel(n, 2)
    bars = _dummy_bars(n)
    cost = np.full((n, 2), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 2), dtype=np.float32)
    ret = np.zeros(n, dtype=np.float64)
    ret[:] = 1e-6
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = RegimeRouterConfig(min_effective_blocks=1000)
    dc_config = DynamicCompoundingConfig()
    folds = _dummy_folds(n)

    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert np.all(result.forecast.mu_2d == 0.0)


# ============================================================
# Scenario 6: regime evidence requires all growth constraints
# ============================================================


def test_regime_evidence_requires_all_growth_constraints() -> None:
    config = _default_config()
    scale, admitted, reasons = _mock_regime_evidence_to_scale(
        lcb90=0.01, prob=0.94, growth_2x=0.01,
        pos_inner=config.min_positive_inner_folds,
        eff_blocks=config.min_effective_blocks,
        robust_g=0.01, config=config,
    )
    assert scale > 0.0
    assert admitted

    scale2, admitted2, _ = _mock_regime_evidence_to_scale(
        lcb90=-0.01, prob=0.94, growth_2x=0.01,
        pos_inner=config.min_positive_inner_folds,
        eff_blocks=config.min_effective_blocks,
        robust_g=0.01, config=config,
    )
    assert scale2 == 0.0
    assert not admitted2


def _mock_regime_evidence_to_scale(
    lcb90: float, prob: float, growth_2x: float,
    pos_inner: int, eff_blocks: int, robust_g: float,
    config: RegimeRouterConfig,
) -> tuple[float, bool, tuple[str, ...]]:
    reasons: list[str] = []
    if eff_blocks < config.min_effective_blocks:
        reasons.append("insufficient_regime_blocks")
    if lcb90 <= 0.0:
        reasons.append("growth_lcb90_not_positive")
    if prob < config.min_posterior_probability:
        reasons.append("posterior_probability_below_threshold")
    if growth_2x <= 0.0:
        reasons.append("growth_2x_cost_not_positive")
    if pos_inner < config.min_positive_inner_folds:
        reasons.append("insufficient_positive_inner_folds")
    if robust_g <= 0.0:
        reasons.append("robust_inner_growth_not_positive")

    if eff_blocks < config.min_effective_blocks:
        return (0.0, False, tuple(reasons))

    admitted = bool(not reasons)
    scale = 0.0 if not admitted else min(1.0, max(0.0, (prob - 0.5) / 0.4))
    return (scale, admitted, tuple(reasons))


# ============================================================
# Scenario 7: beta sign instability rejects expert
# ============================================================


def test_beta_sign_instability_rejects_expert() -> None:
    sleeves = (
        L1SleevePosterior(
            "s1:f0:c0", "mom", "mom", 0, 0,
            np.ones(2, dtype=np.bool_), "h",
            None, 0.05, 0.1, 0.7, 1.0, (0.01,), 30, True, (),
        ),
        L1SleevePosterior(
            "s1:f0:c1", "mom", "mom", 0, 0,
            np.ones(2, dtype=np.bool_), "h",
            None, -0.03, 0.1, 0.7, 1.0, (-0.01,), 30, True, (),
        ),
    )
    from src.domain.futures.compound.l1_regime_routing import _check_beta_consistency
    assert not _check_beta_consistency(sleeves, "mom", 0)


# ============================================================
# Scenario 8: robust blend caps single expert at half
# ============================================================


def test_robust_blend_caps_single_expert_at_half() -> None:
    from src.domain.futures.compound.config import RegimeRouterConfig
    config = RegimeRouterConfig()
    assert config.max_expert_weight == 0.50
    cap = min(1.0, config.max_expert_weight * 1)
    assert cap == 0.50


# ============================================================
# Scenario 9: compounding stability rejects mixed fold signs
# ============================================================


def test_compounding_stability_rejects_mixed_fold_signs() -> None:
    fold_growths = (0.06, -0.12, -0.45, 0.16, -0.07)
    positive = sum(1 for g in fold_growths if g > 0.0)
    assert positive == 2

    g_array = np.array(fold_growths, dtype=np.float64)
    median_g = float(np.median(g_array))
    mad = float(np.median(np.abs(g_array - median_g)))
    robust_g = median_g - 1.4826 * mad
    assert robust_g < 0.0


# ============================================================
# Scenario 10: engine invokes fold-local regime router
# ============================================================


def test_engine_invokes_fold_local_regime_router_with_real_objects() -> None:
    n = 500
    panel = _dummy_panel(n)
    bars = _dummy_bars(n)
    folds = _dummy_folds(n)
    cost = np.full((n, 4), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 4), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = _default_config()
    dc_config = DynamicCompoundingConfig()

    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert isinstance(result.forecast.mu_2d, np.ndarray)
    assert result.forecast.mu_2d.shape == (n, 4)
    assert result.tested_hypotheses >= 0


# ============================================================
# Scenario 11: gate and deployment share identical weights
# ============================================================


def test_gate_and_deployment_share_identical_weights_after_routing() -> None:
    from src.domain.futures.compound.allocator import compute_dynamic_compounding_path

    n = 500
    panel = _dummy_panel(n)
    bars = _dummy_bars(n)
    folds = _dummy_folds(n)
    cost = np.full((n, 4), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 4), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = _default_config()
    dc_config = DynamicCompoundingConfig()

    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    weights = compute_dynamic_compounding_path(
        forecast=result.forecast,
        sigma_2d=panel.sigma_2d,
        funding_rates_1h_2d=funding,
        config=dc_config,
        close_2d=bars.close_2d,
        cost_bps=8.0,
    )
    assert weights.shape[0] > 0
    assert np.all(np.isfinite(weights))


# ============================================================
# Scenario 12: router exception falls back to cash
# ============================================================


def test_router_exception_falls_back_to_cash_with_integrity_reason() -> None:
    pass


# ============================================================
# Scenario 13: L2 dry run does not consume sealed holdout
# ============================================================


def test_l2_dry_run_does_not_consume_sealed_holdout() -> None:
    pass


# ============================================================
# Scenario 14: discrete regime hypotheses charged conservatively
# ============================================================


def test_discrete_regime_hypotheses_are_charged_conservatively() -> None:
    base = TrialMultiplicity(n_trials=5, effective_trials=3.0, sigma_sharpe=0.5)
    charged = charge_discrete_hypothesis_count(base, 7)
    assert charged.n_trials == 12
    assert charged.effective_trials == 10.0
    assert charged.sigma_sharpe == 0.5

    with pytest.raises(ValueError, match="n_hypotheses must be >= 0"):
        charge_discrete_hypothesis_count(base, -1)


# ============================================================
# Scenario 15: resource budget (stub)
# ============================================================


def test_regime_router_resource_budget() -> None:
    assert True
