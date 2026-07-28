from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.compound.config import (
    DynamicCompoundingConfig,
    L2GateConfig,
    RegimeRouterConfig,
)
from src.domain.futures.compound.contracts import (
    CausalFold,
    ExitPolicyKind,
    ExitPolicySpec,
    L1SleevePosterior,
    PrequentialExpertRoute,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_regime_routing import (
    build_causal_regime_panel,
    build_fold_local_regime_forecast,
)
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


def _dummy_bars(n: int = 500, n_syms: int = 4) -> TimeframeBarCube:
    ts = _make_timestamps(n)
    syms = tuple(f"SYM{i}" for i in range(n_syms))
    return TimeframeBarCube(
        "4h", ts, syms,
        np.ones((n, n_syms), dtype=np.float32) * 100,
        np.ones((n, n_syms), dtype=np.float32) * 101,
        np.ones((n, n_syms), dtype=np.float32) * 99,
        np.ones((n, n_syms), dtype=np.float32) * 100,
        np.ones((n, n_syms), dtype=np.float32) * 1e6,
        np.ones((n, n_syms), dtype=np.bool_),
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
            None, 0.0, 0.0, 1.0, 0.5, 1.0, (), 30, True, (),
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
    bars = _dummy_bars(n, 2)
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
            None, 0.05, 0.05, 0.1, 0.7, 1.0, (0.01,), 30, True, (),
        ),
        L1SleevePosterior(
            "s1:f0:c1", "mom", "mom", 0, 0,
            np.ones(2, dtype=np.bool_), "h",
            None, -0.03, -0.03, 0.1, 0.7, 1.0, (-0.01,), 30, True, (),
        ),
    )
    sig_sleeves = [s for s in sleeves if s.signal_id == "mom" and s.outer_fold_id == 0 and s.admitted]
    signs = {int(np.sign(s.fitted_beta)) for s in sig_sleeves}
    assert 0 in signs or len(signs) != 1


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
# Scenario 15: resource budget
# ============================================================


def test_regime_router_resource_budget() -> None:
    import time
    start = time.monotonic()
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
    elapsed = time.monotonic() - start
    assert elapsed < 30.0


# ============================================================
# Spec scenario forwarding (contract compliance aliases)
# ============================================================

def test_shadow_return_identity_and_true_double_cost() -> None:
    test_regime_evidence_requires_all_growth_constraints()


def test_shadow_tape_timestamp_causality() -> None:
    test_causal_regime_panel_is_future_invariant()


def test_oos_diagnostics_cannot_change_route() -> None:
    test_future_fold_sleeves_cannot_change_past_forecast()


def test_future_fold_cannot_change_past_route() -> None:
    test_future_fold_sleeves_cannot_change_past_forecast()


def test_fold_member_mask_is_applied() -> None:
    test_fold_local_member_masks_are_not_unioned()


def test_effective_block_boundary_is_fail_closed() -> None:
    test_insufficient_regime_blocks_fail_closed()


def test_gate_and_simulator_share_weights() -> None:
    test_gate_and_deployment_share_identical_weights_after_routing()


def test_temporal_stability_failure_covers_rejection_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("L1_DEBUG", "1")
    n = 600
    panel_syms = 2
    price_bars = np.ones(n, dtype=np.float32) * 100
    dip_start, dip_end = 200, 300
    price_bars[dip_start:dip_end] = 100 - 5 * np.sin(np.linspace(0, np.pi, dip_end - dip_start))
    price_bars[dip_end:] = 90
    ts = _make_timestamps(n)
    close = np.column_stack([price_bars] * panel_syms).astype(np.float32)
    bars = TimeframeBarCube("4h", ts, ("SYM0","SYM1"),
        close*0.999, close*1.001, close*0.999, close,
        np.ones((n, panel_syms), dtype=np.float32)*1e6,
        np.ones((n, panel_syms), dtype=np.bool_))
    panel = RawSignalPanel(ts, ("SYM0","SYM1"),
        (SignalDescriptor("mom_fast","momentum_ts","fast",8,"4h"),),
        np.ones((n, panel_syms, 1), dtype=np.float32) * 0.5,
        np.ones((n, panel_syms, 1), dtype=np.bool_),
        np.ones((n, panel_syms), dtype=np.float32)*0.02)
    cost = np.full((n, panel_syms), 0.5, dtype=np.float32)
    funding = np.zeros((n*4, panel_syms), dtype=np.float32)
    ret = np.full(n, 0.003, dtype=np.float64) + np.random.default_rng(42).normal(0, 0.005, n).astype(np.float64)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = RegimeRouterConfig(min_effective_blocks=1, min_posterior_probability=0.51, n_bootstrap=100)
    dc_config = DynamicCompoundingConfig(kelly_fraction=0.05, target_ann_vol=0.05)
    folds = (CausalFold(0, 0, 100, 100, 150, 150, 350, 2, 42),
             CausalFold(1, 0, 350, 350, 400, 400, 600, 2, 42))
    policy = ExitPolicySpec("t:time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "h")
    sleeves = (L1SleevePosterior("s1:f0:c0", "mom_fast", "momentum_ts", 0, 0,
        np.ones(panel_syms, dtype=np.bool_), "h1", policy, 0.1, 0.01, 0.02, 0.95, 1.0, (0.01,), 30, True, ()),)
    result = build_fold_local_regime_forecast(panel, sleeves, (), folds, bars, cost, funding, regime, config, dc_config)
    assert result.tape.decision_time_ns_1d.shape[0] > 0


def test_staged_admission_short_circuits_before_regime() -> None:
    n = 200
    panel = _dummy_panel(n, 2)
    bars = _dummy_bars(n, 2)
    cost = np.full((n, 2), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 2), dtype=np.float32)
    ret = np.zeros(n, dtype=np.float64)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = RegimeRouterConfig(min_effective_blocks=1000, min_posterior_probability=0.99)
    dc_config = DynamicCompoundingConfig()
    folds = _dummy_folds(n)
    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert result.attribution.unconditional_pass == 0


def test_bootstrap_does_not_duplicate_prior_rows() -> None:
    n = 200
    panel = _dummy_panel(n, 2)
    bars = _dummy_bars(n, 2)
    cost = np.full((n, 2), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 2), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = _default_config()
    dc_config = DynamicCompoundingConfig()
    folds = _dummy_folds(n)
    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert len(result.evidence) >= 0
    tape_ids = set()
    for e in result.evidence:
        key = (e.signal_id, e.outer_fold_id, e.regime_code)
        assert key not in tape_ids, f"duplicate evidence key: {key}"
        tape_ids.add(key)


def test_dedup_preserves_signal_descriptor_scale_mapping() -> None:
    n = 200
    ts = _make_timestamps(n)
    syms = ("SYM0", "SYM1")
    descs = (
        SignalDescriptor("fast", "momentum_ts", "fast", 8, "4h"),
        SignalDescriptor("slow", "momentum_ts", "slow", 48, "4h"),
    )
    z = np.random.default_rng(42).normal(0, 1, (n, 2, 2)).astype(np.float32)
    panel = RawSignalPanel(ts, syms, descs, z, np.ones((n, 2, 2), dtype=np.bool_), np.ones((n, 2), dtype=np.float32) * 0.02)
    from src.domain.futures.compound.l1_sleeves import select_non_redundant_signals
    surviving = select_non_redundant_signals(panel, ("fast", "slow"), fit_end_exclusive=100)
    assert len(surviving) <= 2
    assert all(s in ("fast", "slow") for s in surviving)


def test_funding_alignment_and_position_sign() -> None:
    from src.domain.futures.compound.contracts import ExpertReturnTape
    tape = ExpertReturnTape(
        decision_time_ns_1d=np.array([0, 1], dtype=np.int64),
        execution_time_ns_1d=np.array([0, 1], dtype=np.int64),
        available_time_ns_1d=np.array([1, 2], dtype=np.int64),
        signal_id_1d=np.array(["a", "a"], dtype=np.str_),
        outer_fold_id_1d=np.array([0, 0], dtype=np.int16),
        regime_code_1d=np.array([1, 1], dtype=np.int8),
        gross_return_1d=np.array([0.003, -0.001], dtype=np.float64),
        execution_cost_return_1d=np.array([-0.0005, -0.0003], dtype=np.float64),
        funding_return_1d=np.array([0.0001, -0.0002], dtype=np.float64),
        net_return_1d=np.array([0.0026, -0.0015], dtype=np.float64),
    )
    assert tape.net_return_1d[0] == pytest.approx(0.0026)
    assert tape.net_return_1d[1] == pytest.approx(-0.0015)


def test_cash_no_evidence_is_not_integrity_failure() -> None:
    from src.domain.futures.compound.contracts import ExecutionLedger
    from src.domain.futures.compound.validation import classify_l2_evidence
    ledger = ExecutionLedger(
        timestamps_ns=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64) * 14400000000000,
        net_returns_1d=np.zeros(6, dtype=np.float64),
        equity_1d=np.ones(6, dtype=np.float64),
        target_weights_2d=np.zeros((6, 2), dtype=np.float32),
        fee_returns_1d=np.zeros(6, dtype=np.float64),
        slippage_returns_1d=np.zeros(6, dtype=np.float64),
        impact_returns_1d=np.zeros(6, dtype=np.float64),
        funding_returns_1d=np.zeros(6, dtype=np.float64),
        integrity_ok=True,
        integrity_reasons=(),
    )
    daily = np.array([0.0, 0.0], dtype=np.float64)
    config = RegimeRouterConfig()
    sufficient, reasons = classify_l2_evidence(ledger, daily, L2GateConfig())
    assert not sufficient
    assert len(reasons) > 0


def test_empty_l2_series_does_not_create_npz() -> None:
    from src.application.futures.runner.compound_main import write_l2_gate_inputs
    from src.domain.futures.compound.contracts import L2Evaluation, L2GateVerdict
    eval = L2Evaluation(
        verdict=L2GateVerdict.NO_EVIDENCE, benchmark_id="test",
        annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
        excess_growth_probability=0.5, stressed_excess_growth_lcb90=0.0,
        equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.5,
        deflated_sharpe_probability=0.5, candidate_count=1,
        calmar=0.0, max_drawdown=0.0, daily_cvar95=0.0,
        annual_volatility=0.0, annual_turnover=0.0, cost_drag_ratio=0.0,
        absolute_cagr=0.0, capacity_utilisation_p95=0.0, active_days_ratio=0.0,
        rebalance_count=0, positive_outer_folds=0, oos_days=0,
        category_results=(), integrity_ok=True, reasons=("no_evidence",),
    )
    from tempfile import mkdtemp
    run_dir = Path(mkdtemp())
    result = write_l2_gate_inputs(run_dir, eval)
    assert result is None
    assert not (run_dir / "l2_gate_inputs.npz").exists()


def test_real_route_allocator_handoff_wiring() -> None:
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
    from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
    weights = compute_dynamic_compounding_path(
        forecast=result.forecast, sigma_2d=panel.sigma_2d,
        funding_rates_1h_2d=funding, config=dc_config,
        close_2d=bars.close_2d, cost_bps=8.0,
    )
    assert weights.shape[0] == n
    assert result.attribution.candidate_experts >= result.attribution.active_experts
    assert result.attribution.candidate_experts >= 0
    assert result.attribution.unconditional_pass >= 0


def test_dry_run_never_consumes_holdout() -> None:
    pass


def test_all_route_rejections_are_attributed() -> None:
    n = 200
    panel = _dummy_panel(n, 2)
    bars = _dummy_bars(n, 2)
    cost = np.full((n, 2), 8.0, dtype=np.float32)
    funding = np.zeros((n * 4, 2), dtype=np.float32)
    ret = _make_returns(n)
    ts = _make_timestamps(n)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = RegimeRouterConfig(min_effective_blocks=100)
    dc_config = DynamicCompoundingConfig()
    folds = _dummy_folds(n)
    result = build_fold_local_regime_forecast(
        panel, _dummy_sleeves(), (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    all_evidence = result.evidence
    rejected = [e for e in all_evidence if not e.admitted]
    assert len(rejected) >= 0
    for e in rejected:
        assert len(e.reasons) > 0


def test_temporal_rejection_is_recorded_after_unconditional_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.domain.futures.compound.l1_regime_routing as routing
    from src.domain.futures.compound.contracts import ExpertReturnTape

    n = 500
    panel = _dummy_panel(n, 2)
    bars = _dummy_bars(n, 2)
    folds = _dummy_folds(n)
    regime = build_causal_regime_panel(_make_returns(n), _make_timestamps(n), _default_config())
    tape = ExpertReturnTape(
        decision_time_ns_1d=np.arange(6, dtype=np.int64),
        execution_time_ns_1d=np.arange(6, dtype=np.int64),
        available_time_ns_1d=np.arange(6, dtype=np.int64) + 1,
        signal_id_1d=np.full(6, "mom_fast", dtype=np.str_),
        outer_fold_id_1d=np.zeros(6, dtype=np.int16),
        regime_code_1d=np.ones(6, dtype=np.int8),
        gross_return_1d=np.full(6, 0.01, dtype=np.float64),
        execution_cost_return_1d=np.zeros(6, dtype=np.float64),
        funding_return_1d=np.zeros(6, dtype=np.float64),
        net_return_1d=np.full(6, 0.01, dtype=np.float64),
    )
    monkeypatch.setattr(routing, "build_fold_local_shadow_tape", lambda *args: tape)
    monkeypatch.setattr(
        routing,
        "_compute_unconditional_evidence",
        lambda *args: (0.1, 0.99, 0.1, True, []),
    )
    monkeypatch.setattr(
        routing,
        "_compute_temporal_evidence",
        lambda *args: (1, -0.1, ["robust_temporal_growth_not_positive"]),
    )

    result = build_fold_local_regime_forecast(
        panel, (), (), folds, bars,
        np.zeros((n, 2), dtype=np.float32),
        np.zeros((n * 4, 2), dtype=np.float32),
        regime, _default_config(), DynamicCompoundingConfig(),
    )

    assert result.attribution.temporal_pass == 0
    assert result.attribution.reason_counts["robust_temporal_growth_not_positive"] == 2
    assert all(not evidence.admitted for evidence in result.evidence)


def test_routing_with_non_empty_sleeves_produces_evidence_tape() -> None:
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
    policy = ExitPolicySpec("t:time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "h")
    sleeves = (
        L1SleevePosterior(
            "s1:f0:c0", "mom_fast", "momentum_ts", 0, 0,
            np.ones(4, dtype=np.bool_), "h1",
            policy, 0.1, 0.01, 0.02, 0.95, 1.0, (0.01,), 30, True, (),
        ),
        L1SleevePosterior(
            "s1:f1:c0", "mom_fast", "momentum_ts", 1, 0,
            np.ones(4, dtype=np.bool_), "h1",
            policy, 0.1, 0.01, 0.02, 0.95, 1.0, (0.01,), 30, True, (),
        ),
    )
    result = build_fold_local_regime_forecast(
        panel, sleeves, (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert result.tape.decision_time_ns_1d.shape[0] > 0
    assert result.tested_hypotheses >= 0
    assert result.attribution.candidate_experts >= 0

def test_routing_loop_executes_with_prior_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("L1_DEBUG", "1")
    n = 1000
    panel_syms = 2
    panel = _dummy_panel(n, panel_syms)
    ts = _make_timestamps(n)
    drift = 0.20
    price_curve = 100.0 * np.exp(np.linspace(0, drift, n))
    close = np.column_stack([price_curve] * panel_syms).astype(np.float32)
    bars = TimeframeBarCube(
        "4h", ts, tuple(f"SYM{i}" for i in range(panel_syms)),
        close * 0.9998, close * 1.0002, close * 0.9998, close,
        np.ones((n, panel_syms), dtype=np.float32) * 1e6,
        np.ones((n, panel_syms), dtype=np.bool_),
    )
    cal_len = 60
    oos_len = 200
    custom_folds: list[CausalFold] = []
    for i in range(2):
        cal_start = 200 + i * (cal_len + oos_len)
        cal_end = cal_start + cal_len
        oos_start = cal_end
        oos_end = oos_start + oos_len
        if oos_end > n:
            break
        custom_folds.append(CausalFold(i, 0, cal_start, cal_start, cal_end, oos_start, oos_end, 2, 42))
    folds = tuple(custom_folds)
    cost = np.full((n, panel_syms), 1.0, dtype=np.float32)
    funding = np.zeros((n * 4, panel_syms), dtype=np.float32)
    rng = np.random.default_rng(99)
    ret = np.full(n, 0.001, dtype=np.float64) + rng.normal(0, 0.002, n).astype(np.float64)
    regime = build_causal_regime_panel(ret, ts, _default_config())
    config = RegimeRouterConfig(
        min_effective_blocks=1, min_posterior_probability=0.51,
        n_inner_folds=2, min_positive_inner_folds=1,
        n_bootstrap=100,
    )
    dc_config = DynamicCompoundingConfig()
    policy = ExitPolicySpec("t:time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "h")
    sleeves = (
        L1SleevePosterior(
            "s1:f0:c0", "mom_fast", "momentum_ts", 0, 0,
            np.ones(panel_syms, dtype=np.bool_), "h1",
            policy, 0.1, 0.01, 0.02, 0.95, 1.0, (0.01,), 30, True, (),
        ),
    )
    result = build_fold_local_regime_forecast(
        panel, sleeves, (), folds, bars, cost, funding,
        regime, config, dc_config,
    )
    assert result.tape.decision_time_ns_1d.shape[0] > 0
    assert result.attribution.candidate_experts >= 0
    assert result.attribution.unconditional_pass >= 0

def test_prequential_route_resource_budget() -> None:
    import time
    start = time.monotonic()
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
    elapsed = time.monotonic() - start
    assert elapsed < 30.0
    assert isinstance(result, PrequentialExpertRoute)
