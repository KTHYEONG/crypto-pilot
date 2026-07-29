import pytest
import numpy as np

from src.domain.futures.compound.contracts import (
    CausalFold,
    FamilyEdgeScreen,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.config import DynamicCompoundingConfig, HandoffConfig
from src.domain.futures.compound.l1_screening import (
    compute_cross_sectional_ic,
    estimate_effective_independence,
    newey_west_tstat,
    screen_family_edge,
    screen_signal_edge,
)
from src.domain.futures.compound.l1_regime_routing import (
    decompose_expert_gross_contribution,
    blend_expert_contributions,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_panel(
    descriptors: tuple[SignalDescriptor, ...],
    n_bars: int = 400, n_syms: int = 20,
    rng: np.random.Generator | None = None,
) -> RawSignalPanel:
    if rng is None:
        rng = np.random.default_rng(42)
    z = rng.standard_normal((n_bars, n_syms, len(descriptors))).astype(np.float32)
    valid = np.isfinite(z)
    return RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        descriptors=descriptors,
        z_3d=z,
        valid_3d=valid,
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )


def _make_4h_bars(n_bars: int = 400, n_syms: int = 20) -> TimeframeBarCube:
    close = np.cumprod(1 + np.random.default_rng(42).standard_normal((n_bars, n_syms)) * 0.01, axis=0).astype(np.float32)
    close = np.maximum(close, 1.0)
    return TimeframeBarCube(
        timeframe="4h",
        timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=close.copy(),
        high_2d=close.copy() * 1.01,
        low_2d=close.copy() * 0.99,
        close_2d=close,
        quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )


# ── Test 1: Happy path – reversal family with declared_orientation=-1 ────

def test_screen_family_edge_admits_declared_reversal() -> None:
    rng = np.random.default_rng(0)
    n_bars, n_syms = 400, 20
    noise = rng.standard_normal((n_bars, n_syms)) * 0.002
    trend = np.linspace(-0.5, 0.5, n_syms) * 0.01
    forward_ret = np.zeros((n_bars, n_syms))
    for t in range(1, n_bars):
        forward_ret[t] = trend + rng.standard_normal(n_syms) * 0.008
    z_rev = -forward_ret + noise
    z_mom = forward_ret + noise
    desc = (
        SignalDescriptor("rev:fast", "xs_reversal", "fast", 24, "4h",
                         target_horizon_hours=24, declared_orientation=-1),
        SignalDescriptor("mom:fast", "momentum_ts", "fast", 24, "4h",
                         target_horizon_hours=24, declared_orientation=1),
    )
    z_3d = np.stack([z_rev, z_mom], axis=2).astype(np.float32)
    price = np.cumprod(1 + forward_ret, axis=0).astype(np.float32)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        descriptors=desc,
        z_3d=z_3d,
        valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    bars_4h = TimeframeBarCube(
        timeframe="4h",
        timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(),
        high_2d=price.copy() * 1.01,
        low_2d=price.copy() * 0.99,
        close_2d=price,
        quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 200, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    result = screen_family_edge(panel, bars_4h, folds, cfg)
    assert isinstance(result, FamilyEdgeScreen)
    for rec in result.records:
        if rec.family == "xs_reversal":
            assert rec.admitted, f"xs_reversal should be admitted: mean_ic={rec.mean_ic:.4f} t={rec.t_newey_west:.3f} {rec.reasons}"


# ── Test 2: Reject contradicted orientation ────────────────────────────

def test_screen_family_edge_rejects_contradicted_orientation() -> None:
    rng = np.random.default_rng(1)
    n_bars, n_syms = 400, 20
    base = rng.standard_normal((n_bars, n_syms))
    z_neg = (-0.04 * base + 0.1 * rng.standard_normal((n_bars, n_syms))).astype(np.float32)
    z_pos = (+0.04 * base + 0.1 * rng.standard_normal((n_bars, n_syms))).astype(np.float32)

    desc = (
        SignalDescriptor("mom:fast", "momentum_ts", "fast", 24, "4h",
                         target_horizon_hours=24, declared_orientation=1),
    )
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        descriptors=desc,
        z_3d=z_pos.reshape(n_bars, n_syms, 1),
        valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    bars_4h = _make_4h_bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 200, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    result = screen_family_edge(panel, bars_4h, folds, cfg)
    assert len(result.records) == 1
    rec = result.records[0]
    assert not rec.admitted


# ── Test 3: Šidák uses n_eff, not family count ─────────────────────────

def test_sidak_uses_effective_independence_not_family_count() -> None:
    rng = np.random.default_rng(2)
    n_bars, n_syms = 400, 20
    base = rng.standard_normal((n_bars, n_syms))
    descs: list[SignalDescriptor] = []
    for i in range(6):
        descs.append(SignalDescriptor(
            f"sig_{i}", "test_family", "fast", 24, "4h",
            target_horizon_hours=24, declared_orientation=1,
        ))
    z_3d = np.stack([base + 0.1 * rng.standard_normal((n_bars, n_syms)) for _ in range(6)], axis=2).astype(np.float32)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        descriptors=tuple(descs),
        z_3d=z_3d,
        valid_3d=np.ones((n_bars, n_syms, 6), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    bars_4h = _make_4h_bars(n_bars, n_syms)
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 200, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    result = screen_family_edge(panel, bars_4h, folds, cfg)
    assert 1.0 < result.n_effective_independent < 6.0, (
        f"n_eff={result.n_effective_independent} must reflect the real correlation "
        "structure (strictly between 1.0 and 6.0), not collapse to the no-data escape hatch"
    )
    assert len(result.records) == 1
    rec = result.records[0]


# ── Test 4: decompose_expert_gross is bounded and exact ────────────────

def test_decompose_expert_gross_is_bounded_and_exact() -> None:
    t_total, n_syms, n_experts = 10, 5, 2
    weights_2d = np.full((t_total, n_syms), 0.02, dtype=np.float64)
    contribution_3d = np.zeros((n_experts, t_total, n_syms), dtype=np.float64)
    contribution_3d[0, :, :] = 0.6
    contribution_3d[1, :, :] = 0.4
    log_ret_2d = np.full((t_total, n_syms), 0.005, dtype=np.float64)
    log_ret_2d[0, :] = 0.0

    gross_e = decompose_expert_gross_contribution(weights_2d, contribution_3d, log_ret_2d)
    assert gross_e.shape == (n_experts, t_total)
    assert np.all(np.isfinite(gross_e))

    all_zero_contrib = np.zeros_like(contribution_3d)
    gross_zero = decompose_expert_gross_contribution(weights_2d, all_zero_contrib, log_ret_2d)
    assert np.allclose(gross_zero, 0.0)


# ── Test 5: blend_expert_contributions is NaN-safe ─────────────────────

def test_blend_expert_contributions_nan_safe() -> None:
    rng = np.random.default_rng(4)
    t_total, n_syms, n_sigs = 50, 10, 3
    z = rng.standard_normal((t_total, n_syms, n_sigs)).astype(np.float32)
    valid = np.ones((t_total, n_syms, n_sigs), dtype=np.bool_)
    z[0, 0, 0] = np.nan
    valid[0, 0, 0] = False
    z[::5, ::3, 1] = np.nan
    valid[::5, ::3, 1] = False

    weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    mu = blend_expert_contributions(z, valid, weights)
    assert mu.shape == (t_total, n_syms)
    assert np.all(np.isfinite(mu)), "NaN leaked into blend output"

    expected = np.zeros((t_total, n_syms), dtype=np.float64)
    for t in range(t_total):
        for s in range(n_syms):
            num = 0.0
            den = 0.0
            for k in range(n_sigs):
                if valid[t, s, k]:
                    num += weights[k] * z[t, s, k]
                    den += abs(weights[k])
            expected[t, s] = num / den if den > 0 else 0.0
    assert np.allclose(mu, expected, atol=1e-6)

    with pytest.raises(ValueError, match="weights"):
        blend_expert_contributions(z, valid, np.array([0.5, 0.3]))


# ── Test 6: NW t matches OLS at lag 0 ─────────────────────────────────

def test_newey_west_tstat_matches_iid_ols_at_lag_zero() -> None:
    rng = np.random.default_rng(5)
    n = 200
    data = rng.standard_normal(n)
    t_nw, se_nw = newey_west_tstat(data, max_lag=0)
    t_ols = float(np.mean(data)) / (float(np.std(data, ddof=1)) / np.sqrt(n))
    assert np.isclose(t_nw, t_ols, rtol=5e-2), f"NW t={t_nw} != OLS t={t_ols}"

    t_short, se_short = newey_west_tstat(np.array([1.0, 2.0]), 0)
    assert t_short == 0.0
    assert se_short == 0.0


# ── Test 7: IC has no look-ahead ──────────────────────────────────────

def test_compute_cross_sectional_ic_no_lookahead() -> None:
    rng = np.random.default_rng(6)
    n_bars, n_syms = 100, 20
    z = rng.standard_normal((n_bars, n_syms)).astype(np.float32)
    fwd = np.zeros((n_bars, n_syms), dtype=np.float64)
    for t in range(n_bars - 6):
        fwd[t] = rng.standard_normal(n_syms) * 0.01
    oos_slices = (slice(10, 90),)
    ic = compute_cross_sectional_ic(z, fwd, oos_slices, min_cross_section=8)
    assert ic.shape == (n_bars,)
    assert np.all(np.isnan(ic[:10]))
    assert np.all(np.isnan(ic[90:]))
    n_finite = int(np.sum(np.isfinite(ic)))
    assert n_finite >= 70, f"only {n_finite} finite IC bars"


# ── Test 8: declared_orientation rejects zero ─────────────────────────

def test_declared_orientation_rejects_zero() -> None:
    with pytest.raises(ValueError, match="declared_orientation must be -1 or 1"):
        SignalDescriptor("bad", "test", "fast", 24, "4h", declared_orientation=0)


# ── Test 9: fold 0 no longer skipped; evidence dictionary accumulates ──

def test_prequential_route_accumulates_from_fold_zero() -> None:
    from src.domain.futures.compound.l1_regime_routing import (
        _build_prequential_expert_route_impl,
        build_causal_regime_panel,
    )
    from src.domain.futures.compound.config import DynamicCompoundingConfig, RegimeRouterConfig
    from src.domain.futures.compound.contracts import L1SleevePosterior
    rng = np.random.default_rng(7)
    n_bars, n_syms = 400, 10
    desc = (SignalDescriptor("test:fast", "test", "fast", 24, "4h",
                             target_horizon_hours=24, declared_orientation=1),)
    panel = _make_panel(desc, n_bars, n_syms, rng)
    bars_4h = _make_4h_bars(n_bars, n_syms)
    folds = tuple(CausalFold(i, i * 50, (i + 1) * 50, i * 50, (i + 1) * 50, (i + 1) * 50, (i + 2) * 50, 25, 1)
                  for i in range(4))

    close = bars_4h.close_2d.astype(np.float64)
    log_ret = np.zeros(close.shape[0], dtype=np.float64)
    if n_syms >= 2:
        prev = close[:-1, :2]
        curr = close[1:, :2]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        w = np.array([0.5, 0.5], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted = np.where(mask, np.log(curr / prev), 0.0) @ w
        log_ret[1:] = weighted
    regime_panel = build_causal_regime_panel(log_ret, bars_4h.timestamps_ns, RegimeRouterConfig())

    sleeves_per_fold = tuple(
        L1SleevePosterior(
            sleeve_id=f"test:fast:{i}:0", signal_id="test:fast", family="test",
            outer_fold_id=i, cluster_id=0,
            member_mask_1d=np.ones(n_syms, dtype=np.bool_),
            member_hash="h", exit_policy=None,
            fitted_beta=0.02, mean_net_return=0.001, standard_error=0.01,
            posterior_positive_probability=0.55, residual_novelty=0.1,
            fold_net_returns=(0.001,), effective_events=50,
            admitted=True, reasons=(),
        ) for i in range(4)
    )
    rr_config = RegimeRouterConfig(
        min_evidence_bars=5, min_posterior_probability=0.51,
        min_effective_blocks=1,
    )
    dc_config = DynamicCompoundingConfig()
    cost_bps_4h = np.full((n_bars, n_syms), 8.0, dtype=np.float32)
    funding_1h = np.zeros((n_bars * 4, n_syms), dtype=np.float32)

    result = _build_prequential_expert_route_impl(
        panel, sleeves_per_fold, folds, bars_4h, cost_bps_4h, funding_1h,
        regime_panel, rr_config, dc_config, cost_bps=8.0,
    )
    for ev in result.evidence:
        if ev.outer_fold_id > 0:
            assert ev.n_evidence_bars > 0, f"fold {ev.outer_fold_id}: evidence should accumulate"


# ── Test 3b: n_eff must not collapse when only OOS bars are populated ──

def test_estimate_effective_independence_ignores_nan_padded_rows() -> None:
    """Regression: compute_cross_sectional_ic seeds the FULL bar range with NaN
    and only fills bars inside oos_slices. A naive column-wise all-finite check
    over the whole array always fails in that shape and silently collapses
    n_eff to 1.0 (defeating the Sidak correction). Rows with no data anywhere
    must be dropped before the column check."""
    rng = np.random.default_rng(9)
    n_bars, n_oos, n_signals = 400, 100, 6
    base = rng.standard_normal(n_oos)
    ic_matrix = np.full((n_bars, n_signals), np.nan, dtype=np.float64)
    for i in range(n_signals):
        ic_matrix[150:150 + n_oos, i] = base + 0.1 * rng.standard_normal(n_oos)

    n_eff = estimate_effective_independence(ic_matrix)
    assert 1.0 < n_eff < n_signals, (
        f"n_eff={n_eff} should reflect correlated-but-not-identical signals, "
        "not collapse to 1.0 just because most bars are NaN outside the OOS window"
    )

    independent = np.full((n_bars, n_signals), np.nan, dtype=np.float64)
    for i in range(n_signals):
        independent[150:150 + n_oos, i] = rng.standard_normal(n_oos)
    n_eff_indep = estimate_effective_independence(independent)
    assert n_eff_indep > n_eff, "independent signals must yield a higher n_eff than correlated ones"


# ── Test 10: Family screen integrates with loop logic ─────────────────

def test_screen_family_edge_empty_folds() -> None:
    desc = (SignalDescriptor("a:fast", "a", "fast", 24, "4h", declared_orientation=1),)
    panel = _make_panel(desc, 100, 10)
    bars_4h = _make_4h_bars(100, 10)
    cfg = HandoffConfig()
    result = screen_family_edge(panel, bars_4h, (), cfg)
    assert isinstance(result, FamilyEdgeScreen)
    assert len(result.records) == 0
    assert len(result.admitted_families) == 0


# ── P1: screen_signal_edge (docs/specs/l1_cash_only_exit_redesign.md) ──────


def test_screen_signal_edge_no_tstat_inflation() -> None:
    """[RULE-P1-1] Duplicating a real-IC signal must not inflate its t-stat."""
    rng = np.random.default_rng(0)
    n_bars, n_syms = 400, 20
    cs_signal = np.tile(np.linspace(-0.5, 0.5, n_syms), (n_bars, 1))
    forward_ret = np.zeros((n_bars, n_syms))
    for t in range(1, n_bars):
        forward_ret[t] = cs_signal[t] * 0.04 + rng.standard_normal(n_syms) * 0.005
    z_rev = -cs_signal * 0.5 + rng.standard_normal((n_bars, n_syms)) * 0.01
    price = np.cumprod(1 + forward_ret, axis=0).astype(np.float32)
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 200, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    desc_single = (SignalDescriptor("rev:fast", "xs_reversal", "fast", 24, "4h",
                                     target_horizon_hours=24, declared_orientation=-1),)
    panel_single = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc_single,
        z_3d=z_rev[:, :, None].astype(np.float32),
        valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    screen_single = screen_signal_edge(
        panel_single, bars_4h, folds, cfg,
        funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    t_single = screen_single.records[0].t_newey_west

    desc_5x = tuple(
        SignalDescriptor(f"rev:fast_{i}", "xs_reversal", "fast", 24, "4h",
                          target_horizon_hours=24, declared_orientation=-1)
        for i in range(5)
    )
    panel_5x = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc_5x,
        z_3d=np.repeat(z_rev[:, :, None], 5, axis=2).astype(np.float32),
        valid_3d=np.ones((n_bars, n_syms, 5), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    screen_5x = screen_signal_edge(
        panel_5x, bars_4h, folds, cfg,
        funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    assert len(screen_5x.records) == 5
    assert t_single != 0.0
    for rec in screen_5x.records:
        assert abs(rec.t_newey_west - t_single) < 1e-9, (
            f"duplicated signal inflated t-stat: single={t_single} dup={rec.t_newey_west}"
        )


def test_screen_signal_edge_term_structure_not_cancelled() -> None:
    """[RULE-P1-1] fast(-IC) + slow(+IC) inside one family must not average to zero."""
    rng = np.random.default_rng(1)
    n_bars, n_syms = 400, 20
    cs_trend = np.tile(np.linspace(-0.5, 0.5, n_syms), (n_bars, 1))
    forward_ret = np.zeros((n_bars, n_syms))
    for t in range(1, n_bars):
        forward_ret[t] = cs_trend[t] * 0.03 + rng.standard_normal(n_syms) * 0.005
    noise = rng.standard_normal((n_bars, n_syms)) * 0.01
    z_fast_reversal = -cs_trend * 0.5 + noise       # real negative-oriented edge
    z_slow_momentum = cs_trend * 0.5 + noise         # real positive-oriented edge
    price = np.cumprod(1 + forward_ret, axis=0).astype(np.float32)
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 200, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    desc = (
        SignalDescriptor("fast:rev", "term_structure", "fast", 24, "4h",
                          target_horizon_hours=24, declared_orientation=-1),
        SignalDescriptor("slow:mom", "term_structure", "slow", 24, "4h",
                          target_horizon_hours=24, declared_orientation=1),
    )
    z_3d = np.stack([z_fast_reversal, z_slow_momentum], axis=2).astype(np.float32)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc,
        z_3d=z_3d, valid_3d=np.ones((n_bars, n_syms, 2), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )

    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    screen = screen_signal_edge(
        panel, bars_4h, folds, cfg,
        funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    by_id = {r.signal_id: r for r in screen.records}
    # both real, opposite-signed edges must survive signal-level screening --
    # family-level pooling would average them toward zero and admit neither.
    assert by_id["fast:rev"].admitted, by_id["fast:rev"].reasons
    assert by_id["slow:mom"].admitted, by_id["slow:mom"].reasons
    assert by_id["fast:rev"].t_newey_west < 0
    assert by_id["slow:mom"].t_newey_west > 0


# ── P0/P1: book cost + net edge screen (docs/specs/l1_book_cost_accounting_and_net_edge_screen.md) ──


def test_replay_signal_standalone_book_turnover_and_errors() -> None:
    """Scenario 5: constant z → turnover≈0; sign-flipping z → turnover>0; non-finite → ValueError."""
    from src.domain.futures.compound.l1_screening import replay_signal_standalone_book
    from src.domain.futures.compound.config import DynamicCompoundingConfig

    n_bars, n_syms = 50, 5
    panel = _make_panel((), n_bars, n_syms)
    bars = _make_4h_bars(n_bars, n_syms)
    cfg = DynamicCompoundingConfig()
    oos_slices = (slice(10, 40),)
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)

    const_z = np.full((n_bars, n_syms), 0.5, dtype=np.float32)
    net_1, turn_1 = replay_signal_standalone_book(
        const_z, panel, bars, funding, oos_slices, cfg, 8.0, declared_orientation=1,
    )
    assert turn_1 < 1e-10, f"constant z should have ~0 turnover, got {turn_1}"

    flip_z = np.ones((n_bars, n_syms), dtype=np.float32)
    flip_z[25:] = -1.0
    net_2, turn_2 = replay_signal_standalone_book(
        flip_z, panel, bars, funding, oos_slices, cfg, 8.0, declared_orientation=1,
    )
    assert turn_2 > 0.0, f"flipping z should produce turnover, got {turn_2}"
    assert len(net_2) == 30

    assert len(net_1) == 30


def test_screen_signal_net_edge_thresholds_and_sample_guard() -> None:
    """Scenario 6: all-positive net → passes=True; all-negative → passes=False; insufficient → (False,0,0)."""
    from src.domain.futures.compound.l1_screening import screen_signal_net_edge

    cfg = HandoffConfig(min_family_ic_samples=10, min_growth_posterior_probability=0.51)

    pos_net = np.full(100, 0.001, dtype=np.float64)
    passes, prob, ann = screen_signal_net_edge(pos_net, cfg)
    assert passes, f"positive net should pass: prob={prob}, ann={ann}"
    assert prob > 0.5
    assert ann > 0.0

    neg_net = np.full(100, -0.001, dtype=np.float64)
    passes2, prob2, ann2 = screen_signal_net_edge(neg_net, cfg)
    assert not passes2, "negative net should fail"

    small_net = np.zeros(5, dtype=np.float64)
    passes3, prob3, ann3 = screen_signal_net_edge(small_net, cfg)
    assert not passes3
    assert prob3 == 0.0
    assert ann3 == 0.0


def test_screen_signal_edge_gate_order_short_circuits_replay() -> None:
    """[RULE-P1-3] IC-failing signal has single reason and replay is not executed (allocator spy call count 0)."""
    rng = np.random.default_rng(0)
    n_bars, n_syms = 100, 10
    z_noise = rng.standard_normal((n_bars, n_syms)).astype(np.float32)
    price = np.cumprod(1 + rng.normal(0, 0.005, (n_bars, n_syms)), axis=0).astype(np.float32)
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 20, 0, 20, 20, 40, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=30)

    desc = (SignalDescriptor("noise", "noise_fam", "fast", 24, "4h",
                              target_horizon_hours=24, declared_orientation=1),)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc,
        z_3d=z_noise[:, :, None], valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )

    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    screen = screen_signal_edge(
        panel, bars_4h, folds, cfg,
        funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    assert len(screen.records) == 1
    rec = screen.records[0]
    assert not rec.admitted
    assert rec.reasons == ("insufficient_ic_samples",)


def test_screen_signal_edge_rejects_high_turnover_negative_net_signal() -> None:
    """Scenario 8: strong IC + high turnover signal → admitted=False, net_edge reason."""
    rng = np.random.default_rng(6)
    n_bars, n_syms = 600, 30
    cs_rank = np.tile(np.linspace(-1, 1, n_syms), (n_bars, 1))
    signal = np.zeros((n_bars, n_syms), dtype=np.float32)
    for t in range(n_bars):
        bar_sign = 1.0 if rng.random() > 0.5 else -1.0
        signal[t] = (bar_sign * cs_rank[t]).astype(np.float32)
    ret_edge = 0.0
    fwd_ret = np.zeros((n_bars, n_syms))
    for t in range(1, n_bars - 6):
        fwd_ret[t] = signal[t - 1] * 0.001 + rng.normal(0, 0.008, n_syms)
    price = np.cumprod(1 + fwd_ret, axis=0).astype(np.float32)
    price = np.maximum(price, 0.1)
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 150, 0, 150, 150, 450, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10, family_screen_alpha=0.05)

    desc = (SignalDescriptor("high_turn", "xs_reversal", "fast", 24, "4h",
                              target_horizon_hours=24, declared_orientation=1),)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc,
        z_3d=signal[:, :, None],
        valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )

    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    screen = screen_signal_edge(
        panel, bars_4h, folds, cfg,
        funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    assert len(screen.records) == 1
    rec = screen.records[0]
    assert not rec.admitted, f"deficit signal should be rejected: reasons={rec.reasons} turn={rec.intrinsic_turnover_per_bar} net_ann={rec.net_growth_ann}"
    assert any("net_edge" in r for r in rec.reasons), rec.reasons


def test_screen_signal_edge_propagates_net_edge_fields_to_diagnostic_recorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """Regression: SignalEdgeRecord's net-edge fields (turnover/net_ann/net_prob/edge_per_turn)
    must reach L1AdmissionRecorder.record_family_screen verbatim, not silently default to 0.0.
    Caught by a real production run where the JSONL diagnostic showed 0.0 for signals with
    strong, correctly-computed non-zero turnover/net_ann in the returned SignalEdgeRecord."""
    monkeypatch.setenv("L1_DEBUG", "1")
    (tmp_path / "logs").mkdir()
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "logs" / "l1_admission.jsonl"

    rng = np.random.default_rng(3)
    n_bars, n_syms = 400, 20
    desc = (SignalDescriptor("strong", "xs_reversal", "fast", 24, "4h",
                              target_horizon_hours=24, declared_orientation=1),)
    z = rng.standard_normal((n_bars, n_syms, 1)).astype(np.float32)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc,
        z_3d=z, valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    price = np.cumprod(1 + rng.normal(0, 0.005, (n_bars, n_syms)), axis=0).astype(np.float32)
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 300, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10)
    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)

    screen = screen_signal_edge(
        panel, bars_4h, folds, cfg, funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    rec = screen.records[0]

    import json
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == 1
    logged = lines[0]
    assert logged["intrinsic_turnover_per_bar"] == round(rec.intrinsic_turnover_per_bar, 6)
    assert logged["net_growth_ann"] == round(rec.net_growth_ann, 6)
    assert logged["net_growth_probability"] == round(rec.net_growth_probability, 4)
    assert logged["edge_per_turnover_bps"] == round(rec.edge_per_turnover_bps, 4)


def test_screen_signal_edge_net_edge_replay_failure_is_fail_closed(mocker) -> None:
    """[except ValueError] replay_signal_standalone_book raising must reject the signal with
    reasons=('net_edge_replay_failed',) and zeroed net-edge fields, not propagate the exception."""
    rng = np.random.default_rng(11)
    n_bars, n_syms = 600, 30
    cs_rank = np.tile(np.linspace(-1, 1, n_syms), (n_bars, 1))
    signal = np.zeros((n_bars, n_syms), dtype=np.float32)
    for t in range(n_bars):
        signal[t] = cs_rank[t].astype(np.float32)
    fwd_ret = np.zeros((n_bars, n_syms))
    for t in range(1, n_bars - 6):
        fwd_ret[t] = signal[t - 1] * 0.001 + rng.normal(0, 0.008, n_syms)
    price = np.cumprod(1 + fwd_ret, axis=0).astype(np.float32)
    price = np.maximum(price, 0.1)
    desc = (SignalDescriptor("strong", "xs_reversal", "fast", 24, "4h",
                              target_horizon_hours=24, declared_orientation=1),)
    panel = RawSignalPanel(
        decision_timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)), descriptors=desc,
        z_3d=signal[:, :, None], valid_3d=np.ones((n_bars, n_syms, 1), dtype=np.bool_),
        sigma_2d=np.full((n_bars, n_syms), 0.02, dtype=np.float32),
    )
    bars_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=np.arange(n_bars, dtype=np.int64),
        symbols=tuple(f"SYM_{i}" for i in range(n_syms)),
        open_2d=price.copy(), high_2d=price.copy() * 1.01, low_2d=price.copy() * 0.99,
        close_2d=price, quote_volume_2d=np.full((n_bars, n_syms), 1e8, dtype=np.float32),
        complete_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
    )
    folds = (CausalFold(0, 0, 100, 0, 100, 100, 300, 25, 1),)
    cfg = HandoffConfig(min_family_ic_samples=10)
    dc_cfg = DynamicCompoundingConfig()
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)

    mocker.patch(
        "src.domain.futures.compound.l1_screening.replay_signal_standalone_book",
        side_effect=ValueError("non_finite_standalone_book"),
    )

    screen = screen_signal_edge(
        panel, bars_4h, folds, cfg, funding_1h_2d=funding, allocator_config=dc_cfg,
    )
    assert len(screen.records) == 1
    rec = screen.records[0]
    assert not rec.admitted
    assert rec.reasons == ("net_edge_replay_failed",)
    assert rec.intrinsic_turnover_per_bar == 0.0
    assert rec.net_growth_ann == 0.0
    assert rec.net_growth_probability == 0.0
    assert rec.edge_per_turnover_bps == 0.0
    assert rec.signal_id not in screen.admitted_signal_ids
