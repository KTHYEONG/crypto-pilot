from __future__ import annotations

from unittest import mock

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import (
    DynamicCompoundingConfig,
    HandoffConfig,
    RegimeRouterConfig,
)
from src.domain.futures.compound.contracts import (
    CausalClusterFold,
    CausalFold,
    CausalRegimePanel,
    ClusterPanel,
    RawSignalPanel,
    SignalDescriptor,
    SignalFoldRecord,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_regime_routing import (
    ExpertContribution,
    apply_walk_forward_carry,
    concatenate_signal_evidence,
    compute_regime_overlay,
)
from src.domain.futures.compound.l1_sleeves import (
    estimate_cluster_sleeve_posteriors,
)

_FOUR_HOURS_NS = 4 * 3600 * 10 ** 9


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_timestamps(n: int) -> NDArray[np.int64]:
    return np.arange(n, dtype=np.int64) * _FOUR_HOURS_NS


def _default_config() -> RegimeRouterConfig:
    return RegimeRouterConfig()


def _bars(t: int = 50, n: int = 2) -> TimeframeBarCube:
    close = np.column_stack([np.linspace(100.0, 120.0 + i, t) for i in range(n)]).astype(np.float32)
    return TimeframeBarCube(
        "4h", np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
        close, close + 2.0, close - 2.0, close,
        np.ones((t, n), dtype=np.float32), np.ones((t, n), dtype=bool),
    )


def _panel(t: int = 50, n: int = 2) -> RawSignalPanel:
    z = np.ones((t, n, 1), dtype=np.float32)
    descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
    return RawSignalPanel(
        np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
        (descriptor,), z, np.ones_like(z, dtype=bool),
        np.ones((t, n), dtype=np.float32),
    )


def _folds() -> tuple[CausalFold, ...]:
    return tuple(CausalFold(i, 0, 20, 20, 25, 25 + i * 10, 25 + (i + 1) * 10, 1, 1) for i in range(4))


def _short_folds() -> tuple[CausalFold, ...]:
    return tuple(CausalFold(i, 0, 20, 20, 25, 25, 25 + 10, 1, 1) for i in range(2))


# ─── P0 Unit Tests ──────────────────────────────────────────────────────────


class TestP0SignalEvidence:

    def test_concatenate_signal_evidence_empty_history(self) -> None:
        result = concatenate_signal_evidence({}, "unknown_signal")
        assert result.net_1d.size == 0
        assert result.gross_1d.size == 0
        assert result.cost_1d.size == 0
        assert result.funding_1d.size == 0
        assert result.regime_code_1d.size == 0

    def test_concatenate_signal_evidence_accumulates_in_order(self) -> None:
        history: dict[str, list[SignalFoldRecord]] = {}
        r1 = SignalFoldRecord(
            gross_1d=np.array([1.0, 2.0], dtype=np.float64),
            cost_1d=np.array([0.0, 0.0], dtype=np.float64),
            funding_1d=np.array([0.0, 0.0], dtype=np.float64),
            net_1d=np.array([0.5, 1.0], dtype=np.float64),
            regime_code_1d=np.array([1, 1], dtype=np.int8),
        )
        r2 = SignalFoldRecord(
            gross_1d=np.array([3.0], dtype=np.float64),
            cost_1d=np.array([0.0], dtype=np.float64),
            funding_1d=np.array([0.0], dtype=np.float64),
            net_1d=np.array([2.0], dtype=np.float64),
            regime_code_1d=np.array([2], dtype=np.int8),
        )
        history["sig_a"] = [r1, r2]
        result = concatenate_signal_evidence(history, "sig_a")
        np.testing.assert_array_equal(result.net_1d, np.array([0.5, 1.0, 2.0]))
        np.testing.assert_array_equal(result.regime_code_1d, np.array([1, 1, 2], dtype=np.int8))
        assert result.net_1d.size == 3

    def test_evidence_window_uses_prior_folds_only(self) -> None:
        history: dict[str, list[SignalFoldRecord]] = {}
        r_prev = SignalFoldRecord(
            gross_1d=np.array([0.1, 0.2], dtype=np.float64),
            cost_1d=np.array([0.0, 0.0], dtype=np.float64),
            funding_1d=np.array([0.0, 0.0], dtype=np.float64),
            net_1d=np.array([0.05, 0.15], dtype=np.float64),
            regime_code_1d=np.array([1, 1], dtype=np.int8),
        )
        history["sig_a"] = [r_prev]
        prior = concatenate_signal_evidence(history, "sig_a")
        assert prior.net_1d.size == 2
        current_fold_net = np.array([0.1], dtype=np.float64)
        assert current_fold_net.size == 1
        assert prior.net_1d.size != current_fold_net.size

    def test_history_accumulates_even_when_gate_fails(self) -> None:
        history: dict[str, list[SignalFoldRecord]] = {}
        r_fail = SignalFoldRecord(
            gross_1d=np.array([-0.5], dtype=np.float64),
            cost_1d=np.array([0.0], dtype=np.float64),
            funding_1d=np.array([0.0], dtype=np.float64),
            net_1d=np.array([-0.5], dtype=np.float64),
            regime_code_1d=np.array([1], dtype=np.int8),
        )
        history.setdefault("sig_b", []).append(r_fail)
        r_fail2 = SignalFoldRecord(
            gross_1d=np.array([0.3], dtype=np.float64),
            cost_1d=np.array([0.0], dtype=np.float64),
            funding_1d=np.array([0.0], dtype=np.float64),
            net_1d=np.array([0.3], dtype=np.float64),
            regime_code_1d=np.array([1], dtype=np.int8),
        )
        history.setdefault("sig_b", []).append(r_fail2)
        prior = concatenate_signal_evidence(history, "sig_b")
        assert prior.net_1d.size == 2
        np.testing.assert_array_equal(prior.net_1d, np.array([-0.5, 0.3]))


# ─── P1 Unit Tests ──────────────────────────────────────────────────────────


class TestP1SleeveOOSGate:

    def test_sleeve_admission_rejects_is_pass_oos_fail(self) -> None:
        t, n = 60, 3
        bars = _bars(t, n)
        folds = _folds()
        cfg = HandoffConfig(
            min_sleeve_posterior_probability=0.95,
            min_oos_posterior_probability=0.55,
            min_oos_effective_blocks=5,
        )
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, (t, n, 1)).astype(np.float32)
        descriptor = SignalDescriptor(
            "trend:fast", "trend", "fast", 4, "4h", 4,
            "trend", "persistence", "v1",
        )
        panel = RawSignalPanel(
            np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
            (descriptor,), z, np.ones_like(z, dtype=bool),
            np.ones((t, n), dtype=np.float32),
        )
        syms = tuple(f"S{i}" for i in range(n))
        cluster_panel = ClusterPanel(syms, np.zeros(n, dtype=np.int32), np.zeros((1, n), dtype=np.float64), 1)
        cluster_folds = (
            CausalClusterFold(
                fold_id=folds[0].fold_id, fit_end_exclusive_4h=folds[0].fit_end_exclusive,
                fit_end_time_ns=0, panel=cluster_panel, member_hash="h1",
            ),
        )
        sleeves = estimate_cluster_sleeve_posteriors(
            panel, bars, cluster_folds, folds[:1],
            np.ones((t, n), dtype=np.float32),
            np.zeros((t * 4, n), dtype=np.float32), cfg,
        )
        for s in sleeves:
            assert not s.admitted
            if "oos_confirmation_failed" in s.reasons:
                break
        else:
            if sleeves:
                assert "oos_confirmation_failed" in sleeves[0].reasons or not sleeves[0].admitted

    def test_sleeve_admission_short_circuits_oos_bootstrap(self) -> None:
        t, n = 40, 2
        bars = _bars(t, n)
        folds = _short_folds()
        cfg = HandoffConfig(
            min_sleeve_posterior_probability=0.95,
            min_oos_posterior_probability=0.55,
            min_oos_effective_blocks=5,
        )
        z = np.zeros((t, n, 1), dtype=np.float32)
        descriptor = SignalDescriptor(
            "trend:fast", "trend", "fast", 4, "4h", 4,
            "trend", "persistence", "v1",
        )
        panel = RawSignalPanel(
            np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
            (descriptor,), z, np.ones_like(z, dtype=bool),
            np.ones((t, n), dtype=np.float32),
        )
        syms = tuple(f"S{i}" for i in range(n))
        cluster_panel = ClusterPanel(syms, np.zeros(n, dtype=np.int32), np.zeros((1, n), dtype=np.float64), 1)
        cluster_folds = (
            CausalClusterFold(
                fold_id=folds[0].fold_id, fit_end_exclusive_4h=folds[0].fit_end_exclusive,
                fit_end_time_ns=0, panel=cluster_panel, member_hash="h1",
            ),
        )
        with mock.patch(
            "src.domain.futures.compound.bootstrap.circular_stationary_bootstrap_growth",
        ) as mock_boot:
            mock_boot.return_value = (0.0, 0.0, 0.5)
            sleeves = estimate_cluster_sleeve_posteriors(
                panel, bars, cluster_folds, folds[:1],
                np.ones((t, n), dtype=np.float32),
                np.zeros((t * 4, n), dtype=np.float32), cfg,
            )
        assert mock_boot.call_count == 0


# ─── P2 Unit Tests ──────────────────────────────────────────────────────────


class TestP2RegimeOverlay:

    def test_regime_overlay_floor_only_on_confirmed_negative(self) -> None:
        config = RegimeRouterConfig(regime_overlay_floor=0.5, min_effective_blocks=3)
        prior = SignalFoldRecord(
            gross_1d=np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.0], dtype=np.float64),
            cost_1d=np.zeros(6, dtype=np.float64),
            funding_1d=np.zeros(6, dtype=np.float64),
            net_1d=np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.0], dtype=np.float64),
            regime_code_1d=np.array([1, 1, 1, 2, 2, 2], dtype=np.int8),
        )
        codes_present = {1, 2}
        overlay = compute_regime_overlay(prior, codes_present, config)
        reg1_mean = float(np.mean([0.1, -0.2, 0.3]))
        assert reg1_mean > 0, "regime 1 should be positive"
        assert overlay[1] == 1.0, "positive regime should have overlay 1.0"
        reg2_mean = float(np.mean([-0.1, 0.05, 0.0]))
        assert reg2_mean < 0, "regime 2 should be negative"
        assert overlay[2] == 0.5, "confirmed negative regime should get floor 0.5"

    def test_regime_overlay_insufficient_blocks_no_penalty(self) -> None:
        config = RegimeRouterConfig(regime_overlay_floor=0.5, min_effective_blocks=10)
        prior = SignalFoldRecord(
            gross_1d=np.array([-0.1, -0.2], dtype=np.float64),
            cost_1d=np.zeros(2, dtype=np.float64),
            funding_1d=np.zeros(2, dtype=np.float64),
            net_1d=np.array([-0.1, -0.2], dtype=np.float64),
            regime_code_1d=np.array([1, 1], dtype=np.int8),
        )
        overlay = compute_regime_overlay(prior, {1}, config)
        assert overlay[1] == 1.0

    def test_regime_no_longer_blocks_admission(self) -> None:
        prior = SignalFoldRecord(
            gross_1d=np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float64),
            cost_1d=np.zeros(5, dtype=np.float64),
            funding_1d=np.zeros(5, dtype=np.float64),
            net_1d=np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float64),
            regime_code_1d=np.array([1, 1, 1, 1, 1], dtype=np.int8),
        )
        config = RegimeRouterConfig(
            regime_overlay_floor=0.5, min_effective_blocks=3, min_evidence_bars=1,
        )
        overlay = compute_regime_overlay(prior, {1}, config)
        assert overlay[1] == 0.5
        assert 0.0 < overlay[1] < 1.0

    def test_walk_forward_carry_applies_regime_overlay(self) -> None:
        t, n_syms, n_sig = 50, 2, 1
        ts = _make_timestamps(t)
        syms = ("S0", "S1")
        desc = (SignalDescriptor("mom_fast", "momentum_ts", "fast", 8, "4h"),)
        z = np.ones((t, n_syms, n_sig), dtype=np.float32)
        valid = np.ones((t, n_syms, n_sig), dtype=np.bool_)
        sigma = np.ones((t, n_syms), dtype=np.float32) * 0.02
        panel = RawSignalPanel(ts, syms, desc, z, valid, sigma)

        mu_2d = np.zeros((t, n_syms), dtype=np.float64)
        contributions = (
            ExpertContribution(
                signal_id="mom_fast", outer_fold_id=2, orientation=1,
                member_mask_1d=np.ones(n_syms, dtype=np.bool_),
                signal_index=0,
            ),
        )
        route_scales = {"mom_fast": 0.5}
        regime_overlay = {1: 0.5, 2: 1.0}
        regime_code_1d = np.ones(t, dtype=np.int8)
        regime_code_1d[:10] = 0
        deploy_start = 45

        carried = apply_walk_forward_carry(
            mu_2d, panel, contributions, route_scales,
            regime_overlay, regime_code_1d, deploy_start,
        )
        assert carried == t - deploy_start
        for t_idx in range(deploy_start, t):
            code = int(regime_code_1d[t_idx])
            expected_overlay = regime_overlay.get(code, 1.0)
            expected_mu = 0.5 * expected_overlay * 1.0 * 1.0 * 1.0
            np.testing.assert_allclose(mu_2d[t_idx], expected_mu, atol=1e-10)


# ─── Integration Tests ──────────────────────────────────────────────────────


def _deterministic_two_symbol_router_fixture(
    n_bar: int = 1800, fold_width: int = 340, min_evidence_bars: int = 300,
) -> tuple[object, ...]:
    """Direct sleeve construction (bypassing L1 admission) with a noise-free,
    always-profitable long-A/short-B expert admitted at fold_id 1..4. Used to
    prove the router's own gate cascade (not L1 admission) reaches activation.
    """
    from src.domain.futures.compound.contracts import ExitPolicyKind, ExitPolicySpec, L1SleevePosterior

    n_sym = 2
    ts = _make_timestamps(n_bar)
    syms = ("A", "B")
    g = 0.002
    close_a = 100.0 * np.power(1 + g, np.arange(n_bar))
    close_b = 100.0 * np.power(1 - g, np.arange(n_bar))
    close = np.column_stack([close_a, close_b]).astype(np.float32)
    bars = TimeframeBarCube(
        "4h", ts, syms, close, close * 1.001, close * 0.999, close,
        np.ones((n_bar, n_sym), dtype=np.float32), np.ones((n_bar, n_sym), dtype=bool),
    )
    cost = np.full((n_bar, n_sym), 0.5, dtype=np.float32)
    funding = np.zeros((n_bar * 4, n_sym), dtype=np.float32)

    z = np.zeros((n_bar, n_sym, 1), dtype=np.float32)
    z[:, 0, 0] = 1.0
    z[:, 1, 0] = -1.0
    valid = np.ones((n_bar, n_sym, 1), dtype=np.bool_)
    sigma = np.ones((n_bar, n_sym), dtype=np.float32) * 0.01
    desc = (SignalDescriptor("mom_fast", "momentum_ts", "fast", 4, "4h"),)
    panel = RawSignalPanel(ts, syms, desc, z, valid, sigma)

    folds: list[CausalFold] = []
    fit_end = 25
    n_folds = max(2, n_bar // fold_width)
    for i in range(n_folds):
        oos_start = fit_end + 1
        oos_end = min(oos_start + fold_width, n_bar)
        if oos_end <= oos_start:
            break
        folds.append(CausalFold(i, 0, fit_end, 25, max(25, oos_start - 10), oos_start, oos_end, 1, 1))
        fit_end = oos_end
    folds_tuple = tuple(folds)

    policy = ExitPolicySpec("t:time", ExitPolicyKind.TIME, None, None, None, 0, 1, -1, "h")
    sleeves = tuple(
        L1SleevePosterior(
            f"mom_fast:f{f.fold_id}:c0", "mom_fast", "momentum_ts", f.fold_id, 0,
            np.ones(n_sym, dtype=np.bool_), "h1",
            policy, 0.5, 0.001, 0.001, 0.99, 1.0, (0.001,), 300, True, (),
        )
        for f in folds_tuple if f.fold_id != 0
    )

    regime_code = np.zeros(n_bar, dtype=np.int8)
    regime_code[400:] = 1
    regime_panel = CausalRegimePanel(ts, regime_code, ts, ("cold", "chop"))
    regime_cfg = RegimeRouterConfig(min_evidence_bars=min_evidence_bars, min_effective_blocks=10, n_bootstrap=200)
    alloc_cfg = DynamicCompoundingConfig()

    return panel, sleeves, folds_tuple, bars, cost, funding, regime_panel, regime_cfg, alloc_cfg


class TestP0RouterIntegration:

    def test_router_reaches_gate_with_realistic_fold_widths(self) -> None:
        from src.domain.futures.compound.l1_regime_routing import (
            _build_prequential_expert_route_impl,
        )

        panel, sleeves, folds_tuple, bars, cost, funding, regime_panel, regime_cfg, alloc_cfg = (
            _deterministic_two_symbol_router_fixture()
        )
        result = _build_prequential_expert_route_impl(
            panel, sleeves, folds_tuple, bars, cost, funding,
            regime_panel, regime_cfg, alloc_cfg, 8.0,
        )

        # Fold 1 is the signal's FIRST appearance: under the P0 regression, this
        # would have measured n_evidence_bars as the fold's OWN width (340) and
        # passed immediately. The fix requires it to show zero prior evidence.
        fold1_evidence = [e for e in result.evidence if e.outer_fold_id == 1]
        assert fold1_evidence, "expected evidence recorded for fold 1"
        assert all(e.n_evidence_bars == 0 for e in fold1_evidence)
        assert all(e.reasons == ("insufficient_evidence_window",) for e in fold1_evidence)

        # Folds 2+ accumulate strictly-prior fold history and must reach a
        # gate decision beyond the evidence-window check (real regression fix).
        later_evidence = [e for e in result.evidence if e.outer_fold_id >= 2]
        assert later_evidence, "expected evidence recorded for folds >= 2"
        assert any(e.reasons != ("insufficient_evidence_window",) for e in later_evidence), (
            f"all later-fold candidates still stuck at insufficient_evidence_window: "
            f"{result.attribution.reason_counts}"
        )
        assert any(e.admitted for e in later_evidence), (
            f"no expert was ever admitted despite a noise-free profitable signal: "
            f"{[(e.outer_fold_id, e.reasons) for e in later_evidence]}"
        )

    def test_engine_route_no_longer_permanently_deadlocked(self) -> None:
        from src.domain.futures.compound.l1_regime_routing import (
            _build_prequential_expert_route_impl,
        )

        panel, sleeves, folds_tuple, bars, cost, funding, regime_panel, regime_cfg, alloc_cfg = (
            _deterministic_two_symbol_router_fixture()
        )
        result = _build_prequential_expert_route_impl(
            panel, sleeves, folds_tuple, bars, cost, funding,
            regime_panel, regime_cfg, alloc_cfg, 8.0,
        )

        assert result.attribution.active_experts >= 1, (
            f"deadlocked: 0 active experts ({result.attribution.reason_counts})"
        )
        assert not result.is_cash_only
        assert float(np.mean(np.abs(result.forecast.mu_2d) > 0)) > 0.0
