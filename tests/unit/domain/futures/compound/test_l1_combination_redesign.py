from __future__ import annotations


import numpy as np
import pytest

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    ExecutionLedger,
    ExitPolicyKind,
    ExitPolicySpec,
    L1SleevePosterior,
    L2BenchmarkSeries,
    RawSignalPanel,
    SignalDescriptor,
)
from src.domain.futures.compound.config import L2GateConfig
from src.domain.futures.compound.l1_sleeves import (
    combine_posterior_sleeves,
    select_non_redundant_signals,
)
from src.domain.futures.compound.multiplicity import TrialMultiplicity
from src.domain.futures.compound.validation import (
    count_effective_candidates,
    evaluate_l2_walk_forward,
    slice_execution_ledger,
)


def _sleeve(
    sig: str, fold: int, mask: list[int], n: int = 4, *,
    family: str = "trend",
    admitted: bool = True,
    horizon: int = 4,
) -> L1SleevePosterior:
    m = np.zeros(n, dtype=bool)
    m[mask] = True
    p = ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, fold, "h")
    return L1SleevePosterior(
        f"{sig}:f{fold}", sig, family, fold, 0, m, "h",
        p, 0.01, 0.05, 0.9, 1.0, (0.01,), 100, admitted, (),
    )


def _panel(t: int = 40, n_sigs: int = 2, n_syms: int = 4) -> RawSignalPanel:
    descs = tuple(
        SignalDescriptor(f"sig_{i}", "trend", "fast", 4, "4h", 4 + i * 20, "", "", "v1")
        for i in range(n_sigs)
    )
    z = np.random.default_rng(42).standard_normal((t, n_syms, n_sigs)).astype(np.float32)
    valid = np.ones((t, n_syms, n_sigs), dtype=bool)
    return RawSignalPanel(
        np.arange(t, dtype=np.int64),
        tuple(f"S{j}" for j in range(n_syms)),
        descs,
        z, valid,
        np.ones((t, n_syms), dtype=np.float32),
    )


def _folds(fit_end: int = 20) -> tuple[CausalFold, ...]:
    return (CausalFold(0, 0, fit_end, fit_end, fit_end + 5, fit_end + 5, fit_end + 10, 1, 1),)


# ── Scenario 1: [LIMIT-01] Each signal gets exactly one equal vote ──

class TestCombinePosteriorSleeves:

    def test_combine_posterior_sleeves_gives_each_signal_one_equal_vote(self) -> None:
        panel = _panel(40, 2, 4)
        z0 = panel.z_3d[:, :, 0].copy()
        z1 = panel.z_3d[:, :, 1].copy()
        sleeves = (
            _sleeve("sig_0", 0, [0, 1], n=4),
            _sleeve("sig_0", 1, [2], n=4),
            _sleeve("sig_0", 2, [3], n=4),
            _sleeve("sig_0", 3, [0, 2], n=4),
            _sleeve("sig_0", 4, [1, 3], n=4),
            _sleeve("sig_0", 5, [0], n=4),
            _sleeve("sig_0", 6, [1], n=4),
            _sleeve("sig_0", 7, [2], n=4),
            _sleeve("sig_0", 8, [3], n=4),
            _sleeve("sig_0", 9, [0, 1], n=4),
            _sleeve("sig_1", 0, [0, 1, 2, 3], n=4),
        )
        config = HandoffConfig(dedup_rho_threshold=1.0)
        forecast = combine_posterior_sleeves(panel, sleeves, (), _folds(), config)
        expected = 0.5 * z0 + 0.5 * z1
        np.testing.assert_allclose(forecast.mu_2d, expected, atol=1e-6)

    def test_combine_posterior_sleeves_unions_member_masks_across_folds(self) -> None:
        panel = _panel(40, 1, 3)
        sleeves = (
            _sleeve("sig_0", 0, [0, 1], n=3),
            _sleeve("sig_0", 1, [2], n=3),
        )
        config = HandoffConfig(dedup_rho_threshold=1.0)
        forecast = combine_posterior_sleeves(panel, sleeves, (), _folds(), config)
        z0 = panel.z_3d[:, :, 0]
        expected = z0.copy()
        np.testing.assert_allclose(forecast.mu_2d, expected, atol=1e-6)

    def test_combine_posterior_sleeves_no_admitted_returns_cash_only(self) -> None:
        panel = _panel(40, 1, 4)
        sleeves = (_sleeve("sig_0", 0, [0, 1], n=4, admitted=False),)
        forecast = combine_posterior_sleeves(panel, sleeves, (), _folds(), HandoffConfig())
        assert np.all(forecast.mu_2d == 0.0)
        assert forecast.admitted_signal_ids == ()


# ── Scenario 4-6: select_non_redundant_signals ──

class TestSelectNonRedundantSignals:

    def test_select_non_redundant_signals_drops_shorter_horizon_of_inverted_pair(self) -> None:
        n_sigs = 2
        t = 200
        panel = _panel(t, n_sigs, 4)
        z = np.zeros((t, 4, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((t, 4)).astype(np.float32) * 0.01
        base = rng.standard_normal((t, 4)).astype(np.float32)
        z[:, :, 0] = base + noise
        z[:, :, 1] = -base + noise
        panel2 = RawSignalPanel(
            panel.decision_timestamps_ns, panel.symbols,
            panel.descriptors, z, panel.valid_3d, panel.sigma_2d,
        )
        result = select_non_redundant_signals(
            panel2, ("sig_0", "sig_1"),
            fit_end_exclusive=190, min_observations=100,
        )
        assert len(result) == 1
        assert result[0] == "sig_1"

    def test_select_non_redundant_signals_keeps_pair_below_min_observations(self) -> None:
        n_sigs = 2
        t = 1000
        panel = _panel(t, n_sigs, 4)
        z = np.zeros((t, 4, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        z[:, :, 0] = rng.standard_normal((t, 4)).astype(np.float32)
        z[:, :, 1] = -z[:, :, 0]
        valid = np.ones((t, 4, 2), dtype=bool)
        valid[900:, :, 0] = False
        panel2 = RawSignalPanel(
            panel.decision_timestamps_ns, panel.symbols,
            panel.descriptors, z, valid, panel.sigma_2d,
        )
        result = select_non_redundant_signals(
            panel2, ("sig_0", "sig_1"),
            fit_end_exclusive=1000,
            min_observations=3700,
        )
        assert len(result) == 2

    def test_select_non_redundant_signals_ignores_correlation_after_fit_window(self) -> None:
        n_sigs = 2
        t = 200
        panel = _panel(t, n_sigs, 4)
        z = np.zeros((t, 4, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        z[:150] = rng.standard_normal((150, 4, 2)).astype(np.float32)
        z[150:, :, 1] = -z[150:, :, 0] + rng.standard_normal((50, 4)).astype(np.float32) * 0.001
        panel2 = RawSignalPanel(
            panel.decision_timestamps_ns, panel.symbols,
            panel.descriptors, z, panel.valid_3d, panel.sigma_2d,
        )
        result = select_non_redundant_signals(
            panel2, ("sig_0", "sig_1"),
            fit_end_exclusive=150,
            rho_threshold=0.90,
        )
        assert len(result) == 2

    def test_empty_signal_ids_returns_empty(self) -> None:
        panel = _panel(40, 1, 4)
        result = select_non_redundant_signals(
            panel, (), fit_end_exclusive=20,
        )
        assert result == ()

    def test_invalid_fit_end_exclusive_raises(self) -> None:
        panel = _panel(40, 1, 4)
        with pytest.raises(ValueError, match="fit_end_exclusive"):
            select_non_redundant_signals(
                panel, ("sig_0",), fit_end_exclusive=0,
            )


# ── Scenario 7: count_effective_candidates ──

class TestCountEffectiveCandidates:

    def test_count_effective_candidates_excludes_dead_descriptors(self) -> None:
        valid = np.zeros((10, 5, 27), dtype=bool)
        for i in range(20):
            valid[:, :, i] = True
        assert count_effective_candidates(valid) == 20

    def test_all_false_returns_zero(self) -> None:
        valid = np.zeros((10, 5, 27), dtype=bool)
        assert count_effective_candidates(valid) == 0

    def test_non_3d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            count_effective_candidates(np.ones((10, 5), dtype=bool))


# ── Scenario 8: cost_drag_ratio bounded fraction ──

def _l2_fixture(n: int = 2400, *, fee_bps_per_bar: float = 0.0):
    """Real ExecutionLedger + benchmark for evaluate_l2_walk_forward, mirroring
    test_l2_growth_gate.py's test_positive_stressed_excess_growth_profile."""
    ns_per_4h = 4 * 3_600_000_000_000
    returns = np.full(n, 0.002, dtype=np.float64)
    fees = np.full(n, -fee_bps_per_bar * 1e-4, dtype=np.float64)
    net_returns = returns + fees
    timestamps = np.arange(n, dtype=np.int64) * np.int64(ns_per_4h)
    equity = np.cumprod(1.0 + net_returns)
    weights = np.full((n, 2), 0.1, dtype=np.float32)
    weights[np.arange(n) % 2 == 0] = 0.12  # oscillate so rebalance_count >= min_rebalances
    ledger = ExecutionLedger(
        timestamps_ns=timestamps, net_returns_1d=net_returns, equity_1d=equity,
        target_weights_2d=weights,
        fee_returns_1d=fees,
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True, integrity_reasons=(),
    )
    daily_ts = timestamps[:n // 6 * 6].reshape(-1, 6)[:, -1] + np.int64(ns_per_4h)
    benchmark = L2BenchmarkSeries(
        benchmark_id="test", timestamps_ns=daily_ts,
        daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
        causal_scale_1d=np.ones(n // 6, dtype=np.float64),
    )
    return ledger, benchmark


class TestCostDragAndAbsoluteCagr:

    def test_evaluate_l2_walk_forward_cost_drag_ratio_is_bounded_fraction(self) -> None:
        n = 2400
        ledger, benchmark = _l2_fixture(n, fee_bps_per_bar=1.0)
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(10, 10.0, 1.0),
            config=L2GateConfig(), bootstrap_seed=42,
        )
        assert 0.0 < result.cost_drag_ratio < 1.0
        assert result.absolute_cagr > 0.0

    def test_zero_fees_zero_drag(self) -> None:
        n = 2400
        ledger, benchmark = _l2_fixture(n, fee_bps_per_bar=0.0)
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, trial_multiplicity=TrialMultiplicity(10, 10.0, 1.0),
            config=L2GateConfig(), bootstrap_seed=42,
        )
        assert result.cost_drag_ratio == pytest.approx(0.0, abs=1e-9)


# ── Scenario 9-10 (integration): see test_engine.py
# test_engine_passes_per_symbol_cost_array_to_simulator and
# test_engine_dry_run_does_not_consume_sealed_holdout for real end-to-end coverage ──


class TestDryRunArtifacts:

    def test_dry_run_flag_in_artifacts(self) -> None:
        from src.application.futures.runner.compound_main import _write_artifacts
        import inspect
        src = inspect.getsource(_write_artifacts)
        assert '"dry_run"' in src


# ── HandoffConfig new field validation ──

class TestHandoffConfigDedupFields:

    def test_defaults(self) -> None:
        cfg = HandoffConfig()
        assert cfg.dedup_rho_threshold == 0.90
        assert cfg.min_dedup_observations == 1000

    def test_invalid_rho_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            HandoffConfig(dedup_rho_threshold=0.0)
        with pytest.raises((AssertionError, ValueError)):
            HandoffConfig(dedup_rho_threshold=1.5)

    def test_invalid_min_obs_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            HandoffConfig(min_dedup_observations=0)
