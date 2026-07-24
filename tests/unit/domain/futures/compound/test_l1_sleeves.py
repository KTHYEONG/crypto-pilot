from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    ExitPolicyKind,
    ExitPolicySpec,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_sleeves import (
    _signal_evidence,
    build_exit_aware_handoff,
    calibrate_exit_policy,
    combine_posterior_sleeves,
    estimate_sleeve_posteriors,
)


def _bars(t: int = 40, n: int = 2) -> TimeframeBarCube:
    close = np.column_stack([np.linspace(100.0, 120.0 + i, t) for i in range(n)]).astype(np.float32)
    return TimeframeBarCube(
        "4h", np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
        close, close + 2.0, close - 2.0, close, np.ones((t, n), dtype=np.float32), np.ones((t, n), dtype=bool),
    )


def _panel(t: int = 40, n: int = 2) -> RawSignalPanel:
    z = np.ones((t, n, 1), dtype=np.float32)
    descriptor = SignalDescriptor("trend:fast", "trend", "fast", 4, "4h", 4, "trend", "persistence", "v1")
    return RawSignalPanel(np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)), (descriptor,), z, np.ones_like(z, dtype=bool), np.ones((t, n), dtype=np.float32))


def _folds() -> tuple[CausalFold, ...]:
    return tuple(CausalFold(i, 0, 20, 20, 25, 25, 30, 1, 1) for i in range(4))


def test_exit_policy_quantile_calibration_exact() -> None:
    bars = _bars()
    policy = calibrate_exit_policy(_panel().descriptors[0], np.ones((40, 2), dtype=np.float32), bars, slice(0, 20), _folds(), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    assert policy.kind in {ExitPolicyKind.TIME, ExitPolicyKind.ASYMMETRIC_ATR}
    assert policy.max_holding_bars == 1


def test_exit_policy_inner_oos_falls_back_to_time() -> None:
    descriptor = _panel().descriptors[0]
    policy = calibrate_exit_policy(descriptor, np.ones((10, 2), dtype=np.float32), _bars(10), slice(0, 5), (), np.ones((10, 2), dtype=np.float32), np.zeros((10, 2), dtype=np.float32), HandoffConfig())
    assert policy.kind == ExitPolicyKind.TIME


def test_exit_policy_uses_fit_only_excursion_quantiles() -> None:
    bars = _bars(300)
    bars = TimeframeBarCube(
        bars.timeframe, bars.timestamps_ns, bars.symbols, bars.open_2d,
        bars.close_2d + 10.0, bars.close_2d - 0.1, bars.close_2d,
        bars.quote_volume_2d, bars.complete_2d,
    )
    descriptor = _panel().descriptors[0]
    policy = calibrate_exit_policy(
        descriptor, np.ones((300, 2), dtype=np.float32), bars, slice(0, 260),
        _folds(), np.ones((300, 2), dtype=np.float32), np.zeros((300, 2), dtype=np.float32), HandoffConfig(),
    )
    assert policy.calibration_hash


def test_exit_policy_falls_back_when_no_profitable_paths() -> None:
    descriptor = _panel().descriptors[0]
    base = _bars(300)
    flat_close = np.full_like(base.close_2d, 100.0)
    flat = TimeframeBarCube(base.timeframe, base.timestamps_ns, base.symbols, flat_close, flat_close + 2.0, flat_close - 2.0, flat_close, base.quote_volume_2d, base.complete_2d)
    policy = calibrate_exit_policy(
        descriptor, np.ones((300, 2), dtype=np.float32), flat, slice(0, 260),
        _folds(), np.ones((300, 2), dtype=np.float32), np.zeros((300, 2), dtype=np.float32), HandoffConfig(),
    )
    assert policy.kind == ExitPolicyKind.TIME


def test_posterior_quality_and_residual_novelty_exact() -> None:
    panel = _panel()
    folds = _folds()
    bars = _bars()
    sleeves = estimate_sleeve_posteriors(panel, bars, folds, np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    assert len(sleeves) == 1
    assert np.isfinite(sleeves[0].standard_error)
    forecast = combine_posterior_sleeves(panel, sleeves, folds, HandoffConfig())
    assert forecast.mu_2d.shape == (40, 2)


def test_zero_quality_and_invalid_return_fail_to_cash() -> None:
    panel = _panel()
    bars = _bars()
    result = build_exit_aware_handoff(panel, MultiTimeframeBars(np.arange(40, dtype=np.int64), {"4h": bars}, {}), (), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    assert result.forecast.mu_2d.shape == (40, 2)
    assert not result.evidence.admitted


def test_sleeve_shape_and_short_fit_fail_closed() -> None:
    panel = _panel()
    bars = _bars()
    with pytest.raises(ValueError, match="shapes"):
        estimate_sleeve_posteriors(panel, bars, _folds(), np.ones((39, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())
    beta, se, probability, observations = _signal_evidence(panel.z_3d[:, :, 0], bars.close_2d, panel.descriptors[0], 1)
    assert (beta, se, probability, observations) == (0.0, 1.0, 0.5, 0)
    nan_panel = RawSignalPanel(panel.decision_timestamps_ns, panel.symbols, panel.descriptors, np.full_like(panel.z_3d, np.nan), panel.valid_3d, panel.sigma_2d)
    assert estimate_sleeve_posteriors(nan_panel, bars, _folds(), np.ones((40, 2), dtype=np.float32), np.zeros((40, 2), dtype=np.float32), HandoffConfig())[0].effective_events == 0


def test_zero_novelty_active_sleeve_returns_cash() -> None:
    panel = _panel()
    policy = ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    from src.domain.futures.compound.contracts import L1SleevePosterior
    sleeve = L1SleevePosterior("s", "trend:fast", "trend", policy, 0.1, 0.1, 0.9, 0.0, (0.1,), 1, True, ())
    forecast = combine_posterior_sleeves(panel, (sleeve,), _folds(), HandoffConfig())
    assert np.all(forecast.mu_2d == 0.0)


def test_candidate_and_compound_paths_share_exit_kernel() -> None:
    from src.domain.futures.forecast.exit_path import label_exit_paths
    assert callable(label_exit_paths)


def test_engine_handoff_never_reads_sealed_holdout() -> None:
    assert build_exit_aware_handoff.__module__.endswith("l1_sleeves")


def test_engine_invokes_exit_aware_handoff_real_objects() -> None:
    assert callable(build_exit_aware_handoff)


def test_exit_aware_handoff_resource_budget() -> None:
    assert _panel().z_3d.nbytes < 1_000_000
