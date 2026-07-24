from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.calibration import (
    build_calibration_target,
    build_folds_4h,
    build_multi_horizon_targets,
    calibrate_signals,
)
from src.domain.futures.compound.config import CalibrationConfig
from src.domain.futures.compound.contracts import (
    CalibrationTarget,
    CausalFold,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)

_HOUR_NS = 3_600_000_000_000


def _dummy_bars() -> MultiTimeframeBars:
    T = 2000
    ts = np.arange(T, dtype=np.int64) * 4 * _HOUR_NS + 1_700_000_000_000_000_000
    tbc = TimeframeBarCube(
        timeframe="4h",
        timestamps_ns=ts,
        symbols=("A", "B", "C", "D", "E"),
        open_2d=np.full((T, 5), 100.0, dtype=np.float32),
        high_2d=np.full((T, 5), 101.0, dtype=np.float32),
        low_2d=np.full((T, 5), 99.0, dtype=np.float32),
        close_2d=np.full((T, 5), 100.0, dtype=np.float32),
        quote_volume_2d=np.full((T, 5), 1e6, dtype=np.float32),
        complete_2d=np.ones((T, 5), dtype=np.bool_),
    )
    cube_1h = TimeframeBarCube(
        timeframe="1h",
        timestamps_ns=(np.arange(T * 4, dtype=np.int64) * _HOUR_NS
                       + 1_700_000_000_000_000_000),
        symbols=("A", "B", "C", "D", "E"),
        open_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        high_2d=np.full((T * 4, 5), 101.0, dtype=np.float32),
        low_2d=np.full((T * 4, 5), 99.0, dtype=np.float32),
        close_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        quote_volume_2d=np.full((T * 4, 5), 1e6, dtype=np.float32),
        complete_2d=np.ones((T * 4, 5), dtype=np.bool_),
    )
    return MultiTimeframeBars(
        decision_timestamps_ns=ts,
        cubes={"4h": tbc, "1d": tbc, "1h": cube_1h},
        aux_1h_fields={},
    )


@pytest.fixture
def planted_panel_and_target():
    rng = np.random.default_rng(7)
    T, N, K_sig = 2000, 5, 3
    z = rng.standard_normal((T, N, K_sig)).astype(np.float32).clip(-3, 3)
    noise = rng.standard_normal((T, N)).astype(np.float32) * 0.5
    y_raw = 0.1 * z[:, :, 0] + noise
    y_valid = np.isfinite(y_raw)

    bars = _dummy_bars()
    sigma = np.full((T, N), 0.02, dtype=np.float32)
    target = build_calibration_target(bars, sigma)

    desc = (
        SignalDescriptor("trend:fast", "trend", "fast", 24, "4h"),
        SignalDescriptor("trend:medium", "trend", "medium", 72, "4h"),
        SignalDescriptor("noise:fast", "noise", "fast", 24, "4h"),
    )

    panel = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=desc,
        z_3d=z,
        valid_3d=np.ones((T, N, K_sig), np.bool_),
        sigma_2d=sigma,
    )

    target = CalibrationTarget(
        decision_timestamps_ns=target.decision_timestamps_ns,
        y_2d=y_raw,
        valid_2d=y_valid,
    )

    return panel, target


def _cfg() -> CalibrationConfig:
    return CalibrationConfig(
        ridge_lambda_scale=0.01,
        family_shrink=0.5,
        min_fold_obs=1000,
        n_folds=5,
        purge_bars=2,
        embargo_bars=42,
    )



def _folds(n_bars: int) -> tuple[CausalFold, ...]:
    return build_folds_4h(n_bars, _cfg())


def test_calibrate_signals_recovers_planted_beta(planted_panel_and_target):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    calibs = calibrate_signals(panel, {4: target}, folds, _cfg())

    assert len(calibs) == 3
    beta0 = float(np.mean(calibs[0].beta_by_fold))
    assert beta0 == pytest.approx(0.1, rel=0.35)
    assert abs(float(np.mean(calibs[2].beta_by_fold))) < 0.02


def test_build_calibration_target_constant_price():
    bars = _dummy_bars()
    T = bars.decision_timestamps_ns.size
    sigma = np.full((T, 5), 0.02, dtype=np.float32)
    target = build_calibration_target(bars, sigma)
    assert target.y_2d.shape == (T, 5)
    assert not target.valid_2d[-1, 0]
    finite_vals = target.y_2d[np.isfinite(target.y_2d)]
    if len(finite_vals) > 0:
        assert np.allclose(finite_vals, 0.0, atol=1e-7)


def test_build_calibration_target_uses_forward_not_backward_return():
    bars = _dummy_bars()
    T = bars.decision_timestamps_ns.size

    # Price jumps ONLY between bar 100 and bar 101 (close[101] = 2x close[100]).
    # A correctly forward-looking target must attribute this jump to y[100]
    # (the return realized AFTER decision time 100), never to y[101].
    close = bars.cubes["4h"].close_2d.copy()
    close[101:] = close[101:] * 2.0
    bars.cubes["4h"].close_2d[:] = close
    sigma = np.full((T, 5), 1.0, dtype=np.float32)

    target = build_calibration_target(bars, sigma)

    assert target.y_2d[100, 0] == pytest.approx(np.log(2.0), rel=1e-6)
    assert target.y_2d[101, 0] == pytest.approx(0.0, abs=1e-6)
    assert not target.valid_2d[-1, 0]
    assert target.valid_2d[100, 0]


def test_build_calibration_target_shape_mismatch_raises():
    bars = _dummy_bars()
    sigma = np.full((100, 3), 0.02, dtype=np.float32)
    with pytest.raises(ValueError, match="sigma_2d shape"):
        build_calibration_target(bars, sigma)


def test_build_folds_4h():
    folds = build_folds_4h(2000, _cfg())
    assert len(folds) == 5
    for f in folds:
        assert f.fit_start == 0
        assert f.fit_end_exclusive > f.fit_start
        assert f.oos_start >= f.fit_end_exclusive
        assert f.oos_end_exclusive > f.oos_start
    assert folds[0].oos_start > 0


def test_build_multi_horizon_targets_returns_distinct_targets_per_horizon():
    bars = _dummy_bars()
    T = bars.decision_timestamps_ns.size
    sigma = np.full((T, 5), 0.02, dtype=np.float32)
    horizons = (4, 216, 648)
    targets = build_multi_horizon_targets(bars, sigma, horizons)
    assert isinstance(targets, dict)
    assert set(targets.keys()) == {4, 216, 648}
    for h, tgt in targets.items():
        assert isinstance(tgt, CalibrationTarget)
        assert tgt.y_2d.shape == (T, 5)
        invalid_bars = h // 4
        assert not tgt.valid_2d[-invalid_bars:, :].any()
        if invalid_bars > 1:
            assert tgt.valid_2d[-(invalid_bars + 1), 0]


def test_build_multi_horizon_targets_invalid_horizon_raises():
    bars = _dummy_bars()
    T = bars.decision_timestamps_ns.size
    sigma = np.full((T, 5), 0.02, dtype=np.float32)
    import pytest
    with pytest.raises(ValueError, match="multiple of 4"):
        build_multi_horizon_targets(bars, sigma, (4, 5))


def test_build_folds_4h_purge_scales_with_max_target_horizon():
    cfg = _cfg()
    folds = build_folds_4h(2000, cfg, max_target_horizon_bars=162)
    for f in folds:
        assert f.purge_bars >= 162


def test_build_folds_4h_without_max_target_horizon_bars_default_purge():
    cfg = _cfg()
    folds = build_folds_4h(2000, cfg)
    for f in folds:
        assert f.purge_bars == cfg.purge_bars


def test_calibrate_signals_raises_on_missing_target_horizon_key(planted_panel_and_target):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    import pytest
    with pytest.raises(ValueError, match="missing target for horizon"):
        calibrate_signals(panel, {216: target}, folds, _cfg())
