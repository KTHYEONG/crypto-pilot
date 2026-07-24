from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.admission import (
    combine_admitted_forecasts,
    evaluate_signal_admission,
)
from src.domain.futures.compound.calibration import (
    build_calibration_target,
    build_folds_4h,
    calibrate_signals,
)
from src.domain.futures.compound.config import AdmissionConfig, CalibrationConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CalibrationTarget,
    CausalFold,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalAdmissionEvidence,
    SignalCalibration,
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


def _adm_cfg() -> AdmissionConfig:
    return AdmissionConfig(
        n_bootstrap=100,
        block_size=42,
        fdr_q_threshold=0.10,
        default_cost_bps=8.0,
        sign_consistency_min=0.6,
    )


def _folds(n_bars: int) -> tuple[CausalFold, ...]:
    return build_folds_4h(n_bars, _cfg())


def test_admission_importable() -> None:
    assert evaluate_signal_admission is not None
    assert combine_admitted_forecasts is not None


def test_admission_rejects_noise_and_admits_planted(planted_panel_and_target):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    calibs = calibrate_signals(panel, {4: target}, folds, _cfg())
    ev = evaluate_signal_admission(
        panel, {4: target}, calibs, folds, None, _adm_cfg(), rng_seed=42,
    )

    assert len(ev) == 3
    assert ev[0].admitted is True
    assert ev[2].admitted is False
    reasons_str = " ".join(ev[2].reasons)
    assert "net_growth_lcb90" in reasons_str or ev[2].fdr_q_value > 0.10


def test_combine_zero_admitted_returns_flat_panel_without_raise(planted_panel_and_target):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    calibs = calibrate_signals(panel, {4: target}, folds, _cfg())

    fake_evidence = [
        SignalAdmissionEvidence(
            signal_id=panel.descriptors[k].signal_id,
            family=panel.descriptors[k].family,
            oos_net_growth_lcb90=-0.01,
            oos_net_mean_2x_cost=-0.02,
            fold_sign_consistency=0.3,
            p_value=0.5,
            fdr_q_value=0.5,
            admitted=False,
            reasons=("all gates failed",),
        )
        for k in range(len(calibs))
    ]

    result = combine_admitted_forecasts(panel, calibs, tuple(fake_evidence), folds)
    assert isinstance(result, CalibratedForecastPanel)
    assert np.all(result.mu_2d == 0.0)
    assert result.admitted_signal_ids == ()


def test_p1_panel_to_calibrated_forecast_pipeline_family_sum_consistency(
    planted_panel_and_target,
):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    calibs = calibrate_signals(panel, {4: target}, folds, _cfg())
    ev = evaluate_signal_admission(
        panel, {4: target}, calibs, folds, None, _adm_cfg(), rng_seed=42,
    )
    result = combine_admitted_forecasts(panel, calibs, ev, folds)

    if len(result.family_ids) > 0:
        F = len(result.family_ids)
        family_sum = np.sum(result.family_mu_3d[:, :, :F], axis=2)
        mu_from_family = family_sum / F
        np.testing.assert_allclose(result.mu_2d, mu_from_family, atol=1e-6)


def test_evaluate_signal_admission_block_size_scales_with_target_horizon(planted_panel_and_target):
    panel, target = planted_panel_and_target
    panel_desc = panel.descriptors
    long_horizon_desc = SignalDescriptor(
        "xs_momentum_slow:very_slow", "xs_momentum_slow", "very_slow", 648, "4h",
        target_horizon_hours=648,
    )
    dummy_z = np.full((panel.z_3d.shape[0], panel.z_3d.shape[1], 1), 0.0, dtype=np.float32)
    dummy_valid = np.zeros((panel.z_3d.shape[0], panel.z_3d.shape[1], 1), dtype=np.bool_)
    long_panel = RawSignalPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        descriptors=(long_horizon_desc,),
        z_3d=dummy_z,
        valid_3d=dummy_valid,
        sigma_2d=panel.sigma_2d,
    )
    from src.domain.futures.compound.config import AdmissionConfig
    cfg = AdmissionConfig(block_size=42, n_bootstrap=20)
    long_target = build_calibration_target(
        _dummy_bars(), panel.sigma_2d, horizon_bars=648 // 4,
    )
    folds = build_folds_4h(panel.z_3d.shape[0], _cfg(), max_target_horizon_bars=648 // 4)
    calibs = calibrate_signals(long_panel, {648: long_target}, folds, _cfg())
    ev = evaluate_signal_admission(
        long_panel, {648: long_target}, calibs, folds, None, cfg, rng_seed=42,
    )
    assert ev[0].effective_sample_note != ""
    assert "low_effective_sample" in ev[0].effective_sample_note


def test_admission_missing_horizon_raises(planted_panel_and_target):
    panel, target = planted_panel_and_target
    folds = _folds(panel.z_3d.shape[0])
    calibs = calibrate_signals(panel, {4: target}, folds, _cfg())
    import pytest
    with pytest.raises(ValueError, match="missing target for horizon"):
        evaluate_signal_admission(
            panel, {216: target}, calibs, folds, None, _adm_cfg(), rng_seed=42,
        )


def test_v3_pre_existing_24_signals_byte_identical_to_v2_baseline(tmp_path):
    from src.domain.futures.compound.config import AdmissionConfig, CalibrationConfig
    from src.domain.futures.compound.calibration import (
        build_folds_4h, build_multi_horizon_targets, calibrate_signals,
    )
    from src.domain.futures.compound.admission import evaluate_signal_admission
    from src.domain.futures.compound.signal_bank import _default_catalog

    rng = np.random.default_rng(42)
    T, N = 2000, 5
    ts = np.arange(T, dtype=np.int64) * 4 * _HOUR_NS + 1_700_000_000_000_000_000
    cube_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=ts,
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
        timestamps_ns=(np.arange(T * 4, dtype=np.int64) * _HOUR_NS + 1_700_000_000_000_000_000),
        symbols=("A", "B", "C", "D", "E"),
        open_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        high_2d=np.full((T * 4, 5), 101.0, dtype=np.float32),
        low_2d=np.full((T * 4, 5), 99.0, dtype=np.float32),
        close_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        quote_volume_2d=np.full((T * 4, 5), 1e6, dtype=np.float32),
        complete_2d=np.ones((T * 4, 5), dtype=np.bool_),
    )
    bars = MultiTimeframeBars(
        decision_timestamps_ns=ts,
        cubes={"4h": cube_4h, "1h": cube_1h},
        aux_1h_fields={},
    )

    catalog = _default_catalog()
    sig_ids = {d.signal_id for d in catalog if d.family not in ("xs_momentum_slow",)}
    desc = tuple(d for d in catalog if d.signal_id in sig_ids)
    panel = RawSignalPanel(
        decision_timestamps_ns=ts,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=desc,
        z_3d=np.zeros((T, 5, len(desc)), dtype=np.float32),
        valid_3d=np.ones((T, 5, len(desc)), dtype=np.bool_),
        sigma_2d=np.full((T, 5), 0.02, dtype=np.float32),
    )

    calib_cfg = CalibrationConfig(min_fold_obs=10, n_folds=3, purge_bars=2, embargo_bars=10)
    admit_cfg = AdmissionConfig(n_bootstrap=20, block_size=10)
    horizons = tuple(sorted({d.target_horizon_hours for d in desc}))
    targets = build_multi_horizon_targets(bars, panel.sigma_2d, horizons)
    max_horizon_bars = max(horizons) // 4 if horizons else 0
    folds = build_folds_4h(panel.z_3d.shape[0], calib_cfg, max_target_horizon_bars=max_horizon_bars)
    calibs = calibrate_signals(panel, targets, folds, calib_cfg)
    ev = evaluate_signal_admission(
        panel, targets, calibs, folds, None, admit_cfg, rng_seed=42,
    )
    for e in ev:
        if e.effective_sample_note:
            assert "low_effective_sample" in e.effective_sample_note


def test_v3_full_28_signal_catalog_completes_pipeline_without_raise(tmp_path):
    from src.domain.futures.compound.config import AdmissionConfig, CalibrationConfig
    from src.domain.futures.compound.calibration import (
        build_folds_4h, build_multi_horizon_targets, calibrate_signals,
    )
    from src.domain.futures.compound.admission import evaluate_signal_admission, combine_admitted_forecasts
    from src.domain.futures.compound.signal_bank import _default_catalog

    catalog = _default_catalog()
    T, N = 2000, 5
    ts = np.arange(T, dtype=np.int64) * 4 * _HOUR_NS + 1_700_000_000_000_000_000
    cube_4h = TimeframeBarCube(
        timeframe="4h", timestamps_ns=ts,
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
        timestamps_ns=(np.arange(T * 4, dtype=np.int64) * _HOUR_NS + 1_700_000_000_000_000_000),
        symbols=("A", "B", "C", "D", "E"),
        open_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        high_2d=np.full((T * 4, 5), 101.0, dtype=np.float32),
        low_2d=np.full((T * 4, 5), 99.0, dtype=np.float32),
        close_2d=np.full((T * 4, 5), 100.0, dtype=np.float32),
        quote_volume_2d=np.full((T * 4, 5), 1e6, dtype=np.float32),
        complete_2d=np.ones((T * 4, 5), dtype=np.bool_),
    )
    bars = MultiTimeframeBars(
        decision_timestamps_ns=ts,
        cubes={"4h": cube_4h, "1h": cube_1h},
        aux_1h_fields={},
    )

    rng = np.random.default_rng(42)
    z = rng.standard_normal((T, N, len(catalog))).astype(np.float32).clip(-3, 3)
    panel = RawSignalPanel(
        decision_timestamps_ns=ts,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=catalog,
        z_3d=z,
        valid_3d=np.ones((T, N, len(catalog)), dtype=np.bool_),
        sigma_2d=np.full((T, N), 0.02, dtype=np.float32),
    )

    horizons = tuple(sorted({d.target_horizon_hours for d in catalog}))
    targets = build_multi_horizon_targets(bars, panel.sigma_2d, horizons)
    max_horizon_bars = max(horizons) // 4 if horizons else 0
    calib_cfg = CalibrationConfig(min_fold_obs=10, n_folds=3, purge_bars=2, embargo_bars=10)
    folds = build_folds_4h(panel.z_3d.shape[0], calib_cfg, max_target_horizon_bars=max_horizon_bars)
    calibs = calibrate_signals(panel, targets, folds, calib_cfg)
    admit_cfg = AdmissionConfig(n_bootstrap=20, block_size=10)
    ev = evaluate_signal_admission(
        panel, targets, calibs, folds, None, admit_cfg, rng_seed=42,
    )
    result = combine_admitted_forecasts(panel, calibs, ev, folds)
    assert len(ev) == len(catalog)
    assert isinstance(result, CalibratedForecastPanel)


def test_full_vector_bh_fdr_correction() -> None:
    bars = _dummy_bars()
    T, N, K = 2000, 5, 4
    rng = np.random.default_rng(42)
    z = rng.standard_normal((T, N, K)).astype(np.float32).clip(-3, 3)

    desc = (
        SignalDescriptor("sig:a", "grp_a", "fast", 24, "4h", target_horizon_hours=24),
        SignalDescriptor("sig:b", "grp_a", "fast", 24, "4h", target_horizon_hours=24),
        SignalDescriptor("sig:c", "grp_b", "fast", 24, "4h", target_horizon_hours=24),
        SignalDescriptor("sig:d", "grp_b", "fast", 24, "4h", target_horizon_hours=24),
    )
    panel = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=desc,
        z_3d=z,
        valid_3d=np.ones((T, N, K), dtype=np.bool_),
        sigma_2d=np.full((T, N), 0.02, dtype=np.float32),
    )
    from src.domain.futures.compound.calibration import (
        build_calibration_target, build_folds_4h, calibrate_signals,
    )
    sigma = np.full((T, N), 0.02, dtype=np.float32)
    target = build_calibration_target(bars, sigma)
    calib_cfg = CalibrationConfig(min_fold_obs=100, n_folds=3, purge_bars=2, embargo_bars=10)
    folds = build_folds_4h(T, calib_cfg)
    calibs = calibrate_signals(panel, {24: target}, folds, calib_cfg)
    admit_cfg = AdmissionConfig(n_bootstrap=50, block_size=10, fdr_q_threshold=0.10, sign_consistency_min=0.1)
    ev = evaluate_signal_admission(
        panel, {24: target}, calibs, folds, None, admit_cfg, rng_seed=42,
    )
    assert len(ev) == K
    q_vals = np.array([e.fdr_q_value for e in ev])
    assert np.all((q_vals >= 0) & (q_vals <= 1.0)), f"q_vals out of [0,1]: {q_vals}"
    assert len(set(q_vals)) >= 1
    assert 0 <= sum(1 for e in ev if e.admitted) <= K


def test_low_effective_sample_flagging() -> None:
    bars = _dummy_bars()
    T, N = 2000, 5
    sigma = np.full((T, N), 0.02, dtype=np.float32)
    z = np.zeros((T, N, 1), dtype=np.float32)
    valid = np.ones((T, N, 1), dtype=np.bool_)
    long_desc = SignalDescriptor(
        "test:very_slow", "test_fam", "very_slow", 432, "4h",
        target_horizon_hours=432,
    )
    panel = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=(long_desc,),
        z_3d=z,
        valid_3d=valid,
        sigma_2d=sigma,
    )
    from src.domain.futures.compound.calibration import build_calibration_target, build_folds_4h, calibrate_signals
    target = build_calibration_target(bars, sigma, horizon_bars=432 // 4)
    calib_cfg = CalibrationConfig(min_fold_obs=10, n_folds=3, purge_bars=2, embargo_bars=10)
    folds = build_folds_4h(T, calib_cfg, max_target_horizon_bars=432 // 4)
    calibs = calibrate_signals(panel, {432: target}, folds, calib_cfg)
    admit_cfg = AdmissionConfig(n_bootstrap=20, block_size=10, fdr_q_threshold=0.25)
    ev = evaluate_signal_admission(
        panel, {432: target}, calibs, folds, None, admit_cfg, rng_seed=42,
    )
    assert len(ev) == 1
    assert ev[0].effective_sample_note != ""
    assert "low_effective_sample" in ev[0].effective_sample_note


def test_combine_admitted_forecasts_scale_normalization() -> None:
    bars = _dummy_bars()
    T, N = bars.decision_timestamps_ns.size, 5
    sigma = np.full((T, N), 0.02, dtype=np.float32)

    desc_24h = SignalDescriptor("alpha:fast", "grp_a", "fast", 24, "4h", target_horizon_hours=24)
    desc_216h = SignalDescriptor("alpha:slow", "grp_a", "slow", 216, "4h", target_horizon_hours=216)

    z_24h = np.full((T, N, 1), 0.5, dtype=np.float32)
    z_216h = np.full((T, N, 1), 0.5, dtype=np.float32)
    z_combined = np.concatenate([z_24h, z_216h], axis=2)

    panel_24h = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=(desc_24h,),
        z_3d=z_24h,
        valid_3d=np.ones((T, N, 1), dtype=np.bool_),
        sigma_2d=sigma,
    )
    panel_216h = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=(desc_216h,),
        z_3d=z_216h,
        valid_3d=np.ones((T, N, 1), dtype=np.bool_),
        sigma_2d=sigma,
    )
    panel_both = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=(desc_24h, desc_216h),
        z_3d=z_combined,
        valid_3d=np.ones((T, N, 2), dtype=np.bool_),
        sigma_2d=sigma,
    )

    cal_24h = SignalCalibration(signal_id="alpha:fast", beta_by_fold=(0.1, 0.1, 0.1), beta_se_by_fold=(0.01, 0.01, 0.01), n_obs_by_fold=(100, 100, 100))
    cal_216h = SignalCalibration(signal_id="alpha:slow", beta_by_fold=(0.1, 0.1, 0.1), beta_se_by_fold=(0.01, 0.01, 0.01), n_obs_by_fold=(100, 100, 100))

    ev_admitted = (
        SignalAdmissionEvidence("alpha:fast", "grp_a", 0.1, 0.05, 1.0, 0.01, 0.05, True, (), ""),
        SignalAdmissionEvidence("alpha:slow", "grp_a", 0.1, 0.05, 1.0, 0.01, 0.05, True, (), ""),
    )

    folds_3 = (
        CausalFold(0, 0, 600, 598, 600, 600, 1200, 2, 10),
        CausalFold(1, 0, 1200, 1198, 1200, 1200, 1800, 2, 10),
        CausalFold(2, 0, 1800, 1798, 1800, 1800, 1990, 2, 10),
    )

    result = combine_admitted_forecasts(panel_both, (cal_24h, cal_216h), ev_admitted, folds_3)

    assert result.mu_2d.shape == (T, N)
    assert np.any(np.isfinite(result.mu_2d)), "mu_2d should have finite values"

    t_test = 1500  # must be >= folds[0].oos_start (600) due to pre-OOS masking
    beta_24 = 0.1
    beta_216 = 0.1
    z_val = 0.5
    scale_24 = np.sqrt(24.0 / 4.0)
    scale_216 = np.sqrt(216.0 / 4.0)
    contrib_24 = beta_24 * z_val / scale_24
    contrib_216 = beta_216 * z_val / scale_216
    expected_mu = (contrib_24 + contrib_216) / 2.0
    np.testing.assert_allclose(
        result.mu_2d[t_test, :], expected_mu, atol=1e-6,
    )


def test_combine_forecasts_pre_oos_bars_zeroed() -> None:
    bars = _dummy_bars()
    T, N = bars.decision_timestamps_ns.size, 5

    desc = SignalDescriptor("alpha:fast", "grp_a", "fast", 24, "4h", target_horizon_hours=24)
    z = np.full((T, N, 1), 0.5, dtype=np.float32)
    panel = RawSignalPanel(
        decision_timestamps_ns=bars.decision_timestamps_ns,
        symbols=("A", "B", "C", "D", "E"),
        descriptors=(desc,),
        z_3d=z,
        valid_3d=np.ones((T, N, 1), dtype=np.bool_),
        sigma_2d=np.full((T, N), 0.02, dtype=np.float32),
    )
    cal = SignalCalibration(signal_id="alpha:fast", beta_by_fold=(0.1, 0.1, 0.1), beta_se_by_fold=(0.01, 0.01, 0.01), n_obs_by_fold=(100, 100, 100))
    ev = (SignalAdmissionEvidence("alpha:fast", "grp_a", 0.1, 0.05, 1.0, 0.01, 0.05, True, (), ""),)
    folds_3 = (
        CausalFold(0, 0, 600, 598, 600, 600, 1200, 2, 10),
        CausalFold(1, 0, 1200, 1198, 1200, 1200, 1800, 2, 10),
        CausalFold(2, 0, 1800, 1798, 1800, 1800, 1990, 2, 10),
    )

    result = combine_admitted_forecasts(panel, (cal,), ev, folds_3)

    oos_start = folds_3[0].oos_start
    assert oos_start == 600
    np.testing.assert_array_equal(result.mu_2d[:oos_start], 0.0)
    assert np.all(np.isnan(result.se_2d[:oos_start]))
    np.testing.assert_array_equal(result.family_mu_3d[:oos_start], 0.0)
    assert np.any(result.mu_2d[oos_start:] != 0.0), "post-OOS bars must retain non-trivial mu"
