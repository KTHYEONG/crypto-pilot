from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    SleeveMetaSamples,
    _purged_train_val_split,
    build_sleeve_meta_dataset,
    evaluate_meta_feasibility,
)


def _make_cache_with_fields(**kwargs: object) -> MagicMock:
    cache = MagicMock(spec=L2SimulationCache)
    for k, v in kwargs.items():
        setattr(cache, k, v)
    return cache


def _make_aligned_with_fields(**kwargs: object) -> MagicMock:
    aligned = MagicMock()
    for k, v in kwargs.items():
        setattr(aligned, k, v)
    return aligned


def _default_close(n_bars: int = 10, n_sym: int = 1) -> NDArray[np.float64]:
    c = np.zeros((n_bars, n_sym), dtype=np.float64)
    for i in range(n_sym):
        c[:, i] = 100.0 + np.arange(n_bars, dtype=np.float64) * (1.0 + i * 0.5)
    return c


class TestBuildMetaDatasetLabelDirectionalSign:
    """S1 — label directional sign."""

    def test_side_positive_rising_close_yields_positive_label(self) -> None:
        T, S, N = 10, 1, 1
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[1:6, 0] = True
        side = np.full((T, S), 1.0, dtype=np.float64)
        exp_net = np.zeros((T, S), dtype=np.float64)
        qw = np.ones((T, S), dtype=np.float64)
        hbars = np.full((T, S), 3.0, dtype=np.float64)
        vol = np.ones((T, N), dtype=np.float64)
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        sleeve_ids = (("sym0", "ma_4h"),)
        sleeve_to_tf = ("4h",)
        close = _default_close(T, N)
        funding = np.zeros((T, N), dtype=np.float64)
        regime = np.zeros(T, dtype=np.int8)

        cache = _make_cache_with_fields(
            signal_mask_2d=mask,
            side_2d=side,
            expected_net_bps_2d=exp_net,
            quality_weight_2d=qw,
            holding_bars_2d=hbars,
            vol_matrix_2d=vol,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=sleeve_to_tf,
        )
        aligned = _make_aligned_with_fields(close_2d=close, funding_2d=funding)

        samples = build_sleeve_meta_dataset(cache, aligned, regime, 0, T, cost_bps=1.0)
        assert len(samples.y) > 0
        assert np.all(samples.y > 0), f"expected y>0 for long+rising, got {samples.y}"

    def test_side_negative_rising_close_yields_negative_label(self) -> None:
        T, S, N = 10, 1, 1
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[1:6, 0] = True
        side = np.full((T, S), -1.0, dtype=np.float64)
        exp_net = np.zeros((T, S), dtype=np.float64)
        qw = np.ones((T, S), dtype=np.float64)
        hbars = np.full((T, S), 3.0, dtype=np.float64)
        vol = np.ones((T, N), dtype=np.float64)
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        sleeve_ids = (("sym0", "ma_4h"),)
        sleeve_to_tf = ("4h",)
        close = _default_close(T, N)
        funding = np.zeros((T, N), dtype=np.float64)
        regime = np.zeros(T, dtype=np.int8)

        cache = _make_cache_with_fields(
            signal_mask_2d=mask,
            side_2d=side,
            expected_net_bps_2d=exp_net,
            quality_weight_2d=qw,
            holding_bars_2d=hbars,
            vol_matrix_2d=vol,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=sleeve_to_tf,
        )
        aligned = _make_aligned_with_fields(close_2d=close, funding_2d=funding)

        samples = build_sleeve_meta_dataset(cache, aligned, regime, 0, T, cost_bps=1.0)
        assert len(samples.y) > 0
        assert np.all(samples.y < 0), f"expected y<0 for short+rising, got {samples.y}"


class TestMetaFeaturesNoLookahead:
    """S2 — features must not use future info."""

    def test_features_unchanged_when_future_close_altered(self) -> None:
        T, S, N = 10, 1, 1
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[2, 0] = True
        side = np.ones((T, S), dtype=np.float64)
        exp_net = np.zeros((T, S), dtype=np.float64)
        qw = np.ones((T, S), dtype=np.float64)
        hbars = np.full((T, S), 3.0, dtype=np.float64)
        vol = np.ones((T, N), dtype=np.float64)
        vol[2, 0] = 0.02
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        sleeve_ids = (("sym0", "ma_4h"),)
        sleeve_to_tf = ("4h",)
        close = _default_close(T, N)
        funding = np.zeros((T, N), dtype=np.float64)
        regime = np.zeros(T, dtype=np.int8)
        regime[2] = 1

        cache = _make_cache_with_fields(
            signal_mask_2d=mask,
            side_2d=side,
            expected_net_bps_2d=exp_net,
            quality_weight_2d=qw,
            holding_bars_2d=hbars,
            vol_matrix_2d=vol,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=sleeve_to_tf,
        )

        aligned = _make_aligned_with_fields(close_2d=close, funding_2d=funding)
        samples = build_sleeve_meta_dataset(cache, aligned, regime, 0, T, cost_bps=1.0)
        assert len(samples.X) == 1
        features_before = samples.X[0].copy()

        close_altered = close.copy()
        close_altered[3:, 0] = 200.0
        aligned2 = _make_aligned_with_fields(close_2d=close_altered, funding_2d=funding)
        samples2 = build_sleeve_meta_dataset(cache, aligned2, regime, 0, T, cost_bps=1.0)
        assert len(samples2.X) == 1
        features_after = samples2.X[0]

        np.testing.assert_array_equal(
            features_before, features_after,
            err_msg="features at t=2 must not depend on future close",
        )


class TestCrossSleeveAgreementFraction:
    """S3 — cross-sleeve agreement = 2/3 for (+,+,-)."""

    def test_agreement_two_thirds(self) -> None:
        T, S, N = 5, 3, 1
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[2, :] = True
        side = np.zeros((T, S), dtype=np.float64)
        side[2, 0] = 1.0
        side[2, 1] = 1.0
        side[2, 2] = -1.0
        exp_net = np.zeros((T, S), dtype=np.float64)
        qw = np.ones((T, S), dtype=np.float64)
        hbars = np.full((T, S), 3.0, dtype=np.float64)
        vol = np.ones((T, N), dtype=np.float64)
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        sleeve_ids = (
            ("sym0", "ma_4h"),
            ("sym0", "bb_4h"),
            ("sym0", "donchian_4h"),
        )
        sleeve_to_tf = ("4h", "4h", "4h")
        close = _default_close(T, N)
        funding = np.zeros((T, N), dtype=np.float64)
        regime = np.zeros(T, dtype=np.int8)

        cache = _make_cache_with_fields(
            signal_mask_2d=mask,
            side_2d=side,
            expected_net_bps_2d=exp_net,
            quality_weight_2d=qw,
            holding_bars_2d=hbars,
            vol_matrix_2d=vol,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=sleeve_to_tf,
        )
        aligned = _make_aligned_with_fields(close_2d=close, funding_2d=funding)
        samples = build_sleeve_meta_dataset(cache, aligned, regime, 0, T, cost_bps=1.0)

        assert len(samples.X) == 3
        agreement_idx = samples.feature_names.index("agreement")
        agreements = samples.X[:, agreement_idx]
        assert np.allclose(agreements, 2.0 / 3.0), f"expected 2/3, got {agreements}"


class TestPurgedEmbargoNoTrainValOverlap:
    """S4 — purged/embargo blocks leakage."""

    def test_no_overlap_after_embargo(self) -> None:
        n = 100
        t_vals = np.arange(n, dtype=np.int64)
        val_start, val_end = 40, 60
        embargo = 10

        train_idx, val_idx = _purged_train_val_split(t_vals, val_start, val_end, embargo)
        assert len(val_idx) == 20

        val_t_min = int(np.min(t_vals[val_idx]))
        val_t_max = int(np.max(t_vals[val_idx]))

        for ti in train_idx:
            tt = int(t_vals[ti])
            in_embargo = (val_t_min - embargo <= tt <= val_t_max + embargo)
            assert not in_embargo, (
                f"train t={tt} inside embargo [{val_t_min - embargo}, {val_t_max + embargo}]"
            )

    def test_embargo_blocks_proximal_train(self) -> None:
        t_vals = np.array([5, 15, 25, 35, 45, 55, 65, 75], dtype=np.int64)
        val_start, val_end = 3, 6
        embargo = 10

        train_idx, val_idx = _purged_train_val_split(t_vals, val_start, val_end, embargo)
        val_t_min = int(np.min(t_vals[val_idx]))
        val_t_max = int(np.max(t_vals[val_idx]))

        for ti in train_idx:
            tt = int(t_vals[ti])
            assert not (val_t_min - embargo <= tt <= val_t_max + embargo), (
                f"train t={tt} should be purged"
            )


class TestMetaIcSignAndSignificance:
    """S5 — meta-IC sign."""

    def test_perfect_prediction_yields_ic_approx_one(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        y = np.where(rng.uniform(size=n) > 0.5, 1.0, -1.0) * rng.uniform(0.5, 2.0, size=n)
        x1 = y * 0.5 + rng.normal(0, 0.01, size=n)
        x2 = rng.normal(0, 0.1, size=n)
        X = np.column_stack([x1, x2])

        samples = SleeveMetaSamples(
            X=X.astype(np.float64),
            y=y.astype(np.float64),
            event_t=np.arange(n, dtype=np.int64),
            event_sym=np.zeros(n, dtype=np.int64),
            sleeve_tf=("4h",) * n,
            sleeve_family=("ma",) * n,
            feature_names=("x1", "x2"),
        )

        report = evaluate_meta_feasibility(
            samples, n_splits=3, embargo_bars=5, threshold_quantile=0.5,
        )

        assert np.isfinite(report.oos_meta_ic)
        assert report.oos_meta_ic > 0.8, f"expected IC≈1, got {report.oos_meta_ic}"

    def test_random_features_yield_ic_near_zero(self) -> None:
        n = 200
        rng = np.random.default_rng(99)
        y = rng.normal(0, 1.0, size=n)
        X = rng.normal(0, 1.0, size=(n, 5))

        samples = SleeveMetaSamples(
            X=X.astype(np.float64),
            y=y.astype(np.float64),
            event_t=np.arange(n, dtype=np.int64),
            event_sym=np.zeros(n, dtype=np.int64),
            sleeve_tf=("4h",) * n,
            sleeve_family=("ma",) * n,
            feature_names=("f1", "f2", "f3", "f4", "f5"),
        )

        report = evaluate_meta_feasibility(
            samples, n_splits=3, embargo_bars=5, threshold_quantile=0.5,
        )

        if np.isfinite(report.oos_meta_ic):
            assert abs(report.oos_meta_ic) < 0.3, f"expected IC~0, got {report.oos_meta_ic}"


class TestNetEdgeLiftPositiveWhenScoreInformative:
    """S6 — net-edge lift positive for informative meta-score."""

    def test_lift_positive_when_score_identifies_high_y(self) -> None:
        n = 100
        rng = np.random.default_rng(7)
        y = np.where(rng.uniform(size=n) > 0.5, 2.0, -2.0) * rng.uniform(0.5, 1.5, size=n)
        x1 = y * 0.5 + rng.normal(0, 0.05, size=n)
        x2 = rng.normal(0, 0.2, size=n)
        X = np.column_stack([x1, x2])

        samples = SleeveMetaSamples(
            X=X.astype(np.float64),
            y=y.astype(np.float64),
            event_t=np.arange(n, dtype=np.int64),
            event_sym=np.zeros(n, dtype=np.int64),
            sleeve_tf=("4h",) * n,
            sleeve_family=("ma",) * n,
            feature_names=("x1", "x2"),
        )

        report = evaluate_meta_feasibility(
            samples, n_splits=3, embargo_bars=5, threshold_quantile=0.5,
        )

        if np.isfinite(report.net_edge_lift_bps):
            assert report.net_edge_lift_bps > 0.0, (
                f"expected positive lift, got {report.net_edge_lift_bps}"
            )


class TestMetaDatasetEmptyAndBoundarySafe:
    """S7 — empty/boundary safe."""

    def test_empty_active_sleeves_returns_empty(self) -> None:
        T, S, N = 10, 2, 1
        mask = np.zeros((T, S), dtype=np.bool_)
        side = np.zeros((T, S), dtype=np.float64)
        exp_net = np.zeros((T, S), dtype=np.float64)
        qw = np.ones((T, S), dtype=np.float64)
        hbars = np.ones((T, S), dtype=np.float64)
        vol = np.ones((T, N), dtype=np.float64)
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        sleeve_ids = (("sym0", "ma_4h"), ("sym0", "bb_4h"))
        sleeve_to_tf = ("4h", "4h")
        close = _default_close(T, N)
        funding = np.zeros((T, N), dtype=np.float64)
        regime = np.zeros(T, dtype=np.int8)

        cache = _make_cache_with_fields(
            signal_mask_2d=mask,
            side_2d=side,
            expected_net_bps_2d=exp_net,
            quality_weight_2d=qw,
            holding_bars_2d=hbars,
            vol_matrix_2d=vol,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=sleeve_to_tf,
        )
        aligned = _make_aligned_with_fields(close_2d=close, funding_2d=funding)
        samples = build_sleeve_meta_dataset(cache, aligned, regime, 0, T, cost_bps=1.0)

        assert len(samples.y) == 0
        assert samples.X.shape == (0, 0)

    def test_evaluate_empty_samples_returns_nan_gracefully(self) -> None:
        samples = SleeveMetaSamples(
            X=np.empty((0, 0), dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            event_t=np.empty(0, dtype=np.int64),
            event_sym=np.empty(0, dtype=np.int64),
            sleeve_tf=(),
            sleeve_family=(),
            feature_names=(),
        )
        report = evaluate_meta_feasibility(
            samples, n_splits=3, embargo_bars=5, threshold_quantile=0.5,
        )

        assert np.isnan(report.oos_meta_ic)
        assert report.n_oos == 0
