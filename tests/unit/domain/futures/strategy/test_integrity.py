from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import FeatureIntegrityConfig
from src.domain.futures.strategy.contracts import FeaturePanel
from src.domain.futures.strategy.integrity import (
    select_features,
    verify_data_integrity,
    verify_feature_integrity,
)


def _aligned_base() -> AlignedMarketData:
    t_len, n_len = 12, 2
    dt = np.array(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(4 * i, "h") for i in range(t_len)],
        dtype="datetime64[ns]",
    )
    px = np.full((t_len, n_len), 100.0, dtype=np.float64)
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=px.copy(),
        high_2d=px.copy() + 2.0,
        low_2d=px.copy() - 2.0,
        close_2d=px.copy(),
        volume_2d=np.full((t_len, n_len), 10.0, dtype=np.float64),
        funding_2d=np.zeros((t_len, n_len), dtype=np.float64),
        active_mask=np.ones((t_len, n_len), dtype=bool),
        warm_mask=np.zeros((t_len, n_len), dtype=bool),
        entry_block_mask=np.zeros((t_len, n_len), dtype=bool),
        kill_mask=np.zeros((t_len, n_len), dtype=bool),
    )


def test_verify_data_integrity_detects_zero_price_and_nan_decomposition() -> None:
    aligned = _aligned_base()
    aligned.open_2d[2, 0] = 0.0
    aligned.close_2d[2, 0] = 0.0
    fwd = np.zeros((12, 2), dtype=np.float64)
    eligible = np.ones((12, 2), dtype=bool)
    fwd[8:, 0] = np.nan
    eligible[8:, 0] = False
    report = verify_data_integrity(
        aligned,
        oos_start_idx=8,
        forward_gross_ret=fwd,
        eligible_mask=eligible,
    )
    assert report.hard_fail is True
    assert report.zero_price_ratio > 0.0
    assert abs(sum(report.nan_decomposition.values()) - 1.0) < 1e-9


def test_verify_data_integrity_coverage_within_eligible_full_when_no_dropout() -> None:
    # Arrange: all eligible cells have finite target
    aligned = _aligned_base()
    t_len, n_len = 12, 2
    fwd = np.ones((t_len, n_len), dtype=np.float64) * 0.01
    eligible = np.ones((t_len, n_len), dtype=bool)

    # Act
    report = verify_data_integrity(
        aligned,
        oos_start_idx=8,
        forward_gross_ret=fwd,
        eligible_mask=eligible,
    )

    # Assert
    import pytest
    assert report.coverage_within_eligible == pytest.approx(1.0)
    assert report.hard_fail is False


def test_verify_data_integrity_coverage_within_eligible_detects_dropout() -> None:
    # Arrange: eligible OOS cells where price is invalid (open=0) → target NaN via price corruption.
    # fwd must be NaN AND open/close must be ≤0 to hit the price_missing bucket.
    aligned = _aligned_base()
    t_len, n_len = 12, 2
    fwd = np.ones((t_len, n_len), dtype=np.float64) * 0.01
    eligible = np.ones((t_len, n_len), dtype=bool)
    # Corrupt OOS price for symbol 0 → both fwd NaN and price invalid
    fwd[10:, 0] = np.nan
    aligned.open_2d[10:, 0] = 0.0
    aligned.close_2d[10:, 0] = 0.0

    # Act
    report = verify_data_integrity(
        aligned,
        oos_start_idx=8,
        forward_gross_ret=fwd,
        eligible_mask=eligible,
    )

    # Assert: coverage drops below 1.0 and price_missing is detected
    assert report.coverage_within_eligible < 1.0
    assert report.nan_decomposition["price_missing"] > 0.0


def test_verify_feature_integrity_constant_drift_redundant_and_select() -> None:
    t_len, n_len = 30, 6
    dt = np.array(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(4 * i, "h") for i in range(t_len)],
        dtype="datetime64[ns]",
    )
    rng = np.random.default_rng(7)
    f0 = np.ones((t_len, n_len), dtype=np.float32)  # constant
    f1 = rng.normal(0.0, 1.0, size=(t_len, n_len)).astype(np.float32)
    f2 = f1.copy()  # redundant with f1
    f3 = rng.normal(0.0, 1.0, size=(t_len, n_len)).astype(np.float32)
    f3[20:] += 6.0  # drift in OOS
    vals = np.stack([f0, f1, f2, f3], axis=2)
    panel = FeaturePanel(
        datetimes=dt,
        symbols=tuple(f"S{i}" for i in range(n_len)),
        values=vals,
        feature_names=("const", "sig_a", "sig_b", "drift"),
        valid_mask=np.ones((t_len, n_len), dtype=bool),
    )
    target = rng.normal(0.0, 1.0, size=(t_len, n_len))
    report = verify_feature_integrity(
        panel,
        train_slice=slice(0, 20),
        oos_slice=slice(20, t_len),
        target_2d=target,
        breakeven_ic=0.01,
    )
    assert "const" in report.constant_features
    assert "drift" in report.drifted_features
    assert any({a, b} == {"sig_a", "sig_b"} for a, b, _ in report.redundant_pairs)
    selected = select_features(report, panel.feature_names, FeatureIntegrityConfig(min_keep=2))
    assert "const" not in selected

