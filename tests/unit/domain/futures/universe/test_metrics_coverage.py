"""Unit tests for metrics_coverage.py.

docs/specs/l1-nontrend-diversification-measure-first.md C1 (Phase 1b) 참조.
OI/LSR metrics parquet 캐시의 심볼별 커버리지를 측정해 xs_oi_skew 등
OI/LSR 의존 family 트랙을 진행할지(track_go) 판정한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.metrics_coverage import (
    MIN_ELIGIBLE_SYMBOLS,
    MIN_METRICS_COVERAGE_RATIO,
    compute_metrics_coverage_report,
)


@pytest.fixture
def metrics_dir(tmp_path: Path) -> Path:
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    full = pd.DataFrame({
        "datetime": idx,
        "sum_open_interest": np.linspace(1e6, 2e6, 100),
        "long_short_ratio": np.full(100, 1.2),
    })
    full.to_parquet(tmp_path / "AAAUSDT_metrics.parquet")
    sparse = full.copy()
    sparse.loc[sparse.index[:60], "sum_open_interest"] = np.nan  # 40% coverage
    sparse.to_parquet(tmp_path / "BBBUSDT_metrics.parquet")
    return tmp_path


class TestComputeMetricsCoverageReport:
    def test_full_coverage_marks_eligible(self, metrics_dir: Path) -> None:
        report = compute_metrics_coverage_report(("AAAUSDT",), data_dir=metrics_dir)

        assert report.entries[0].eligible is True
        assert report.entries[0].oi_non_null_ratio == pytest.approx(1.0)
        assert report.entries[0].lsr_non_null_ratio == pytest.approx(1.0)

    def test_sparse_below_threshold_not_eligible(self, metrics_dir: Path) -> None:
        report = compute_metrics_coverage_report(("BBBUSDT",), data_dir=metrics_dir)

        assert report.entries[0].eligible is False
        assert report.entries[0].oi_non_null_ratio == pytest.approx(0.4)

    def test_missing_file_returns_zero_entry(self, metrics_dir: Path) -> None:
        report = compute_metrics_coverage_report(("ZZZUSDT",), data_dir=metrics_dir)

        entry = report.entries[0]
        assert entry.n_rows == 0
        assert entry.eligible is False
        assert entry.oi_non_null_ratio == 0.0
        assert entry.lsr_non_null_ratio == 0.0
        assert entry.first_valid_ts is None

    def test_track_go_false_when_below_min_symbols(self, metrics_dir: Path) -> None:
        report = compute_metrics_coverage_report(
            ("AAAUSDT", "BBBUSDT", "ZZZUSDT"), data_dir=metrics_dir,
        )

        assert report.n_eligible == 1
        assert report.track_go is False

    def test_track_go_true_when_min_symbols_met(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        for sym in ("A1USDT", "A2USDT", "A3USDT"):
            df = pd.DataFrame({
                "datetime": idx,
                "sum_open_interest": np.linspace(1e6, 2e6, 100),
                "long_short_ratio": np.full(100, 1.2),
            })
            df.to_parquet(tmp_path / f"{sym}_metrics.parquet")

        report = compute_metrics_coverage_report(
            ("A1USDT", "A2USDT", "A3USDT"), data_dir=tmp_path,
        )

        assert report.n_eligible == 3
        assert report.track_go is True

    def test_respects_time_window(self, metrics_dir: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        # Only the back half of the window (indices 60-99) has valid OI.
        window_start = idx[70]
        window_end = idx[99]

        report = compute_metrics_coverage_report(
            ("BBBUSDT",), data_dir=metrics_dir, start=window_start, end=window_end,
        )

        # Within [70, 99], sparse frame (NaN at [0:60]) is fully valid.
        assert report.entries[0].oi_non_null_ratio == pytest.approx(1.0)

    def test_default_thresholds_match_spec_constants(self) -> None:
        assert MIN_METRICS_COVERAGE_RATIO == pytest.approx(0.70)
        assert MIN_ELIGIBLE_SYMBOLS == 3
