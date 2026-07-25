from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.domain.futures.data_lake.run_windows import (
    QuarterlyWindowConfig,
    QuarterlyRunWindow,
    resolve_completed_quarter_window,
)


class TestQuarterlyWindowConfig:
    def test_default_config_valid(self) -> None:
        cfg = QuarterlyWindowConfig()
        assert cfg.warmup_days == 90
        assert cfg.l1_days == 365
        assert cfg.l2_days == 365
        assert cfg.l3_days == 90

    def test_non_positive_l3_raises(self) -> None:
        with pytest.raises(ValueError, match="l3_days must be positive"):
            QuarterlyWindowConfig(l3_days=0)

    def test_non_positive_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_days must be positive"):
            QuarterlyWindowConfig(warmup_days=-1)


class TestQuarterlyRunWindow:
    def test_valid_window(self) -> None:
        window = QuarterlyRunWindow(
            requested_date=date(2026, 7, 25),
            cutoff_date=date(2026, 6, 30),
            acquisition_start_ns=1000,
            l1_start_ns=2000,
            l2_start_ns=3000,
            l3_start_ns=4000,
            cutoff_exclusive_ns=5000,
        )
        assert window.requested_date == date(2026, 7, 25)
        assert window.cutoff_date == date(2026, 6, 30)
        assert window.acquisition_start_ns < window.l1_start_ns < window.l2_start_ns < window.l3_start_ns < window.cutoff_exclusive_ns

    def test_non_increasing_boundaries_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            QuarterlyRunWindow(
                requested_date=date(2026, 7, 25),
                cutoff_date=date(2026, 6, 30),
                acquisition_start_ns=5000,
                l1_start_ns=2000,
                l2_start_ns=3000,
                l3_start_ns=4000,
                cutoff_exclusive_ns=5000,
            )


class TestResolveCompletedQuarterWindow:
    def test_resolve_july_request_to_prior_quarter_end(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 7, 25),
            QuarterlyWindowConfig(),
        )
        assert window.cutoff_date == date(2026, 6, 30)
        assert window.cutoff_exclusive_ns > 0

        cutoff_dt = datetime.fromtimestamp(window.cutoff_exclusive_ns / 1_000_000_000, tz=UTC)
        assert cutoff_dt.day == 1
        assert cutoff_dt.month == 7
        assert cutoff_dt.year == 2026

        assert window.acquisition_start_ns < window.l1_start_ns < window.l2_start_ns < window.l3_start_ns < window.cutoff_exclusive_ns

    def test_cutoff_excludes_current_quarter_partition(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 7, 25),
            QuarterlyWindowConfig(),
        )
        cutoff_dt = datetime.fromtimestamp(window.cutoff_exclusive_ns / 1_000_000_000, tz=UTC)
        assert cutoff_dt <= datetime(2026, 7, 1, tzinfo=UTC)
        july_start = datetime(2026, 7, 1, tzinfo=UTC)
        assert window.cutoff_exclusive_ns <= int(july_start.timestamp() * 1_000_000_000)

    def test_august_request_still_june_cutoff(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 8, 15),
            QuarterlyWindowConfig(),
        )
        assert window.cutoff_date == date(2026, 6, 30)

    def test_april_request_resolves_to_march(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 4, 10),
            QuarterlyWindowConfig(),
        )
        assert window.cutoff_date == date(2026, 3, 31)

    def test_layers_match_spec(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 7, 25),
            QuarterlyWindowConfig(),
        )
        cutoff_ns = window.cutoff_exclusive_ns
        ns_per_day = 86_400_000_000_000

        l3_start_ns = cutoff_ns - 90 * ns_per_day
        l2_start_ns = l3_start_ns - 365 * ns_per_day
        l1_start_ns = l2_start_ns - 365 * ns_per_day
        acq_start_ns = l1_start_ns - 90 * ns_per_day

        assert abs(window.l3_start_ns - l3_start_ns) <= ns_per_day
        assert abs(window.l2_start_ns - l2_start_ns) <= ns_per_day
        assert abs(window.l1_start_ns - l1_start_ns) <= ns_per_day
        assert abs(window.acquisition_start_ns - acq_start_ns) <= ns_per_day

    def test_january_request(self) -> None:
        window = resolve_completed_quarter_window(
            date(2026, 1, 15),
            QuarterlyWindowConfig(),
        )
        assert window.cutoff_date == date(2025, 12, 31)
