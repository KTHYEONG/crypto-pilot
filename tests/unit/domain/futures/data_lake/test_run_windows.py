from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from src.domain.futures.compound.contracts import CausalityError
from src.domain.futures.data_lake.run_windows import (
    QuarterlyWindowConfig,
    QuarterlyRunWindow,
    clamp_window_to_available_data,
    resolve_completed_quarter_window,
)


class TestQuarterlyWindowConfig:
    def test_default_config_valid(self) -> None:
        cfg = QuarterlyWindowConfig()
        assert cfg.warmup_days == 90
        assert cfg.l1_days == 365
        assert cfg.l2_days == 362
        assert cfg.l3_days == 90
        assert cfg.warmup_days + cfg.l1_days + cfg.l2_days + cfg.l3_days == 907
        assert cfg.warmup_days + cfg.l1_days + cfg.l2_days + cfg.l3_days <= 910

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
        l2_start_ns = l3_start_ns - 362 * ns_per_day
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


class TestClampWindowToAvailableData:
    _DAY_NS = 86_400 * 1_000_000_000

    def _window(self, acquisition_offset_days: int = 900) -> QuarterlyRunWindow:
        base = date(2026, 6, 30)
        cutoff_exclusive_ns = (datetime(base.year, base.month, base.day, tzinfo=UTC) + timedelta(days=1)).timestamp() * 1_000_000_000
        l3_start_ns = int(cutoff_exclusive_ns - 90 * self._DAY_NS)
        l2_start_ns = int(l3_start_ns - 362 * self._DAY_NS)
        l1_start_ns = int(l2_start_ns - 365 * self._DAY_NS)
        acq_start_ns = int(l1_start_ns - acquisition_offset_days * self._DAY_NS)
        return QuarterlyRunWindow(
            requested_date=base,
            cutoff_date=base,
            acquisition_start_ns=acq_start_ns,
            l1_start_ns=l1_start_ns,
            l2_start_ns=l2_start_ns,
            l3_start_ns=l3_start_ns,
            cutoff_exclusive_ns=int(cutoff_exclusive_ns),
        )

    def test_clamp_window_noop_when_data_sufficient(self) -> None:
        window = self._window(900)
        result = clamp_window_to_available_data(
            window, actual_data_start_ns=window.acquisition_start_ns, min_l2_days=300,
        )
        assert result == window

    def test_clamp_window_shrinks_l2_preserves_l1(self) -> None:
        window = self._window(900)
        l3_start = window.l3_start_ns
        cutoff = window.cutoff_exclusive_ns
        late_start_ns = window.acquisition_start_ns + 10 * self._DAY_NS
        result = clamp_window_to_available_data(
            window, actual_data_start_ns=late_start_ns, min_l2_days=300,
        )
        assert result.l3_start_ns == l3_start
        assert result.cutoff_exclusive_ns == cutoff
        assert result.l2_start_ns > window.l2_start_ns
        l1_duration = result.l2_start_ns - result.l1_start_ns
        assert l1_duration == window.l2_start_ns - window.l1_start_ns

    def test_clamp_window_raises_below_floor(self) -> None:
        window = self._window(900)
        late_start_ns = window.acquisition_start_ns + 200 * self._DAY_NS
        with pytest.raises(CausalityError):
            clamp_window_to_available_data(
                window, actual_data_start_ns=late_start_ns, min_l2_days=300,
            )
