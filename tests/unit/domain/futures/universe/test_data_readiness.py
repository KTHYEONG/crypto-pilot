"""Unit tests for data_readiness planner: coverage inspection, sync planning, derived cache."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.optimization.opt_config import LayeredWindow
from src.domain.futures.universe.data_readiness import (
    DataRequirement,
    build_base_data_plan,
    build_derived_cache_plan,
    build_ltf_1m_plan,
    resolve_pipeline_requirements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGIME_FLOOR = date(2023, 1, 1)


def _make_layered_window() -> LayeredWindow:
    return LayeredWindow(
        fetch_start=date(2023, 4, 29),
        l1_start=date(2023, 4, 29),
        l2_start=date(2025, 1, 1),
        holdout_start=date(2026, 1, 1),
        holdout_end=date(2026, 6, 30),
        regime_floor=_REGIME_FLOOR,
    )


def _create_parquet(
    path: Path,
    start_ns: int,
    end_ns: int,
    bar_duration_ns: int,
    gap_ranges: list[tuple[int, int]] | None = None,
) -> None:
    """Create a parquet file with synthetic timestamps.

    gap_ranges: list of (start_offset_bars, end_offset_bars) to remove.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(start_ns, end_ns, bar_duration_ns, dtype=np.int64)

    # Apply gap removal
    if gap_ranges:
        mask = np.ones(len(timestamps), dtype=bool)
        for gs, ge in gap_ranges:
            mask[gs:ge] = False
        timestamps = timestamps[mask]

    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="ns", utc=True)})
    df.to_parquet(path, index=False)


def _date_to_ns(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").value)


def _manifest_entry(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
        "first_bar_ns": _date_to_ns(date(2023, 4, 29)),
        "last_bar_ns": _date_to_ns(date(2026, 6, 30)),
        "coverage_ratio": 1.0,
        "max_gap_bars_observed": 0,
        "duplicate_count": 0,
    }


# ---------------------------------------------------------------------------
# Scenario 1: Base cache hit
# ---------------------------------------------------------------------------
class TestBaseCacheHit:
    """When 1h/4h/1d manifest matches unchanged files and full interval, targets must be empty."""

    def test_empty_targets_when_cache_full(self, tmp_path: Path) -> None:
        bar_4h = 14400 * 1_000_000_000
        start_ns = _date_to_ns(date(2023, 4, 29))
        end_ns = _date_to_ns(date(2026, 6, 30)) + bar_4h * 5
        cache_file = tmp_path / "4h" / "BTCUSDT.parquet"
        _create_parquet(cache_file, start_ns, end_ns, bar_4h)

        manifest = {"BTCUSDT::4h": _manifest_entry(cache_file)}

        requirement = DataRequirement(
            timeframe="4h",
            start=date(2023, 4, 29),
            end=date(2026, 6, 30),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest=manifest,
        )
        assert len(plan.targets) == 0


# ---------------------------------------------------------------------------
# Scenario 2: Endpoint shortage
# ---------------------------------------------------------------------------
class TestEndpointShortage:
    """When 4h ends before phase required end, only a 4h target is produced."""

    def test_only_affected_tf_targeted(self, tmp_path: Path) -> None:
        bar_1h = 3600 * 1_000_000_000
        bar_4h = 14400 * 1_000_000_000
        bar_1d = 86400 * 1_000_000_000
        start_ns = _date_to_ns(date(2023, 4, 29))
        end_ns = _date_to_ns(date(2026, 6, 30))

        # Full 1h and 1d coverage
        _create_parquet(tmp_path / "1h" / "BTCUSDT.parquet", start_ns, end_ns + bar_1h * 5, bar_1h)
        _create_parquet(tmp_path / "1d" / "BTCUSDT.parquet", start_ns, end_ns + bar_1d * 5, bar_1d)

        # 4h truncated early (missing last 3 months)
        short_4h_end = _date_to_ns(date(2026, 3, 31))
        _create_parquet(tmp_path / "4h" / "BTCUSDT.parquet", start_ns, short_4h_end, bar_4h)

        manifest: dict[str, object] = {
            "BTCUSDT::1h": _manifest_entry(tmp_path / "1h" / "BTCUSDT.parquet"),
            "BTCUSDT::1d": _manifest_entry(tmp_path / "1d" / "BTCUSDT.parquet"),
        }

        req_1h = DataRequirement(
            timeframe="1h",
            start=date(2023, 4, 29),
            end=date(2026, 6, 30),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        req_4h = DataRequirement(
            timeframe="4h",
            start=date(2023, 4, 29),
            end=date(2026, 6, 30),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        req_1d = DataRequirement(
            timeframe="1d",
            start=date(2022, 10, 1),
            end=date(2026, 6, 30),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )

        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(req_1h, req_4h, req_1d),
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest=manifest,
        )
        tf_targets = {t.timeframe for t in plan.targets}
        assert tf_targets == {"4h"}
        for t in plan.targets:
            assert t.symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# Scenario 3: Internal gap
# ---------------------------------------------------------------------------
class TestInternalGap:
    """When 1h has 7 missing bars and max gap is 6, status must be 'gap'."""

    def test_gap_detected(self, tmp_path: Path) -> None:
        bar_1h = 3600 * 1_000_000_000
        start_ns = _date_to_ns(date(2023, 4, 29))
        end_ns = _date_to_ns(date(2023, 5, 15))
        _create_parquet(
            tmp_path / "1h" / "BTCUSDT.parquet",
            start_ns,
            end_ns + bar_1h * 5,
            bar_1h,
            gap_ranges=[(50, 58)],
        )

        manifest: dict[str, object] = {}
        requirement = DataRequirement(
            timeframe="1h",
            start=date(2023, 4, 29),
            end=date(2023, 5, 15),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest=manifest,
        )
        cov_map = {c.symbol + "::" + c.timeframe: c for c in plan.coverage}
        btc_1h = cov_map.get("BTCUSDT::1h")
        assert btc_1h is not None
        assert btc_1h.status == "gap"
        assert any(t.reason == "gap" for t in plan.targets)


# ---------------------------------------------------------------------------
# Scenario from spec boilerplate: LTF plan admits scope and respects budget
# ---------------------------------------------------------------------------
class TestLtfPlanBudget:
    """Only admitted symbols appear; max symbols budget is respected."""

    def test_admitted_scope_and_budget(self, tmp_path: Path) -> None:
        requirement = DataRequirement(
            timeframe="1m",
            start=date(2023, 4, 29),
            end=date(2026, 6, 30),
            required=False,
            consumer_phases=("l0", "l1", "l2", "l3"),
            min_coverage_ratio=0.80,
        )
        admitted = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
        plan = build_ltf_1m_plan(
            admitted_symbols=admitted,
            universe_priority={"AAAUSDT": 1.0, "BBBUSDT": 1.0, "CCCUSDT": 0.5},
            requirement=requirement,
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            min_coverage_ratio=0.80,
            max_symbols=2,
            manifest={},
        )
        assert plan.selected_symbols == ("AAAUSDT", "BBBUSDT")
        assert set(plan.selected_symbols) <= set(admitted)


# ---------------------------------------------------------------------------
# Scenario from spec boilerplate: derived cache is source-versioned
# ---------------------------------------------------------------------------
class TestDerivedCacheSourceVersioned:
    """Derived cache plan reflects source fingerprint changes."""

    def test_unknown_source_triggers_materialize(self, tmp_path: Path) -> None:
        source_manifest: dict[str, object] = {"AAAUSDT::1h": {"fingerprint": "v1"}}
        first = build_derived_cache_plan(
            symbols=("AAAUSDT",),
            requirements=(),
            enabled_timeframes=("2h", "6h", "8h", "12h"),
            cache_root=tmp_path,
            source_manifest=source_manifest,
        )
        assert all(t.status == "materialize" for t in first.targets)


# ---------------------------------------------------------------------------
# Scenario 10: 1m optional failure must not block base
# ---------------------------------------------------------------------------
class TestLtfOptionalFailure:
    """1m coverage failure must not affect base readiness."""

    def test_ltf_excluded_when_budget_exhausted(self, tmp_path: Path) -> None:
        """When 1m is missing and no budget remains, symbol is excluded."""
        bar_1m = 60 * 1_000_000_000
        start_ns = _date_to_ns(date(2023, 4, 29))
        end_ns = _date_to_ns(date(2026, 6, 30))

        # Two covered symbols
        for sym in ("AAAUSDT", "BBBUSDT"):
            _create_parquet(tmp_path / "1m" / f"{sym}.parquet", start_ns, end_ns + bar_1m * 5, bar_1m)
        CCC = "CCCUSDT"

        manifest: dict[str, object] = {
            "AAAUSDT::1m": _manifest_entry(tmp_path / "1m" / "AAAUSDT.parquet"),
            "BBBUSDT::1m": _manifest_entry(tmp_path / "1m" / "BBBUSDT.parquet"),
        }

        requirement = DataRequirement(
            timeframe="1m",
            start=date(2023, 4, 29),
            end=date(2026, 6, 30),
            required=False,
            consumer_phases=("l0", "l1", "l2", "l3"),
            min_coverage_ratio=0.80,
        )
        plan = build_ltf_1m_plan(
            admitted_symbols=(CCC,),
            universe_priority={CCC: 0.5},
            requirement=requirement,
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            min_coverage_ratio=0.80,
            max_symbols=0,
            manifest=manifest,
        )
        assert len(plan.selected_symbols) == 0
        assert CCC not in plan.covered_symbols


# ---------------------------------------------------------------------------
# Scenario: resolve_pipeline_requirements
# ---------------------------------------------------------------------------
class TestResolvePipelineRequirements:
    """resolve_pipeline_requirements resolves phase-specific requirements."""

    def test_l3_requirements_have_correct_end(self) -> None:
        lw = _make_layered_window()
        reqs = resolve_pipeline_requirements(phase="l3", layered_window=lw)
        assert len(reqs) == 3
        tf_set = {r.timeframe for r in reqs}
        assert tf_set == {"1h", "4h", "1d"}
        for r in reqs:
            assert r.end == date(2026, 6, 30)
            assert r.required is True

    def test_invalid_layered_window_raises(self) -> None:
        with pytest.raises(ValueError, match="layered_window must have"):
            resolve_pipeline_requirements(phase="l3", layered_window=object())


# ---------------------------------------------------------------------------
# Scenario: Empty parquet file
# ---------------------------------------------------------------------------
class TestEmptyParquet:
    """An empty parquet file must produce 'missing' status."""

    def test_empty_file_is_missing(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "1h" / "BTCUSDT.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": []}).to_parquet(cache_file)

        manifest: dict[str, object] = {}
        requirement = DataRequirement(
            timeframe="1h",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest=manifest,
        )
        cov_map = {c.symbol + "::" + c.timeframe: c for c in plan.coverage}
        assert cov_map["BTCUSDT::1h"].status == "missing"


# ---------------------------------------------------------------------------
# Scenario: Lifecycle unavailable interval
# ---------------------------------------------------------------------------
class TestLifecycleUnavailable:
    """When onboard_date > required end, status is unavailable_lifecycle."""

    def test_onboard_after_end_is_unavailable(self, tmp_path: Path) -> None:
        requirement = DataRequirement(
            timeframe="1h",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        lifecycle: dict[str, object] = {
            "BTCUSDT": {"onboard_date": date(2024, 6, 1), "delivery_date": None},
        }
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol=lifecycle,
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest={},
        )
        cov_map = {c.symbol + "::" + c.timeframe: c for c in plan.coverage}
        assert cov_map["BTCUSDT::1h"].status == "unavailable_lifecycle"


# ---------------------------------------------------------------------------
# Scenario: Derived cache ready (fingerprint match)
# ---------------------------------------------------------------------------
class TestDerivedCacheReady:
    """When derived file exists and source fingerprint matches, status is ready."""

    def test_ready_when_fingerprint_matches(self, tmp_path: Path) -> None:
        derived_dir = tmp_path / "derived_ohlcv" / "6h"
        derived_dir.mkdir(parents=True, exist_ok=True)
        derived_file = derived_dir / "BTCUSDT.parquet"
        empty_df = pd.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
        empty_df.to_parquet(derived_file)

        source_manifest: dict[str, object] = {
            "BTCUSDT::1h": {"fingerprint": "abc123"},
            "BTCUSDT::6h": {"source_fingerprint": "abc123"},
        }
        plan = build_derived_cache_plan(
            symbols=("BTCUSDT",),
            requirements=(),
            enabled_timeframes=("6h",),
            cache_root=tmp_path,
            source_manifest=source_manifest,
        )
        assert len(plan.targets) == 1
        assert plan.targets[0].status == "ready"


# ---------------------------------------------------------------------------
# Scenario: build_base_data_plan with unavailable lifecycle in required_failures
# ---------------------------------------------------------------------------
class TestBasePlanUnavailableLifecycleInFailures:
    """Unavailable lifecycle for a required requirement must appear in required_failures."""

    def test_unavailable_lifecycle_is_failure(self, tmp_path: Path) -> None:
        requirement = DataRequirement(
            timeframe="4h",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        lifecycle: dict[str, object] = {
            "BTCUSDT": {"onboard_date": date(2024, 6, 1)},
        }
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol=lifecycle,
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest={},
        )
        assert len(plan.required_failures) > 0


# ---------------------------------------------------------------------------
# Scenario: Manifest non-dict entry fallthrough
# ---------------------------------------------------------------------------
class TestManifestNonDictEntry:
    """A non-dict manifest entry must fall through to file inspection."""

    def test_non_dict_manifest_entry_fallthrough(self, tmp_path: Path) -> None:
        bar_1h = 3600 * 1_000_000_000
        start_ns = int(pd.Timestamp(date(2024, 1, 1), tz="UTC").value)
        end_ns = int(pd.Timestamp(date(2024, 1, 31), tz="UTC").value)
        cache_file = tmp_path / "1h" / "BTCUSDT.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        timestamps = np.arange(start_ns, end_ns, bar_1h, dtype=np.int64)
        df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="ns", utc=True)})
        df.to_parquet(cache_file, index=False)

        manifest: dict[str, object] = {"BTCUSDT::1h": "not_a_dict"}
        requirement = DataRequirement(
            timeframe="1h",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            required=True,
            consumer_phases=("l0", "l1", "l2", "l3"),
        )
        plan = build_base_data_plan(
            symbols=("BTCUSDT",),
            requirements=(requirement,),
            lifecycle_by_symbol={},
            cache_root=tmp_path,
            max_gap_bars=6,
            manifest=manifest,
        )
        cov_map = {c.symbol + "::" + c.timeframe: c for c in plan.coverage}
        assert cov_map["BTCUSDT::1h"].status == "ready"
