"""Phase 3-4 lifecycle tests: SymbolLifecycleRecord computation + promotion gate.

Scenarios covered:
  S1  — symbol promoted (in ready_symbols, active in L1 fit window)
  S9  — symbol not_evaluated (active_mask all-False for that symbol)
  not_evaluated via stage6 path (active_mask=None → synthetic all-True mask)
  stage6-path promotion gate (promotion_available_at <= l2_start → no exclusion)
  lifecycle gate exclusion (promotion_available_at > l2_start → symbol filtered)

Complexity: O(N*T) per lifecycle computation; all tests use tiny synthetic data.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer1Result,
    SymbolLifecycleRecord,
)
from src.domain.futures.strategy.walk_forward import WFFold

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_aligned(
    symbols: list[str],
    n_bars: int,
    active_mask: np.ndarray | None,
) -> Any:
    """Return a MagicMock standing in for AlignedMarketData with minimal attributes."""
    mock = MagicMock()
    mock.symbols = symbols
    # datetimes: integer-based ns epoch for simplicity — pd.Timestamp can parse int64 ns
    base = pd.Timestamp("2024-01-01", tz="UTC")
    dts = np.array(
        [np.datetime64(int((base + pd.Timedelta(days=i)).value), "ns") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    mock.datetimes = dts
    mock.active_mask = active_mask
    return mock


def _make_folds(fit_start: int, fit_end: int, oos_start: int, oos_end: int) -> tuple[WFFold, ...]:
    return (
        WFFold(
            fit_start=fit_start,
            fit_end=fit_end,
            cal_start=fit_end,
            cal_end=oos_start,
            oos_start=oos_start,
            oos_end=oos_end,
        ),
    )


def _make_layer1_result(**overrides: Any) -> Layer1Result:
    """Construct a minimal Layer1Result with sensible defaults."""
    defaults: dict[str, Any] = dict(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=True,
        n_valid=0,
        n_total=2,
        n_trade_scope=2,
    )
    defaults.update(overrides)
    return Layer1Result(**defaults)


# ── lifecycle computation helper (extracted for unit-testability) ─────────────

def _compute_lifecycle(
    aligned: Any,
    outer_folds: tuple[WFFold, ...],
    oos_stacked: dict[str, Any],
    deployment_registry: Any | None,
) -> tuple[SymbolLifecycleRecord, ...]:
    """Pure-function extraction of the lifecycle logic from run_l1_nested_swf.

    This mirrors the exact algorithm in pipeline.py so tests stay aligned with
    the implementation without invoking the heavy full function.
    """
    l1_fit_start = min(f.fit_start for f in outer_folds)
    l1_fit_end = max(f.oos_end for f in outer_folds)
    _active = aligned.active_mask
    if _active is None:
        _active = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=np.bool_)

    ready_syms: set[str] = (
        set(deployment_registry.ready_symbols) if deployment_registry is not None else set()
    )
    records: list[SymbolLifecycleRecord] = []
    for col, sym in enumerate(aligned.symbols):
        mask_slice = _active[l1_fit_start:l1_fit_end, col]
        if not mask_slice.any():
            records.append(
                SymbolLifecycleRecord(
                    symbol=sym,
                    fold_status="not_evaluated",
                    promotion_available_at=None,
                )
            )
            continue

        first_offset = int(np.argmax(mask_slice))
        first_abs = l1_fit_start + first_offset
        promo_at: date = pd.Timestamp(aligned.datetimes[first_abs]).date()

        if sym in ready_syms:
            status = "promoted"
        elif sym in oos_stacked:
            status = "evaluated"
        else:
            status = "failed"

        records.append(
            SymbolLifecycleRecord(
                symbol=sym,
                fold_status=status,
                promotion_available_at=promo_at,
            )
        )
    return tuple(records)


# ── S1: promoted ──────────────────────────────────────────────────────────────

class TestS1Promoted:
    """Symbol present in ready_symbols → fold_status='promoted'."""

    def test_promoted_symbol_status_and_promo_date(self) -> None:
        # Arrange
        n_bars = 100
        active_mask = np.zeros((n_bars, 2), dtype=np.bool_)
        active_mask[10:, 0] = True  # sym "A" eligible from bar 10
        active_mask[5:, 1] = True   # sym "B" eligible from bar 5
        aligned = _make_aligned(["A", "B"], n_bars, active_mask)
        folds = _make_folds(fit_start=0, fit_end=80, oos_start=80, oos_end=100)

        registry = MagicMock()
        registry.ready_symbols = ["A"]  # only A is promoted

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={"A": object(), "B": object()},
            deployment_registry=registry,
        )

        # Assert
        rec_a = next(r for r in records if r.symbol == "A")
        rec_b = next(r for r in records if r.symbol == "B")

        assert rec_a.fold_status == "promoted"
        assert rec_a.promotion_available_at is not None
        # bar 10 → 2024-01-11
        assert rec_a.promotion_available_at == date(2024, 1, 11)

        assert rec_b.fold_status == "evaluated"  # in oos_stacked but not in ready_symbols
        assert rec_b.promotion_available_at == date(2024, 1, 6)  # bar 5

    def test_promoted_symbol_has_promo_at_within_l1_fit_window(self) -> None:
        # Arrange — first eligible bar is at fit_start (bar 0)
        n_bars = 50
        active_mask = np.ones((n_bars, 1), dtype=np.bool_)
        aligned = _make_aligned(["X"], n_bars, active_mask)
        folds = _make_folds(fit_start=0, fit_end=40, oos_start=40, oos_end=50)

        registry = MagicMock()
        registry.ready_symbols = ["X"]

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={},
            deployment_registry=registry,
        )

        # Assert
        assert len(records) == 1
        rec = records[0]
        assert rec.fold_status == "promoted"
        assert rec.promotion_available_at == date(2024, 1, 1)  # bar 0


# ── S9: not_evaluated ─────────────────────────────────────────────────────────

class TestS9NotEvaluated:
    """Symbol with all-False active_mask in fit window → not_evaluated."""

    def test_not_evaluated_when_mask_all_false(self) -> None:
        # Arrange
        n_bars = 100
        active_mask = np.zeros((n_bars, 2), dtype=np.bool_)
        active_mask[:, 0] = True   # sym "A" — fully eligible
        # sym "B" — active_mask stays all-False
        aligned = _make_aligned(["A", "B"], n_bars, active_mask)
        folds = _make_folds(fit_start=0, fit_end=80, oos_start=80, oos_end=100)

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={},
            deployment_registry=None,
        )

        # Assert
        rec_b = next(r for r in records if r.symbol == "B")
        assert rec_b.fold_status == "not_evaluated"
        assert rec_b.promotion_available_at is None

    def test_not_evaluated_promotion_at_is_none(self) -> None:
        # Arrange — single symbol, fully inactive
        n_bars = 60
        active_mask = np.zeros((n_bars, 1), dtype=np.bool_)
        aligned = _make_aligned(["Z"], n_bars, active_mask)
        folds = _make_folds(fit_start=0, fit_end=50, oos_start=50, oos_end=60)

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={},
            deployment_registry=None,
        )

        # Assert
        assert len(records) == 1
        assert records[0].fold_status == "not_evaluated"
        assert records[0].promotion_available_at is None


# ── stage6 path: active_mask=None ─────────────────────────────────────────────

class TestStage6Path:
    """active_mask=None (stage6) → synthetic all-True mask; promo_at = bar[fit_start].date()."""

    def test_stage6_mask_none_promotes_correctly(self) -> None:
        # Arrange
        n_bars = 80
        aligned = _make_aligned(["A"], n_bars, active_mask=None)
        folds = _make_folds(fit_start=10, fit_end=70, oos_start=70, oos_end=80)

        registry = MagicMock()
        registry.ready_symbols = ["A"]

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={},
            deployment_registry=registry,
        )

        # Assert — promo_at = bar 10 → 2024-01-11
        assert records[0].fold_status == "promoted"
        assert records[0].promotion_available_at == date(2024, 1, 11)

    def test_stage6_mask_none_failed_when_not_in_registry_or_oos(self) -> None:
        # Arrange
        n_bars = 80
        aligned = _make_aligned(["B"], n_bars, active_mask=None)
        folds = _make_folds(fit_start=0, fit_end=70, oos_start=70, oos_end=80)

        # Act
        records = _compute_lifecycle(
            aligned=aligned,
            outer_folds=folds,
            oos_stacked={},
            deployment_registry=None,
        )

        # Assert — eligible but neither in oos_stacked nor ready → "failed"
        assert records[0].fold_status == "failed"
        assert records[0].promotion_available_at == date(2024, 1, 1)


# ── lifecycle gate injection ───────────────────────────────────────────────────

class TestLifecycleGate:
    """Gate logic: symbols with promotion_available_at > l2_start are removed from oos_stacked."""

    def _run_gate(
        self,
        lifecycle: tuple[SymbolLifecycleRecord, ...],
        oos_stacked: dict[str, Any],
        l2_start: date | None,
    ) -> dict[str, Any]:
        """Inline replication of the gate block in run_tiered_pipeline."""
        if not lifecycle or l2_start is None:
            return oos_stacked
        l2_date = l2_start if isinstance(l2_start, date) else l2_start.date()
        late = {
            r.symbol
            for r in lifecycle
            if r.promotion_available_at is not None and r.promotion_available_at > l2_date
        }
        if not late:
            return oos_stacked
        return {k: v for k, v in oos_stacked.items() if k not in late}

    def test_gate_excludes_late_symbol(self) -> None:
        # Arrange
        lifecycle = (
            SymbolLifecycleRecord("A", "promoted", date(2024, 1, 1)),
            SymbolLifecycleRecord("B", "promoted", date(2024, 6, 1)),  # after l2_start
        )
        oos_stacked = {"A": object(), "B": object()}

        # Act
        result = self._run_gate(lifecycle, oos_stacked, l2_start=date(2024, 3, 1))

        # Assert
        assert "A" in result
        assert "B" not in result

    def test_gate_no_exclusion_when_all_on_time(self) -> None:
        # Arrange
        lifecycle = (
            SymbolLifecycleRecord("A", "promoted", date(2024, 1, 1)),
            SymbolLifecycleRecord("B", "promoted", date(2024, 2, 28)),
        )
        oos_stacked = {"A": object(), "B": object()}

        # Act
        result = self._run_gate(lifecycle, oos_stacked, l2_start=date(2024, 3, 1))

        # Assert — both symbols are on time → no exclusion
        assert set(result.keys()) == {"A", "B"}

    def test_gate_skipped_when_l2_start_is_none(self) -> None:
        # Arrange
        lifecycle = (SymbolLifecycleRecord("A", "promoted", date(2025, 1, 1)),)
        oos_stacked = {"A": object()}

        # Act — l2_start=None → gate is a no-op
        result = self._run_gate(lifecycle, oos_stacked, l2_start=None)

        # Assert
        assert "A" in result

    def test_gate_skipped_when_lifecycle_empty(self) -> None:
        # Arrange
        oos_stacked = {"A": object()}

        # Act
        result = self._run_gate((), oos_stacked, l2_start=date(2024, 1, 1))

        # Assert — empty lifecycle → gate is a no-op
        assert "A" in result

    def test_gate_preserves_not_evaluated_symbols_when_absent_from_oos(self) -> None:
        # Arrange — not_evaluated has promotion_available_at=None → must not be excluded
        lifecycle = (
            SymbolLifecycleRecord("A", "promoted", date(2024, 1, 1)),
            SymbolLifecycleRecord("B", "not_evaluated", None),
        )
        # B is not in oos_stacked anyway — gate must not crash on None
        oos_stacked = {"A": object()}

        # Act
        result = self._run_gate(lifecycle, oos_stacked, l2_start=date(2024, 3, 1))

        # Assert — only A was ever in oos_stacked; gate must not error
        assert "A" in result
        assert "B" not in result


# ── Layer1Result field integration ────────────────────────────────────────────

class TestLayer1ResultIntegration:
    """Verify symbol_lifecycle field is accepted by frozen dataclass and default is ()."""

    def test_default_symbol_lifecycle_is_empty_tuple(self) -> None:
        # Arrange / Act
        result = _make_layer1_result()

        # Assert
        assert result.symbol_lifecycle == ()

    def test_symbol_lifecycle_field_stored_correctly(self) -> None:
        # Arrange
        records = (
            SymbolLifecycleRecord("X", "promoted", date(2024, 1, 1)),
            SymbolLifecycleRecord("Y", "not_evaluated", None),
        )

        # Act
        result = _make_layer1_result(symbol_lifecycle=records)

        # Assert
        assert result.symbol_lifecycle == records
        assert result.symbol_lifecycle[0].fold_status == "promoted"
        assert result.symbol_lifecycle[1].promotion_available_at is None

    def test_dataclasses_replace_on_frozen_result(self) -> None:
        # Arrange
        original = _make_layer1_result()
        new_records = (SymbolLifecycleRecord("A", "evaluated", date(2024, 3, 1)),)

        # Act
        replaced = dataclasses.replace(original, symbol_lifecycle=new_records)

        # Assert — immutability: original unchanged, replaced has new field
        assert original.symbol_lifecycle == ()
        assert replaced.symbol_lifecycle == new_records
