"""Warmup-aware point-in-time coverage audit deciding feature admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.mhs.features import FeatureSpec


def _anchor_positions(
    spec: FeatureSpec,
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> dict[str, int | None]:
    """First row position with any non-null required input per symbol."""
    combined: pd.DataFrame | None = None
    for column in spec.required_columns:
        panel = panels[column]
        notna = panel.notna()
        combined = notna if combined is None else (combined | notna)
    assert combined is not None
    anchors: dict[str, int | None] = {}
    positions = np.arange(len(mask.index))
    for symbol in mask.columns:
        col_any = combined[symbol].to_numpy(dtype=bool)
        hits = positions[col_any]
        anchors[str(symbol)] = int(hits[0]) if len(hits) else None
    return anchors


def _validate_admission_inputs(
    spec: FeatureSpec,
    feature: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> None:
    if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
        raise ValueError(f"feature '{spec.name}' and mask must be identically indexed and columned")
    for column in spec.required_columns:
        if column not in panels:
            raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {column}")
        panel = panels[column]
        if not panel.index.equals(mask.index) or list(panel.columns) != list(mask.columns):
            raise ValueError(f"required panel '{column}' and mask must be identically indexed and columned")


def feature_admission_coverage(
    spec: FeatureSpec,
    feature: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
) -> dict[int, float]:
    """Per-calendar-year admission coverage of ``feature`` with warmup excluded.

    Audits only rows strictly before ``cutoff`` (all rows when None). A mask
    cell excludes warmup only from the symbol's anchor through the next
    ``spec.warmup_bars`` rows, where the anchor is the first row at which any of
    ``spec.required_columns`` is non-null for that symbol; a symbol with no
    non-null required input has every mask cell auditable (a dead source is a
    gap, never warmup). Per year: zero mask cells -> ``0.0``; mask cells but no
    auditable cell -> year omitted (pure warmup); otherwise covered auditable
    cells / auditable cells. Mask cells before the first observed input count
    as missing, so a future source start cannot rewrite past admission.

    Returns:
        ``{year: coverage}``; empty when no row precedes ``cutoff`` or every
        year is pure warmup.
    Raises:
        ValueError: ``feature``, ``mask`` or a required panel not identically
            indexed and columned, or a required column absent from ``panels``.
    """
    _validate_admission_inputs(spec, feature, panels, mask)
    vectors = _admission_row_vectors(spec, feature, panels, mask)
    rows = np.ones(len(mask.index), dtype=bool) if cutoff is None else np.asarray(mask.index < cutoff)
    return _coverage_from_vectors(*(vector[rows] for vector in vectors))


def _is_admitted(spec: FeatureSpec, coverage: dict[int, float]) -> bool:
    return bool(coverage) and all(cov >= spec.min_coverage for cov in coverage.values())


def _admission_row_vectors(
    spec: FeatureSpec,
    feature: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce auditable/covered counts to per-row vectors once per spec."""
    anchors = _anchor_positions(spec, panels, mask)
    warmup = int(spec.warmup_bars)
    n = len(mask.index)
    positions = np.arange(n)
    mask_np = mask.to_numpy(dtype=bool)
    feat_np = feature.notna().to_numpy(dtype=bool)
    auditable = np.zeros_like(mask_np, dtype=bool)
    for j, symbol in enumerate(mask.columns):
        anchor = anchors[str(symbol)]
        col_mask = mask_np[:, j]
        if anchor is None:
            auditable[:, j] = col_mask
        else:
            auditable[:, j] = col_mask & ((positions < anchor) | (positions >= anchor + warmup))
    covered = auditable & feat_np
    mask_per_row = mask_np.sum(axis=1).astype(np.int64)
    auditable_per_row = auditable.sum(axis=1).astype(np.int64)
    covered_per_row = covered.sum(axis=1).astype(np.int64)
    years = mask.index.year.to_numpy()
    return mask_per_row, auditable_per_row, covered_per_row, years


def _coverage_from_vectors(
    mask_per_row: np.ndarray,
    auditable_per_row: np.ndarray,
    covered_per_row: np.ndarray,
    years: np.ndarray,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for year in sorted(set(years.tolist())):
        sel = years == year
        mask_cells = int(mask_per_row[sel].sum())
        if mask_cells == 0:
            out[int(year)] = 0.0
            continue
        aud = int(auditable_per_row[sel].sum())
        if aud == 0:
            continue
        out[int(year)] = float(int(covered_per_row[sel].sum()) / aud)
    return out
