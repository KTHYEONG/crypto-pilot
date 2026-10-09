"""Per-symbol funding charge attribution for one streamed replay."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


class FundingChunk:
    """Daily charge buffer for a window, with per-column bar-sized scratch."""

    def __init__(self, grid: pd.DatetimeIndex, columns: Sequence[str], p0: int) -> None:
        days, self._positions = np.unique(grid[p0:].normalize(), return_inverse=True)
        self.days = pd.DatetimeIndex(days)
        self.columns = list(columns)
        self.charges = np.zeros((len(days), len(columns)), dtype="float64")
        self._p0 = p0

    def add_column(self, j: int, charged: npt.NDArray[np.float64]) -> None:
        """Accumulate one symbol's kept bar charges in observation order."""
        np.add.at(self.charges[:, j], self._positions, charged[self._p0:])


class FundingAttribution:
    """Per-symbol funding charge totals and UTC-daily attribution for one replay; same sign convention as the ledger's `funding_charge` (negative is income)."""

    def __init__(self, columns: Sequence[str]) -> None:
        self._totals: dict[str, float] = dict.fromkeys(columns, 0.0)
        self._daily_chunks: list[pd.DataFrame] = []

    def add_chunk(
        self,
        grid: pd.DatetimeIndex,
        local_cols: Sequence[str],
        charged: npt.NDArray[np.float64],
        p0: int,
    ) -> None:
        """Bucket one window chunk's kept per-bar charges into daily per-symbol frames."""
        chunk = self.begin_chunk(grid, local_cols, p0)
        for j in range(len(local_cols)):
            chunk.add_column(j, charged[:, j])
        self.commit_chunk(chunk)

    def begin_chunk(self, grid: pd.DatetimeIndex, local_cols: Sequence[str], p0: int) -> FundingChunk:
        """Create a daily-sized buffer without materializing a bar-by-symbol charge plane."""
        return FundingChunk(grid, local_cols, p0)

    def commit_chunk(self, chunk: FundingChunk) -> None:
        """Retain the completed daily frame and update per-symbol totals."""
        for j, sym in enumerate(chunk.columns):
            self._totals[sym] = float(self._totals[sym] + float(chunk.charges[:, j].sum()))
        self._daily_chunks.append(pd.DataFrame(chunk.charges, index=chunk.days, columns=chunk.columns))

    def totals(self) -> dict[str, float]:
        """Copy of per-symbol charge totals in canonical column order."""
        return dict(self._totals)

    def daily_frame(self, columns: Sequence[str]) -> pd.DataFrame:
        """UTC-indexed daily per-symbol charge frame in canonical order, float64."""
        daily = pd.concat(self._daily_chunks)
        daily = daily.groupby(daily.index).sum()
        daily = daily.reindex(columns=list(columns), fill_value=0.0).astype("float64")
        daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
        return daily
