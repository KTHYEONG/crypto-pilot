from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.contracts import FoldSpec, LongMatrixDataset
from src.domain.futures.strategy.inference import assemble_alpha_panel, infer_fold_alpha


def test_assemble_alpha_panel_builds_expected_columns() -> None:
    dt = np.array(
        [np.datetime64("2024-01-01T00:00:00"), np.datetime64("2024-01-01T04:00:00")],
        dtype="datetime64[ns]",
    )
    symbols = ("BTCUSDT", "ETHUSDT")
    ev_grid = np.array([[0.02, -0.03], [-0.01, 0.04]], dtype=np.float32)

    panel = assemble_alpha_panel(datetimes=dt, symbols=symbols, ev_grid=ev_grid, clip_abs=0.05)

    assert list(panel.columns) == ["alpha_long", "alpha_short"]
    assert float(panel.loc[(dt[0], "BTCUSDT"), "alpha_long"]) > 0.0
    assert float(panel.loc[(dt[0], "ETHUSDT"), "alpha_short"]) > 0.0


def test_assemble_alpha_panel_raises_on_non_finite() -> None:
    dt = np.array([np.datetime64("2024-01-01T00:00:00")], dtype="datetime64[ns]")
    symbols = ("BTCUSDT",)
    ev_grid = np.array([[np.nan]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="non-finite"):
        assemble_alpha_panel(datetimes=dt, symbols=symbols, ev_grid=ev_grid, clip_abs=0.1)


def test_assemble_alpha_panel_validates_metadata_non_finite() -> None:
    dt = np.array([np.datetime64("2024-01-01T00:00:00")], dtype="datetime64[ns]")
    symbols = ("BTCUSDT",)
    ev_grid = np.array([[0.01]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="metadata contains non-finite values"):
        assemble_alpha_panel(
            datetimes=dt,
            symbols=symbols,
            ev_grid=ev_grid,
            clip_abs=0.1,
            forecast_metadata={
                "q10_long": np.array([0.0], dtype=np.float32),
                "q50_long": np.array([np.nan], dtype=np.float32),
            },
        )


def test_infer_fold_alpha_maps_rows_back_to_grid() -> None:
    fold = FoldSpec(
        fold_id=3,
        train_start=0,
        train_end=1,
        valid_start=1,
        valid_end=2,
        test_start=2,
        test_end=3,
        purge_bars=1,
        embargo_bars=1,
    )
    ds = LongMatrixDataset(
        X=np.zeros((2, 1), dtype=np.float32),
        y_rank=np.zeros((2,), dtype=np.int32),
        y_ev=np.zeros((2,), dtype=np.float32),
        group=np.array([2], dtype=np.int32),
        sample_weight=np.ones((2,), dtype=np.float32),
        index_map=np.array([[2, 0], [2, 1]], dtype=np.int64),
        feature_names=("f0",),
    )
    out = infer_fold_alpha(
        fold=fold,
        test=ds,
        ev_test=np.array([0.01, -0.02], dtype=np.float32),
        t_size=4,
        n_size=3,
    )
    assert out.fold_id == 3
    assert float(out.ev_grid[2, 0]) == pytest.approx(0.01)
    assert float(out.ev_grid[2, 1]) == pytest.approx(-0.02)
