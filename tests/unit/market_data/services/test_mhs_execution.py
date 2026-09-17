"""Unit isolation for src.market_data.services.mhs_execution manifest sealing."""

from __future__ import annotations

import json


def test_refresh_mhs_execution_manifest_seals_required_inputs(tmp_path) -> None:
    import src.market_data.services.mhs_execution as module

    manifest = tmp_path / "plan.json"
    manifest.write_text(
        json.dumps({"timeframe": "3m", "start": "2025-01-01", "end": "2025-01-02", "symbols": []}),
        encoding="utf-8",
    )
    result = module.refresh_mhs_execution_manifest(manifest)
    attestation = tmp_path / "plan.inputs.json"
    assert attestation.exists()
    assert result["input_manifest_path"] == str(attestation)
    assert isinstance(result["input_manifest_digest"], str)
    assert len(result["input_manifest_digest"]) == 64
    sealed = json.loads(attestation.read_text(encoding="utf-8"))
    assert sealed["digest"] == result["input_manifest_digest"]
    assert sealed["files"] == []


def _write_ohlcv_bars(path, labels) -> None:
    import pandas as pd

    ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    pd.DataFrame({"timestamp": ms.to_numpy(dtype="int64")}).to_parquet(path)


def _write_mark_bars(path, labels, close=100.0) -> None:
    import pandas as pd

    ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    pd.DataFrame(
        {"timestamp": ms.to_numpy(dtype="int64"), "datetime": labels, "close": close}
    ).to_parquet(path)


def test_future_bar_deletion_leaves_past_eligibility_unchanged(tmp_path) -> None:
    """Future deletion: eligibility at or before the cutoff is unchanged."""
    import pandas as pd
    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=201, freq="3min", tz="UTC")
    mask_grid = pd.date_range(grid[0], grid[-1], freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=mask_grid, columns=["S0"])
    cutoff = grid[0] + pd.Timedelta(hours=3)
    root = tmp_path
    (root / "3m").mkdir()
    _write_ohlcv_bars(root / "3m" / "S0.parquet", grid)
    before, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    future_kept = grid[(grid <= cutoff) | (grid >= cutoff + pd.Timedelta(hours=4))]
    _write_ohlcv_bars(root / "3m" / "S0.parquet", future_kept)
    after, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    assert before.loc[cutoff, "S0"]
    assert after.loc[cutoff, "S0"]
    assert before.loc[:cutoff].equals(after.loc[:cutoff])


def test_future_mark_corruption_leaves_past_eligibility_unchanged(tmp_path, monkeypatch) -> None:
    """Future mark changes do not move earlier eligibility."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=200, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    cutoff = grid[50]
    path = tmp_path / "S0.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid)
    before, _ = apply_dynamic_mark_gap_exclusion(mask)
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    frame = table.to_pandas()
    frame.loc[frame.index[100:], "close"] = -5.0
    frame.to_parquet(path)
    after, _ = apply_dynamic_mark_gap_exclusion(mask)
    assert before.loc[:cutoff].equals(after.loc[:cutoff])
    assert bool(after.loc[cutoff, "S0"])


def test_observed_threshold_starts_at_threshold_not_last_bar(tmp_path) -> None:
    """Restriction begins at last-observed plus threshold; the bar stays usable."""
    import pandas as pd
    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=201, freq="3min", tz="UTC")
    hole = grid[(grid < pd.Timestamp("2022-01-01 04:00", tz="UTC")) | (grid >= pd.Timestamp("2022-01-01 06:00", tz="UTC"))]
    root = tmp_path
    (root / "3m").mkdir()
    _write_ohlcv_bars(root / "3m" / "S0.parquet", hole)
    mask_grid = pd.date_range(grid[0], grid[-1], freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=mask_grid, columns=["S0"])
    adjusted, excluded = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    assert bool(adjusted.loc["2022-01-01 04:00", "S0"])
    assert not bool(adjusted.loc["2022-01-01 05:00", "S0"])
    assert not bool(adjusted.loc["2022-01-01 06:00", "S0"])
    assert bool(adjusted.loc["2022-01-01 07:00", "S0"])
    assert excluded["S0"] == (
        (pd.Timestamp("2022-01-01 05:00", tz="UTC"), pd.Timestamp("2022-01-01 06:00", tz="UTC")),
    )


def test_trailing_gap_detected_causally_and_recovery_restores(tmp_path) -> None:
    """A stopped source excludes only past-threshold decisions; resume restores."""
    import pandas as pd
    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=121, freq="3min", tz="UTC")
    end = grid[60]
    root = tmp_path
    (root / "3m").mkdir()
    _write_ohlcv_bars(root / "3m" / "S0.parquet", grid[:61])
    mask_grid = pd.date_range(grid[0], grid[-1], freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=mask_grid, columns=["S0"])
    stopped, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    assert bool(stopped.loc[end, "S0"])
    assert not bool(stopped.loc[end + pd.Timedelta(hours=2), "S0"])
    _write_ohlcv_bars(root / "3m" / "S0.parquet", grid)
    resumed, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    assert bool(resumed.to_numpy().all())
    assert stopped.loc[:end].equals(resumed.loc[:end])


def test_gap_exclusion_rejects_bad_interval_and_threshold(tmp_path) -> None:
    """Unsupported timeframe or threshold fails closed with ValueError."""
    import pandas as pd
    import pytest
    from src.market_data.services.mhs_execution import (
        apply_dynamic_gap_exclusion,
        apply_dynamic_mark_gap_exclusion,
    )

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    with pytest.raises(ValueError, match=r".+"):
        apply_dynamic_gap_exclusion(mask, "7m", root=str(tmp_path))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r".+"):
            apply_dynamic_gap_exclusion(mask, "3m", root=str(tmp_path), min_gap_hours=bad)
        with pytest.raises(ValueError, match=r".+"):
            apply_dynamic_mark_gap_exclusion(mask, min_gap_hours=bad)
    with pytest.raises(ValueError, match=r".+"):
        apply_dynamic_mark_gap_exclusion(mask, timeframe="3m")


def test_gap_exclusion_unreadable_source_raises(tmp_path, monkeypatch) -> None:
    """Corrupt execution/mark sources fail provenance, not silent exclusion."""
    import pandas as pd
    import pytest
    import src.market_data.services.futures_collection as fc
    from src.common.errors import DataIntegrityError
    from src.market_data.services.mhs_execution import (
        apply_dynamic_gap_exclusion,
        apply_dynamic_mark_gap_exclusion,
    )

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    (root / "3m" / "S0.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_dynamic_gap_exclusion(mask, "3m", root=str(tmp_path / "ohlcv"))
    mark_path = tmp_path / "mark.parquet"
    mark_path.write_bytes(b"not a parquet file")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: mark_path)
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_dynamic_mark_gap_exclusion(mask)


def test_mark_gap_exclusion_root_has_no_isolation_effect(tmp_path, monkeypatch) -> None:
    """The mark root argument never changes resolution or results."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=30, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    path = tmp_path / "S0.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid)
    first, _ = apply_dynamic_mark_gap_exclusion(mask, root=str(tmp_path / "a"))
    second, _ = apply_dynamic_mark_gap_exclusion(mask, root=str(tmp_path / "b"))
    assert first.equals(second)


def test_gap_exclusion_rejects_non_numeric_threshold() -> None:
    """A non-numeric threshold is unsupported, not a raw TypeError."""
    import pandas as pd
    import pytest
    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    with pytest.raises(ValueError, match=r".+"):
        apply_dynamic_gap_exclusion(mask, "3m", min_gap_hours="bad")


def test_missing_source_file_excludes_everything(tmp_path, monkeypatch) -> None:
    """An absent source is a whole-run gap, reported for diagnostics."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import (
        apply_dynamic_gap_exclusion,
        apply_dynamic_mark_gap_exclusion,
    )

    grid = pd.date_range("2022-01-01", periods=5, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    adjusted, excluded = apply_dynamic_gap_exclusion(mask, "3m", root=str(root))
    assert not bool(adjusted.to_numpy().any())
    assert excluded["S0"] == ((grid[0], grid[-1]),)
    missing = tmp_path / "absent.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: missing)
    adjusted, excluded = apply_dynamic_mark_gap_exclusion(mask)
    assert not bool(adjusted.to_numpy().any())
    assert excluded["S0"] == ((grid[0], grid[-1]),)


def test_ohlcv_source_without_timestamp_raises(tmp_path) -> None:
    """A timestamp-less execution file fails provenance."""
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(root / "3m" / "S0.parquet")
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_dynamic_gap_exclusion(mask, "3m", root=str(root))


def test_mark_source_without_close_raises(tmp_path, monkeypatch) -> None:
    """A close-less mark file fails mark provenance."""
    import pandas as pd
    import pytest
    import src.market_data.services.futures_collection as fc
    from src.common.errors import DataIntegrityError
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    path = tmp_path / "noclose.parquet"
    pd.DataFrame({"timestamp": [1640995200000]}).to_parquet(path)
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_dynamic_mark_gap_exclusion(mask)


def test_mark_timestamp_only_file_derives_labels(tmp_path, monkeypatch) -> None:
    """Without a datetime column, mark labels derive from millisecond stamps."""
    import numpy as np
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=10, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    path = tmp_path / "tsclose.parquet"
    ms = (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    pd.DataFrame({"timestamp": ms.to_numpy(dtype="int64"), "close": 100.0}).to_parquet(path)
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    adjusted, excluded = apply_dynamic_mark_gap_exclusion(mask)
    assert bool(adjusted.to_numpy().all())
    assert excluded == {}
    assert np.array_equal(adjusted.to_numpy(), mask.to_numpy())


def test_mark_source_without_time_columns_raises(tmp_path, monkeypatch) -> None:
    """A mark file with neither datetime nor timestamp fails provenance."""
    import pandas as pd
    import pytest
    import src.market_data.services.futures_collection as fc
    from src.common.errors import DataIntegrityError
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    mask = pd.DataFrame(
        True,
        index=pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        columns=["S0"],
    )
    path = tmp_path / "notime.parquet"
    pd.DataFrame({"close": [100.0, 101.0]}).to_parquet(path)
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_dynamic_mark_gap_exclusion(mask)


def test_leading_span_below_threshold_stays_eligible(tmp_path, monkeypatch) -> None:
    """A late listing inside the threshold keeps early decisions eligible."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=300, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    path = tmp_path / "late.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid[100:])
    adjusted, _ = apply_dynamic_mark_gap_exclusion(mask, min_gap_hours=720.0)
    assert bool(adjusted.loc[grid[50], "S0"])
    assert bool(adjusted.loc[grid[150], "S0"])


def test_leading_span_above_threshold_excluded_until_first_bar(tmp_path, monkeypatch) -> None:
    """A leading absence past the threshold excludes until bars are published."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=2000, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    path = tmp_path / "late.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid[800:])
    adjusted, excluded = apply_dynamic_mark_gap_exclusion(mask, min_gap_hours=720.0)
    assert not bool(adjusted.loc[grid[100], "S0"])
    assert not bool(adjusted.loc[grid[799], "S0"])
    assert bool(adjusted.loc[grid[801], "S0"])
    assert excluded["S0"][0][0] == grid[0]
