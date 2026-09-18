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
        (pd.Timestamp("2022-01-01 00:00", tz="UTC"), pd.Timestamp("2022-01-01 00:00", tz="UTC")),
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
    assert not bool(resumed.loc[grid[0], "S0"])
    assert bool(resumed.iloc[1:].to_numpy().all())
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
    assert not bool(adjusted.loc[grid[0], "S0"])
    assert bool(adjusted.loc[grid[1]:].to_numpy().all())
    assert excluded == {"S0": ((grid[0], grid[0]),)}
    assert np.array_equal(adjusted.loc[grid[1]:].to_numpy(), mask.loc[grid[1]:].to_numpy())


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
    """No published observation excludes, even inside the threshold (corrected baseline)."""
    import pandas as pd
    import src.market_data.services.futures_collection as fc
    from src.market_data.services.mhs_execution import apply_dynamic_mark_gap_exclusion

    grid = pd.date_range("2022-01-01", periods=300, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0"])
    path = tmp_path / "late.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid[100:])
    adjusted, _ = apply_dynamic_mark_gap_exclusion(mask, min_gap_hours=720.0)
    assert not bool(adjusted.loc[grid[50], "S0"])
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


def _causal_ns(hours: float = 0.0, minutes: float = 0.0) -> int:
    return int(hours * 3_600_000_000_000 + minutes * 60_000_000_000)


def test_causal_gap_future_only_first_labels_exclude_identically() -> None:
    """Two future-only first labels exclude the same past decision."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    decisions = np.array([_causal_ns(hours=10)], dtype="int64")
    member = np.array([True])
    early = np.array([decisions[0] + _causal_ns(minutes=30)], dtype="int64")
    late = np.array([decisions[0] + _causal_ns(hours=2)], dtype="int64")
    assert _causal_gap_excluded(member, decisions, early, lag, gap).tolist() == [True]
    assert _causal_gap_excluded(member, decisions, late, lag, gap).tolist() == [True]


def test_causal_gap_missing_sources_match_future_only_masks() -> None:
    """Absent, empty and future-only sources produce identical past masks."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    decisions = np.arange(5, dtype="int64") * _causal_ns(hours=1)
    member = np.ones(5, dtype=bool)
    future = np.array([decisions[-1] + _causal_ns(hours=2)], dtype="int64")
    absent = _causal_gap_excluded(member, decisions, None, lag, gap)
    empty = _causal_gap_excluded(member, decisions, np.zeros(0, dtype="int64"), lag, gap)
    future_only = _causal_gap_excluded(member, decisions, future, lag, gap)
    assert absent.tolist() == [True] * 5
    assert empty.tolist() == absent.tolist()
    assert future_only.tolist() == absent.tolist()


def test_causal_gap_publication_at_fence_is_usable() -> None:
    """A label published exactly at T is observable, subject to the gap threshold."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    decisions = np.array([lag], dtype="int64")
    member = np.array([True])
    labels = np.array([0], dtype="int64")
    assert _causal_gap_excluded(member, decisions, labels, lag, gap).tolist() == [False]


def test_causal_gap_publication_after_fence_is_unusable() -> None:
    """A label published just after T cannot clear the exclusion."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    decisions = np.array([lag - 1], dtype="int64")
    member = np.array([True])
    labels = np.array([0], dtype="int64")
    assert _causal_gap_excluded(member, decisions, labels, lag, gap).tolist() == [True]


def test_causal_gap_trailing_absence_threshold_boundary() -> None:
    """Absence below the threshold stays eligible; equality and above exclude."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    labels = np.array([0], dtype="int64")
    decisions = np.array([gap - 1, gap, gap + 1], dtype="int64")
    member = np.ones(3, dtype=bool)
    assert _causal_gap_excluded(member, decisions, labels, lag, gap).tolist() == [
        False,
        True,
        True,
    ]


def test_causal_gap_future_perturbation_leaves_prefix_masks_unchanged() -> None:
    """Altered future recovery, tail and file end never move masks through T."""
    import numpy as np
    import pandas as pd

    from src.market_data.services.mhs_execution import (
        _apply_causal_gap_exclusion,
        _causal_gap_excluded,
        roster_membership_intervals,
    )

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    prefix = np.arange(10, dtype="int64") * _causal_ns(minutes=3)
    cutoff = prefix[-1] + lag
    decisions = prefix + lag
    member = np.ones(len(decisions), dtype=bool)
    recovery_a = np.concatenate([prefix, np.array([cutoff + gap * 2], dtype="int64")])
    recovery_b = np.concatenate([prefix, np.array([cutoff + gap * 5], dtype="int64")])
    assert (
        _causal_gap_excluded(member, decisions, recovery_a, lag, gap).tolist()
        == _causal_gap_excluded(member, decisions, recovery_b, lag, gap).tolist()
    )
    grid = pd.DatetimeIndex(
        pd.to_datetime(decisions, unit="ns", utc=True), name="decision"
    )
    frame = pd.DataFrame(True, index=grid, columns=["S0"])
    intervals = roster_membership_intervals(frame)
    first, _ = _apply_causal_gap_exclusion(frame, intervals, {"S0": recovery_a}, lag, gap)
    second, _ = _apply_causal_gap_exclusion(frame, intervals, {"S0": recovery_b}, lag, gap)
    assert first.equals(second)


def test_causal_gap_reentry_without_observation_stays_excluded() -> None:
    """Re-entered membership with no published bar still excludes new exposure."""
    import numpy as np

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    decisions = np.arange(6, dtype="int64") * _causal_ns(hours=1)
    member = np.array([True, True, False, False, True, True])
    future = np.array([decisions[-1] + _causal_ns(hours=2)], dtype="int64")
    assert _causal_gap_excluded(member, decisions, future, lag, gap).tolist() == [True] * 6


def test_causal_gap_empty_output_and_malformed_inputs() -> None:
    """Empty aligned inputs succeed; mismatched shapes and lags raise ValueError."""
    import numpy as np
    import pytest

    from src.market_data.services.mhs_execution import _causal_gap_excluded

    lag = _causal_ns(minutes=3)
    gap = _causal_ns(hours=1)
    empty_member = np.zeros(0, dtype=bool)
    empty_decisions = np.zeros(0, dtype="int64")
    out = _causal_gap_excluded(empty_member, empty_decisions, None, lag, gap)
    assert out.dtype == bool
    assert out.tolist() == []
    good_member = np.ones(2, dtype=bool)
    good_decisions = np.arange(2, dtype="int64")
    with pytest.raises(ValueError, match=r".+"):
        _causal_gap_excluded(good_member[:1], good_decisions, None, lag, gap)
    with pytest.raises(ValueError, match=r".+"):
        _causal_gap_excluded(
            good_member, good_decisions, np.zeros((2, 2), dtype="int64"), lag, gap
        )
    for bad_lag, bad_gap in ((0, gap), (-1, gap), (lag, 0), (lag, -5), ("bad", gap)):
        with pytest.raises(ValueError, match=r".+"):
            _causal_gap_excluded(good_member, good_decisions, None, bad_lag, bad_gap)


def _ns_of(labels) -> object:
    import numpy as np
    import pandas as pd

    return np.asarray(
        pd.DatetimeIndex(labels).as_unit("ns").asi8, dtype="int64"
    )


def test_read_ohlcv_labels_projects_timestamp_only(tmp_path, monkeypatch) -> None:
    """Wide multi-group files decode only the timestamp plane with one worker."""
    import pandas as pd
    import pyarrow.parquet as pq
    from src.market_data.services import mhs_execution as mod

    grid = pd.date_range("2022-01-01", periods=12, freq="3min", tz="UTC")
    ms = (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    frame = pd.DataFrame(
        {
            "timestamp": ms.to_numpy(dtype="int64"),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        }
    )
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    path = root / "3m" / "S0.parquet"
    frame.to_parquet(path, row_group_size=4)
    seen: dict[str, object] = {}

    real_file = pq.ParquetFile
    orig_read = pq.ParquetFile.read_row_group

    def spy_read(self, i, columns=None, use_threads=True, **kwargs):
        seen["columns"] = list(columns or [])
        seen["threads"] = use_threads
        return orig_read(self, i, columns=columns, use_threads=use_threads, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read)
    monkeypatch.setattr(
        pq, "read_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("full read"))
    )
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(AssertionError("full read"))
    )
    out = mod._read_ohlcv_labels("S0", "3m", str(root))
    assert seen["columns"] == ["timestamp"]
    assert seen["threads"] is False
    assert out is not None
    assert out.tolist() == _ns_of(grid).tolist()
    assert real_file is not None


def test_read_mark_labels_projects_time_and_close(tmp_path, monkeypatch) -> None:
    """Datetime precedence holds and unrelated mark columns stay undecoded."""
    import pandas as pd
    import pyarrow.parquet as pq
    from src.market_data.services import mhs_execution as mod
    import src.market_data.services.futures_collection as fc

    grid = pd.date_range("2022-01-01", periods=9, freq="1h", tz="UTC")
    ms = (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    path = tmp_path / "M.parquet"
    pd.DataFrame(
        {
            "timestamp": ms.to_numpy(dtype="int64"),
            "datetime": grid,
            "close": 100.0,
            "open": 1.0,
            "funding": 0.01,
        }
    ).to_parquet(path, row_group_size=4)
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    seen: dict[str, object] = {}
    orig_read = pq.ParquetFile.read_row_group

    def spy_read(self, i, columns=None, use_threads=True, **kwargs):
        seen["columns"] = list(columns or [])
        seen["threads"] = use_threads
        return orig_read(self, i, columns=columns, use_threads=use_threads, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read)
    out = mod._read_mark_labels("M", "1h")
    assert seen["columns"] == ["datetime", "close"]
    assert seen["threads"] is False
    assert out is not None
    assert out.tolist() == _ns_of(grid).tolist()


def test_bounded_scan_skips_future_row_groups(tmp_path, monkeypatch) -> None:
    """Trusted statistics after the useful bound are never decoded."""
    import pandas as pd
    import pyarrow.parquet as pq
    from src.market_data.services import mhs_execution as mod

    grid = pd.date_range("2022-01-01", periods=24, freq="3min", tz="UTC")
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    path = root / "3m" / "S0.parquet"
    _write_ohlcv_bars(path, grid)
    import pyarrow as pa

    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=6)
    bound = _ns_of(grid[:6])[-1]
    decoded: list[int] = []
    orig_read = pq.ParquetFile.read_row_group

    def spy_read(self, i, columns=None, use_threads=False, **kwargs):
        decoded.append(i)
        return orig_read(self, i, columns=columns, use_threads=use_threads, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read)
    out = mod._read_ohlcv_labels("S0", "3m", str(root), observed_through_ns=int(bound))
    assert out is not None
    assert out.tolist() == _ns_of(grid[:6]).tolist()
    assert 1 not in decoded
    assert 2 not in decoded
    assert 3 not in decoded
    assert decoded == [0]
    assert pa is not None


def test_bounded_scan_without_statistics_preserves_labels(tmp_path, monkeypatch) -> None:
    """Missing statistics fall back to bounded scanning with exact labels."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from src.market_data.services import mhs_execution as mod

    grid = pd.date_range("2022-01-01", periods=10, freq="3min", tz="UTC")
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    path = root / "3m" / "S0.parquet"
    _write_ohlcv_bars(path, grid)
    full = mod._read_ohlcv_labels("S0", "3m", str(root))

    class _Stats:
        has_min_max = False
        min = 0
        max = 0

    class _Col:
        is_stats_set = False
        statistics = _Stats()

    monkeypatch.setattr(mod, "_row_group_ms_min_ns", lambda column: None)
    probe = mod._row_group_ms_min_ns(_Col())
    assert probe is None
    bound = _ns_of(grid)[-1]
    bounded = mod._read_ohlcv_labels("S0", "3m", str(root), observed_through_ns=int(bound))
    assert full is not None
    assert bounded is not None
    assert np.array_equal(full, bounded)
    assert pq is not None
    assert pd is not None


def test_bounded_scan_preserves_predecessor_for_first_decision(tmp_path) -> None:
    """Predecessor history before the first decision keeps the trailing gap."""
    import pandas as pd
    from src.market_data.services.mhs_execution import (
        _apply_causal_gap_exclusion,
        _read_ohlcv_labels,
        apply_dynamic_gap_exclusion,
        roster_membership_intervals,
    )

    grid = pd.date_range("2022-01-01", periods=300, freq="3min", tz="UTC")
    history = grid[:100]
    resumed = grid[250:]
    labels_grid = history.append(resumed)
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    _write_ohlcv_bars(root / "3m" / "S0.parquet", labels_grid)
    mask_grid = pd.date_range(grid[200], grid[260], freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=mask_grid, columns=["S0"])
    bounded, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    decision_ns = _ns_of(mask_grid)
    full_labels = _read_ohlcv_labels("S0", "3m", str(root))
    oracle, _ = _apply_causal_gap_exclusion(
        mask,
        roster_membership_intervals(mask),
        {"S0": full_labels},
        3 * 60_000_000_000,
        int(1.0 * 3_600_000_000_000),
    )
    assert decision_ns is not None
    assert bounded.equals(oracle)


def test_read_ohlcv_labels_deduplicates_unordered(tmp_path) -> None:
    """Duplicate unordered labels across groups collapse to sorted unique."""
    import numpy as np
    import pandas as pd
    from src.market_data.services.mhs_execution import _read_ohlcv_labels

    grid = pd.date_range("2022-01-01", periods=8, freq="3min", tz="UTC")
    mixed = list(grid[[3, 1, 3, 0, 7, 2, 1, 5]])
    ms = (pd.DatetimeIndex(mixed) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(
        milliseconds=1
    )
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    pd.DataFrame({"timestamp": ms.to_numpy(dtype="int64")}).to_parquet(
        root / "3m" / "S0.parquet", row_group_size=3
    )
    out = _read_ohlcv_labels("S0", "3m", str(root))
    assert out is not None
    assert np.array_equal(out, np.unique(_ns_of(grid[[0, 1, 2, 3, 5, 7]])))


def test_read_mark_labels_rejects_invalid_closes(tmp_path, monkeypatch) -> None:
    """NaN, zero, negative and infinite closes never count as valid marks."""
    import numpy as np
    import pandas as pd
    from src.market_data.services import mhs_execution as mod
    import src.market_data.services.futures_collection as fc

    grid = pd.date_range("2022-01-01", periods=5, freq="1h", tz="UTC")
    path = tmp_path / "M.parquet"
    pd.DataFrame(
        {"datetime": grid, "close": [float("nan"), 0.0, -3.0, float("inf"), float("-inf")]}
    ).to_parquet(path)
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    out = mod._read_mark_labels("M", "1h")
    assert out is not None
    assert len(out) == 0
    assert np is not None


def test_public_masks_hold_single_symbol_history(tmp_path, monkeypatch) -> None:
    """Reused decode buffers still yield per-symbol gaps, proving bounded lifetime."""
    import numpy as np
    import pandas as pd
    from src.market_data.services import mhs_execution as mod

    grid = pd.date_range("2022-01-01", periods=6, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=grid, columns=["S0", "S1"])
    steady = _ns_of(grid)
    gapped = _ns_of(grid[:2])
    shared = {"buf": np.zeros(0, dtype="int64")}

    def fake_read(symbol, timeframe, root, *, observed_through_ns=None):
        want = steady if symbol == "S0" else gapped
        if observed_through_ns is not None:
            want = want[want <= int(observed_through_ns)]
        shared["buf"] = np.asarray(want, dtype="int64")
        return shared["buf"]

    monkeypatch.setattr(mod, "_read_ohlcv_labels", fake_read)
    monkeypatch.setattr(
        mod, "_apply_causal_gap_exclusion", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retained"))
    )
    adjusted, _ = mod.apply_dynamic_gap_exclusion(mask, "1h", root=str(tmp_path), min_gap_hours=2.0)
    assert bool(adjusted.loc[grid[-1], "S0"])
    assert not bool(adjusted.loc[grid[-1], "S1"])


def test_bounded_provenance_errors_carry_symbol_and_path(tmp_path, monkeypatch) -> None:
    """Absence is None; corrupt or schema-invalid sources name symbol and path."""
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.market_data.services import mhs_execution as mod
    import src.market_data.services.futures_collection as fc

    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    assert mod._read_ohlcv_labels("ABSENT", "3m", str(root)) is None
    missing = tmp_path / "missing.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: missing)
    assert mod._read_mark_labels("ABSENT", "1h") is None
    bad = root / "3m" / "BAD.parquet"
    bad.write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError, match=r"BAD"):
        mod._read_ohlcv_labels("BAD", "3m", str(root))
    noschema = root / "3m" / "NOSCHEMA.parquet"
    pd.DataFrame({"close": [1.0]}).to_parquet(noschema)
    with pytest.raises(DataIntegrityError, match=r"NOSCHEMA"):
        mod._read_ohlcv_labels("NOSCHEMA", "3m", str(root))
    mark_bad = tmp_path / "mark_bad.parquet"
    mark_bad.write_bytes(b"not a parquet file")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: mark_bad)
    with pytest.raises(DataIntegrityError, match=r"S0"):
        mod._read_mark_labels("S0", "1h")
    for name, frame in (
        ("NO_CLOSE", pd.DataFrame({"datetime": pd.date_range("2022-01-01", periods=2, tz="UTC")})),
        ("NO_TIME", pd.DataFrame({"close": [1.0, 2.0]})),
        ("EMPTY", pd.DataFrame({"datetime": [], "close": []})),
    ):
        p = tmp_path / f"{name}.parquet"
        frame.to_parquet(p)
        monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe, _p=p: _p)
        with pytest.raises(DataIntegrityError, match=r".+"):
            mod._read_mark_labels("S0", "1h")


def test_bounded_masks_match_full_history_oracle(tmp_path, monkeypatch) -> None:
    """Hourly, three-minute and mark bounded masks equal the corrected oracle."""
    import numpy as np
    import pandas as pd
    from src.market_data.services.mhs_execution import (
        _apply_causal_gap_exclusion,
        _read_mark_labels,
        _read_ohlcv_labels,
        apply_dynamic_gap_exclusion,
        apply_dynamic_mark_gap_exclusion,
        roster_membership_intervals,
    )
    import src.market_data.services.futures_collection as fc

    grid3 = pd.date_range("2022-01-01", periods=400, freq="3min", tz="UTC")
    hole3 = grid3[(grid3 < grid3[100]) | (grid3 >= grid3[200])]
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    (root / "1h").mkdir(parents=True)
    _write_ohlcv_bars(root / "3m" / "S0.parquet", hole3)
    grid1 = pd.date_range("2022-01-01", periods=60, freq="1h", tz="UTC")
    hole1 = grid1[(grid1 < grid1[10]) | (grid1 >= grid1[30])]
    _write_ohlcv_bars(root / "1h" / "S0.parquet", hole1)
    mark_path = tmp_path / "mark.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: mark_path)
    _write_mark_bars(mark_path, grid1[(grid1 < grid1[10]) | (grid1 >= grid1[30])])
    mask3 = pd.DataFrame(True, index=pd.date_range(grid3[0], grid3[-1], freq="1h", tz="UTC"), columns=["S0"])
    bounded3, excl3 = apply_dynamic_gap_exclusion(mask3, "3m", root=str(root), min_gap_hours=1.0)
    oracle3, oracle_excl3 = _apply_causal_gap_exclusion(
        mask3,
        roster_membership_intervals(mask3),
        {"S0": _read_ohlcv_labels("S0", "3m", str(root))},
        3 * 60_000_000_000,
        int(1.0 * 3_600_000_000_000),
    )
    assert bounded3.equals(oracle3)
    assert excl3 == oracle_excl3
    mask1 = pd.DataFrame(True, index=grid1, columns=["S0"])
    bounded1, _ = apply_dynamic_gap_exclusion(mask1, "1h", root=str(root), min_gap_hours=1.0)
    oracle1, _ = _apply_causal_gap_exclusion(
        mask1,
        roster_membership_intervals(mask1),
        {"S0": _read_ohlcv_labels("S0", "1h", str(root))},
        60 * 60_000_000_000,
        int(1.0 * 3_600_000_000_000),
    )
    assert bounded1.equals(oracle1)
    boundedm, exclm = apply_dynamic_mark_gap_exclusion(mask1, min_gap_hours=1.0)
    oraclem, oracle_exclm = _apply_causal_gap_exclusion(
        mask1,
        roster_membership_intervals(mask1),
        {"S0": _read_mark_labels("S0", "1h")},
        1 * 3_600_000_000_000,
        int(1.0 * 3_600_000_000_000),
    )
    assert boundedm.equals(oraclem)
    assert exclm == oracle_exclm
    assert np is not None


def test_row_group_stat_helpers_handle_untrusted_metadata() -> None:
    """Untrusted or malformed statistics never guess coverage."""
    from types import SimpleNamespace
    from src.market_data.services import mhs_execution as mod

    assert mod._row_group_ms_min_ns(None) is None
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=None, is_stats_set=True)) is None
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=SimpleNamespace(has_min_max=False), is_stats_set=True)) is None
    assert mod._row_group_datetime_min_ns(None) is None
    assert mod._row_group_datetime_min_ns(SimpleNamespace(statistics=None, is_stats_set=True)) is None

    class _Boom:
        is_stats_set = True

        @property
        def statistics(self):
            raise RuntimeError("boom")

    assert mod._row_group_ms_min_ns(_Boom()) is None
    assert mod._row_group_datetime_min_ns(_Boom()) is None
    none_stat = SimpleNamespace(has_min_max=True, min=None, max=None)
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=none_stat, is_stats_set=False)) is None
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=none_stat, is_stats_set=True)) is None
    assert mod._row_group_datetime_min_ns(SimpleNamespace(statistics=none_stat, is_stats_set=True)) is None
    bool_stat = SimpleNamespace(has_min_max=True, min=True, max=True)
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=bool_stat, is_stats_set=True)) is None
    str_stat = SimpleNamespace(has_min_max=True, min="bad", max="bad")
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=str_stat, is_stats_set=True)) is None
    inf_stat = SimpleNamespace(has_min_max=True, min=float("inf"), max=float("inf"))
    assert mod._row_group_ms_min_ns(SimpleNamespace(statistics=inf_stat, is_stats_set=True)) is None
    assert mod._row_group_datetime_min_ns(SimpleNamespace(statistics=str_stat, is_stats_set=True)) is None


def test_bounded_scan_boundary_and_decode_failures(tmp_path, monkeypatch) -> None:
    """Empty, filtered and undecodable groups fail closed with provenance."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import pytest
    from src.common.errors import DataIntegrityError
    from src.market_data.services import mhs_execution as mod
    import src.market_data.services.futures_collection as fc

    grid = pd.date_range("2022-01-01", periods=6, freq="3min", tz="UTC")
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True)
    _write_ohlcv_bars(root / "3m" / "S0.parquet", grid)
    early = int(_ns_of(grid[:1])[-1]) - 1
    monkeypatch.setattr(mod, "_row_group_ms_min_ns", lambda column: None)
    out = mod._read_ohlcv_labels("S0", "3m", str(root), observed_through_ns=early)
    assert out is not None
    assert len(out) == 0
    nan_path = root / "3m" / "NAN.parquet"
    pd.DataFrame({"timestamp": [float("nan")] * 3}).to_parquet(nan_path, row_group_size=2)
    nan_out = mod._read_ohlcv_labels("NAN", "3m", str(root))
    assert nan_out is not None
    assert len(nan_out) == 0
    real_read = pq.ParquetFile.read_row_group

    def _boom(self, *args, **kwargs):
        raise RuntimeError("decode boom")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _boom)
    with pytest.raises(DataIntegrityError, match=r"S0"):
        mod._read_ohlcv_labels("S0", "3m", str(root))
    mark_path = tmp_path / "mark.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: mark_path)
    _write_mark_bars(mark_path, grid)
    with pytest.raises(DataIntegrityError, match=r"S0"):
        mod._read_mark_labels("S0", "1h")
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", real_read)
    orig_to_datetime = pd.to_datetime

    def _boom_dt(*args, **kwargs):
        raise RuntimeError("time boom")

    monkeypatch.setattr(pd, "to_datetime", _boom_dt)
    with pytest.raises(DataIntegrityError, match=r"S0"):
        mod._read_ohlcv_labels("S0", "3m", str(root))
    with pytest.raises(DataIntegrityError, match=r"S0"):
        mod._read_mark_labels("S0", "1h")
    monkeypatch.setattr(pd, "to_datetime", orig_to_datetime)
    monkeypatch.setattr(mod, "_row_group_datetime_min_ns", lambda column: None)
    early_mark = int(_ns_of(grid[:1])[-1]) - 1
    empty_mark = mod._read_mark_labels("S0", "1h", observed_through_ns=early_mark)
    assert empty_mark is not None
    assert len(empty_mark) == 0
    assert np is not None


def test_bounded_mark_scan_skips_future_row_groups(tmp_path, monkeypatch) -> None:
    """Mark groups entirely after the useful bound are never decoded."""
    import pyarrow.parquet as pq
    from src.market_data.services import mhs_execution as mod
    import src.market_data.services.futures_collection as fc
    import pandas as pd

    grid = pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC")
    path = tmp_path / "mark.parquet"
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: path)
    _write_mark_bars(path, grid)
    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=4)
    bound = int(_ns_of(grid[:4])[-1])
    decoded: list[int] = []
    orig_read = pq.ParquetFile.read_row_group

    def spy_read(self, i, columns=None, use_threads=False, **kwargs):
        decoded.append(i)
        return orig_read(self, i, columns=columns, use_threads=use_threads, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy_read)
    out = mod._read_mark_labels("S0", "1h", observed_through_ns=bound)
    assert out is not None
    assert out.tolist() == _ns_of(grid[:4]).tolist()
    assert decoded == [0]
