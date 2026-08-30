"""Record storage scenarios."""

import time
from pathlib import Path

import pandas as pd

from src.live.records import append_typed_frame


def test_SCENARIO_REC_02_no_deletion_ever(tmp_path: Path, monkeypatch) -> None:
    dtypes = {"decision_time": "datetime64[ns, UTC]", "value": "float64"}
    directory = tmp_path / "records"
    # 24 months
    for i in range(24):
        year = 2026 + (i // 12)
        month = 1 + (i % 12)
        ts = pd.Timestamp(f"{year}-{month:02d}-15 00:00:00", tz="UTC")
        df = pd.DataFrame({"decision_time": [ts], "value": [float(i)]})
        append_typed_frame(df, directory, "x", time_column="decision_time", dtypes=dtypes)
    files = list(directory.glob("x_*.parquet"))
    assert len(files) == 24
    # monkeypatch Path.unlink to raise
    orig_unlink = Path.unlink

    def failing_unlink(self, *a, **kw):
        raise OSError("unlink should not be called")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    # append should still succeed
    ts = pd.Timestamp("2028-01-15 00:00:00", tz="UTC")
    df = pd.DataFrame({"decision_time": [ts], "value": [99.0]})
    # need month 202801 already exists, so it will read+concat
    written = append_typed_frame(df, directory, "x", time_column="decision_time", dtypes=dtypes)
    assert written
    # check source files have no unlink
    import pathlib

    for fp in ["src/live/records.py", "src/live/fills.py", "src/live/microstructure.py", "src/live/tax_ledger.py"]:
        content = pathlib.Path(fp).read_text(encoding="utf-8")
        assert "unlink" not in content
    # restore not needed


def test_SCENARIO_REC_03_monthly_partition_isolation(tmp_path: Path) -> None:
    dtypes = {"decision_time": "datetime64[ns, UTC]", "value": "float64"}
    directory = tmp_path / "rec"
    ts1 = pd.Timestamp("2026-01-31 00:00:00", tz="UTC")
    ts2 = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")
    df = pd.DataFrame({"decision_time": [ts1, ts2], "value": [1.0, 2.0]})
    written = append_typed_frame(df, directory, "x", time_column="decision_time", dtypes=dtypes)
    assert len(written) == 2
    p1 = directory / "x_202601.parquet"
    p2 = directory / "x_202602.parquet"
    assert p1.exists()
    assert p2.exists()
    assert len(pd.read_parquet(p1)) == 1
    assert len(pd.read_parquet(p2)) == 1
    mtime_before = p1.stat().st_mtime
    time.sleep(0.02)
    ts3 = pd.Timestamp("2026-02-15 00:00:00", tz="UTC")
    df2 = pd.DataFrame({"decision_time": [ts3], "value": [3.0]})
    append_typed_frame(df2, directory, "x", time_column="decision_time", dtypes=dtypes)
    mtime_after = p1.stat().st_mtime
    assert mtime_after == mtime_before
    assert len(pd.read_parquet(p1)) == 1
    assert len(pd.read_parquet(p2)) == 2
# SCENARIO_REC_02-no-deletion-ever
# SCENARIO_REC_03-monthly-partition-isolation
