"""Invariant guards for atomic parquet read-modify-write and quarantine."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pandas as pd
import pytest
from pyarrow.lib import ArrowInvalid

from src.common import parquet_io as pio


def _frame(rows: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"a": rows})


def test_write_failure_leaves_old_file_intact(tmp_path: Path) -> None:
    """A failed serialization keeps the previous partition bytes and no temp residue."""
    path = tmp_path / "part.parquet"
    _frame([1, 2, 3]).to_parquet(path, index=False, compression="zstd")
    before = path.read_bytes()
    original = pd.DataFrame.to_parquet

    def _bad(self: pd.DataFrame, target: object, *args: object, **kwargs: object) -> None:
        Path(str(target)).write_bytes(b"12345")
        raise RuntimeError("boom")

    pd.DataFrame.to_parquet = _bad  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="boom"):
            pio.write_parquet_atomic(_frame([9]), path, compression="zstd")
    finally:
        pd.DataFrame.to_parquet = original
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
    assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_successful_write_round_trips_with_no_residue(tmp_path: Path) -> None:
    """Two atomic writes leave exactly the second frame and only the target file."""
    path = tmp_path / "part.parquet"
    pio.write_parquet_atomic(_frame([1, 2]), path, compression="zstd")
    out = pio.write_parquet_atomic(_frame([3, 4]), path, compression="zstd")
    assert out == path
    assert pd.read_parquet(path)["a"].tolist() == [3, 4]
    assert [p.name for p in tmp_path.iterdir()] == ["part.parquet"]


def test_temp_file_is_hidden_same_directory_sibling(tmp_path: Path) -> None:
    """The atomic-replace source lives in the target directory as a hidden .tmp file."""
    path = tmp_path / "part.parquet"
    seen: dict[str, str] = {}
    original = os.replace

    def _recording(src: object, dst: object) -> None:
        seen["src"] = str(src)
        seen["dst"] = str(dst)
        original(src, dst)

    os.replace = _recording  # type: ignore[assignment]
    try:
        pio.write_parquet_atomic(_frame([1]), path, compression="zstd")
    finally:
        os.replace = original
    src = Path(seen["src"])
    assert src.parent == path.parent
    assert src.name.startswith(".")
    assert src.name.endswith(".tmp")


def test_truncated_file_is_quarantined_not_overwritten(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A truncated partition is moved aside with identical bytes and an ERROR record."""
    path = tmp_path / "part.parquet"
    _frame(list(range(10))).to_parquet(path, index=False, compression="zstd")
    raw = path.read_bytes()
    truncated = raw[: len(raw) // 2]
    path.write_bytes(truncated)
    with caplog.at_level(logging.ERROR, logger="ParquetIO"):
        assert pio.read_parquet_or_quarantine(path, stage="live_fills") is None
    assert not path.exists()
    quarantined = list((tmp_path / "_quarantine").glob("part.parquet.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == truncated
    assert any("status=QUARANTINED" in r.getMessage() for r in caplog.records)


def test_zero_byte_file_is_quarantined(tmp_path: Path) -> None:
    """An empty file reads as quarantined, not as an empty frame."""
    path = tmp_path / "part.parquet"
    path.write_bytes(b"")
    assert pio.read_parquet_or_quarantine(path, stage="s") is None
    assert not path.exists()
    assert len(list((tmp_path / "_quarantine").glob("part.parquet.*.corrupt"))) == 1


def test_page_level_decode_error_is_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page-level decompression damage (OSError without errno) is moved aside."""

    def _raise(*args: object, **kwargs: object) -> pd.DataFrame:
        raise OSError("ZSTD decompression failed")

    path = tmp_path / "p2.parquet"
    _frame([1]).to_parquet(path, index=False)
    monkeypatch.setattr(pd, "read_parquet", _raise)
    assert pio.read_parquet_or_quarantine(path, stage="s") is None
    assert not path.exists()
    assert len(list((tmp_path / "_quarantine").glob("p2.parquet.*.corrupt"))) == 1


def test_os_error_propagates_without_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine OS failures keep the file in place and create no quarantine directory."""

    def _raise(*args: object, **kwargs: object) -> pd.DataFrame:
        raise PermissionError(errno.EACCES, "denied")

    path = tmp_path / "p3.parquet"
    _frame([1]).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(pd, "read_parquet", _raise)
    with pytest.raises(PermissionError):
        pio.read_parquet_or_quarantine(path, stage="s")
    assert path.read_bytes() == before
    assert not (tmp_path / "_quarantine").exists()


def test_non_decode_error_propagates_with_path_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-decode failures (e.g. ValueError) propagate without moving the file."""

    def _raise(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ValueError("bad schema")

    path = tmp_path / "p4.parquet"
    _frame([1]).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(pd, "read_parquet", _raise)
    with pytest.raises(ValueError, match="bad schema"):
        pio.read_parquet_or_quarantine(path, stage="s")
    assert path.read_bytes() == before
    assert not (tmp_path / "_quarantine").exists()


def test_missing_partition_reads_as_absent(tmp_path: Path) -> None:
    """A non-existent path returns None without creating a quarantine directory."""
    assert pio.read_parquet_or_quarantine(tmp_path / "nope.parquet", stage="s") is None
    assert not (tmp_path / "_quarantine").exists()


def test_quarantine_target_collision_gets_unique_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing quarantine target for the same stamp gains a counter suffix."""
    from datetime import datetime, timezone

    fixed = datetime(2026, 9, 24, 5, 0, 0, 123456, tzinfo=timezone.utc)  # noqa: UP017

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return fixed

    monkeypatch.setattr(pio, "datetime", _FrozenDatetime)
    path = tmp_path / "part.parquet"
    _frame([1]).to_parquet(path, index=False)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    stamp = "20260924T050000123456Z"
    first = tmp_path / "_quarantine" / f"part.parquet.{stamp}.{os.getpid()}.corrupt"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"older")
    assert pio.read_parquet_or_quarantine(path, stage="s") is None
    second = tmp_path / "_quarantine" / f"part.parquet.{stamp}.{os.getpid()}.1.corrupt"
    assert second.read_bytes() == raw[: len(raw) // 2]
    assert first.read_bytes() == b"older"


def test_decode_classifier_boundaries() -> None:
    """ArrowInvalid and errnoless OSError decode-fail; errno OSError and ValueError do not."""
    assert pio.is_undecodable_parquet_error(ArrowInvalid("bad")) is True
    assert pio.is_undecodable_parquet_error(OSError("x")) is True
    assert pio.is_undecodable_parquet_error(OSError(errno.EIO, "x")) is False
    assert pio.is_undecodable_parquet_error(ValueError("x")) is False
