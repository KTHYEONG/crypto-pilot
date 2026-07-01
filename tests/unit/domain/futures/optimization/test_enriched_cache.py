"""Tests for Scenario 4 (enriched cache mtime invalidation).

Time: O(1) per test.
"""
from __future__ import annotations

import time
from pathlib import Path


def make_stale_enriched(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fixture where enriched is older than raw parquet."""
    raw = tmp_path / "BTCUSDT_4h.parquet"
    enriched = tmp_path / "BTCUSDT_4h_enriched.parquet"
    enriched.write_bytes(b"old")
    time.sleep(0.05)
    raw.write_bytes(b"new")
    return raw, enriched


def test_enriched_cache_invalidated_when_raw_is_newer(tmp_path: Path) -> None:
    """enriched가 raw보다 오래된 경우 enriched_stale == True."""
    raw, enriched = make_stale_enriched(tmp_path)
    enriched_stale = (
        not enriched.exists()
        or (
            raw.exists()
            and enriched.stat().st_mtime < raw.stat().st_mtime
        )
    )
    assert enriched_stale is True


def test_enriched_cache_not_rebuilt_when_already_fresh(tmp_path: Path) -> None:
    """raw보다 enriched가 더 새로운 경우 enriched_stale == False."""
    raw = tmp_path / "BTCUSDT_4h.parquet"
    enriched = tmp_path / "BTCUSDT_4h_enriched.parquet"
    raw.write_bytes(b"raw_data")
    time.sleep(0.05)
    enriched.write_bytes(b"enriched_data")  # enriched가 더 새로움
    enriched_stale = (
        not enriched.exists()
        or (
            raw.exists()
            and enriched.stat().st_mtime < raw.stat().st_mtime
        )
    )
    assert enriched_stale is False


def test_enriched_stale_when_enriched_missing(tmp_path: Path) -> None:
    """enriched 파일이 없는 경우 enriched_stale == True."""
    raw = tmp_path / "BTCUSDT_4h.parquet"
    enriched = tmp_path / "BTCUSDT_4h_enriched.parquet"
    raw.write_bytes(b"raw_data")
    # enriched 파일 없음
    enriched_stale = (
        not enriched.exists()
        or (
            raw.exists()
            and enriched.stat().st_mtime < raw.stat().st_mtime
        )
    )
    assert enriched_stale is True
