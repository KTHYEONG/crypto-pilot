from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from src.domain.futures.strategy.tiered_workflow.lifecycle import (
    TfResourceReleaseReport,
    release_completed_tf_resources,
)


def test_release_report_dataclass() -> None:
    report = TfResourceReleaseReport(
        tf="4h", feature_cache_bytes=1024, removed_from_per_tf_map=True, primary_retained=False
    )
    assert report.tf == "4h"
    assert report.feature_cache_bytes == 1024


def test_release_non_primary_removes_from_map() -> None:
    primary = MagicMock()
    aligned_tf = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": aligned_tf}

    result = release_completed_tf_resources(
        tf="4h",
        aligned_tf=aligned_tf,
        primary_aligned=primary,
        per_tf_aligned=per_tf_map,
    )

    assert result.feature_cache_bytes >= 0
    assert result.removed_from_per_tf_map is True
    assert result.primary_retained is False
    assert "4h" not in per_tf_map


def test_release_primary_retained_in_map() -> None:
    primary = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": primary}

    result = release_completed_tf_resources(
        tf="4h",
        aligned_tf=primary,
        primary_aligned=primary,
        per_tf_aligned=per_tf_map,
    )

    assert result.primary_retained is True
    assert result.removed_from_per_tf_map is False
    assert "4h" in per_tf_map


def test_release_no_per_tf_map_does_not_crash() -> None:
    primary = MagicMock()
    aligned_tf = MagicMock()

    result = release_completed_tf_resources(
        tf="4h",
        aligned_tf=aligned_tf,
        primary_aligned=primary,
        per_tf_aligned=None,
    )

    assert result.feature_cache_bytes >= 0
    assert result.removed_from_per_tf_map is False


def test_release_idempotent_second_call() -> None:
    primary = MagicMock()
    aligned_tf = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": aligned_tf}

    r1 = release_completed_tf_resources(
        tf="4h", aligned_tf=aligned_tf, primary_aligned=primary, per_tf_aligned=per_tf_map,
    )
    r2 = release_completed_tf_resources(
        tf="4h", aligned_tf=aligned_tf, primary_aligned=primary, per_tf_aligned=per_tf_map,
    )

    assert r1.removed_from_per_tf_map is True
    assert r2.removed_from_per_tf_map is False
    assert "4h" not in per_tf_map


def test_release_feature_cache_error_suppressed() -> None:
    primary = MagicMock()
    aligned_tf = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": aligned_tf}

    with patch(
        "src.domain.futures.strategy.tiered_workflow.lifecycle.release_aligned_feature_cache",
        side_effect=RuntimeError("test error"),
    ):
        result = release_completed_tf_resources(
            tf="4h",
            aligned_tf=aligned_tf,
            primary_aligned=primary,
            per_tf_aligned=per_tf_map,
        )

    assert result.feature_cache_bytes == 0
    assert result.removed_from_per_tf_map is True


def test_release_primary_debug_log(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.lifecycle as lc

    monkeypatch.setattr(lc.logger, "level", logging.DEBUG)

    primary = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": primary}

    result = release_completed_tf_resources(
        tf="4h",
        aligned_tf=primary,
        primary_aligned=primary,
        per_tf_aligned=per_tf_map,
    )

    assert result.primary_retained is True
    assert result.removed_from_per_tf_map is False


def test_release_cache_clear_raises_suppressed(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.lifecycle as lc

    def raiser():
        raise RuntimeError("cache clear failed")

    monkeypatch.setattr(lc, "clear_aligned_data_maps_cache", raiser)

    primary = MagicMock()
    aligned_tf = MagicMock()
    per_tf_map: dict[str, MagicMock] = {"4h": aligned_tf}

    result = release_completed_tf_resources(
        tf="4h",
        aligned_tf=aligned_tf,
        primary_aligned=primary,
        per_tf_aligned=per_tf_map,
    )

    assert result.removed_from_per_tf_map is True
