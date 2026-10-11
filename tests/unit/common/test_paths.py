from __future__ import annotations

from src.common import paths


def test_public_archive_dir_anchors_under_data_dir() -> None:
    assert paths.PUBLIC_ARCHIVE_DIR == paths.DATA_DIR / "public_archive"
