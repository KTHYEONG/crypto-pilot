"""Rclone filter contract: public-archive mirror is backed up, temp files never are."""

from __future__ import annotations

from pathlib import Path

FILTER = Path("deploy/crypto-pilot.rclone-filter")


def _rules() -> list[str]:
    return [
        line.strip()
        for line in FILTER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_public_archive_included_before_futures_exclusion() -> None:
    rules = _rules()
    include = rules.index("+ /public_archive/**")
    assert include < rules.index("- /futures/**")
    assert include < rules.index("- **")


def test_transient_exclusions_stay_first() -> None:
    rules = _rules()
    include = rules.index("+ /public_archive/**")
    for pattern in ("- *.tmp", "- *.tmp-*", "- *.partial"):
        assert rules.index(pattern) < include
