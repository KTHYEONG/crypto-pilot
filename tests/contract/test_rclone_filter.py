"""Rclone filter contract: archive mirror bypasses the data step, temp files never ship."""

from __future__ import annotations

from pathlib import Path

FILTER = Path("deploy/crypto-pilot.rclone-filter")


def _rules() -> list[str]:
    return [
        line.strip()
        for line in FILTER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_public_archive_excluded_from_data_step() -> None:
    rules = _rules()
    excluded = rules.index("- /public_archive/**")
    assert excluded < rules.index("- /futures/**")
    assert excluded < rules.index("- **")
    assert not [r for r in rules if r.startswith("+ /public_archive")]


def test_irreplaceable_streams_still_included() -> None:
    rules = _rules()
    ordered = [
        "+ /state/**",
        "+ /archive/**",
        "+ /futures/liquidations/**",
        "+ /futures/venue_rules/**",
        "+ /live_capture/**",
    ]
    positions = [rules.index(rule) for rule in ordered]
    assert positions == sorted(positions)


def test_transient_exclusions_stay_first() -> None:
    rules = _rules()
    archive = rules.index("- /public_archive/**")
    for pattern in ("- *.tmp", "- *.tmp-*", "- *.partial"):
        assert rules.index(pattern) < archive
