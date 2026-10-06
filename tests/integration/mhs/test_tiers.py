"""Heavy-tier consistency and fork-worker cap contracts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.integration.mhs._tiers import (
    HEAVY_DIAGNOSTIC_FIXTURES,
    fork_worker_cap,
    heavy_tier_violations,
)


@dataclass
class _FakeItem:
    nodeid: str
    path: Path
    fixturenames: list[str] = field(default_factory=list)
    marks: list[pytest.Mark] = field(default_factory=list)

    def iter_markers(self, name: str | None = None) -> Iterator[pytest.Mark]:
        if name is None:
            yield from self.marks
        else:
            for mark in self.marks:
                if mark.name == name:
                    yield mark


def _item(
    tmp_path: Path,
    name: str,
    fixtures: list[str],
    mark_names: list[str],
    *,
    outside: bool = False,
) -> _FakeItem:
    root = tmp_path / "suite"
    root.mkdir(exist_ok=True)
    if outside:
        other = tmp_path / "elsewhere" / f"{name}.py"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.touch()
        path = other
        nodeid = f"{other}::{name}"
    else:
        target = root / f"{name}.py"
        target.touch()
        path = target
        nodeid = f"{target}::{name}"
    marks = [getattr(pytest.mark, m).mark for m in mark_names]
    return _FakeItem(nodeid=nodeid, path=path, fixturenames=list(fixtures), marks=marks)


def test_cap_applies_only_inside_xdist_workers() -> None:
    assert fork_worker_cap({"PYTEST_XDIST_WORKER": "gw0"}) == 1
    assert fork_worker_cap({}) is None


def test_fork_plan_follows_execution_mode() -> None:
    from src.mhs.parallel import plan_worker_count

    expected = 1 if "PYTEST_XDIST_WORKER" in os.environ else min(3, os.cpu_count() or 1)
    assert plan_worker_count(3, 1, ram_guard=False) == expected


@pytest.mark.mhs_parallel_parity
def test_parity_marker_lifts_the_cap() -> None:
    from src.mhs.parallel import plan_worker_count

    assert plan_worker_count(3, 1, ram_guard=False) == min(3, os.cpu_count() or 1)


def test_heavy_fixture_without_tier_mark_is_flagged(tmp_path: Path) -> None:
    item = _item(tmp_path, "t1", ["report"], [])
    violations = heavy_tier_violations([item], tmp_path / "suite")  # type: ignore[list-item]
    assert len(violations) == 1
    assert item.nodeid in violations[0]


def test_slow_heavy_consumers_stay_in_slow_tier(tmp_path: Path) -> None:
    item = _item(tmp_path, "t2", ["synthetic_market", "report"], ["slow"])
    assert heavy_tier_violations([item], tmp_path / "suite") == []  # type: ignore[list-item]


def test_tiers_are_disjoint(tmp_path: Path) -> None:
    item = _item(tmp_path, "t3", ["report"], ["slow", "e2e_heavy"])
    violations = heavy_tier_violations([item], tmp_path / "suite")  # type: ignore[list-item]
    assert len(violations) == 1
    assert item.nodeid in violations[0]


def test_stale_tier_mark_is_flagged(tmp_path: Path) -> None:
    item = _item(tmp_path, "t4", ["tmp_path"], ["e2e_heavy"])
    violations = heavy_tier_violations([item], tmp_path / "suite")  # type: ignore[list-item]
    assert len(violations) == 1
    assert item.nodeid in violations[0]


def test_all_independent_tier_violations_are_reported(tmp_path: Path) -> None:
    item = _item(tmp_path, "stale_slow", ["tmp_path"], ["slow", "e2e_heavy"])
    assert heavy_tier_violations([item], tmp_path / "suite") == [
        f"{item.nodeid}: e2e_heavy and slow marks must stay disjoint",
        f"{item.nodeid}: e2e_heavy mark without heavy diagnostic fixture",
    ]


def test_class_level_marks_count(tmp_path: Path) -> None:
    item = _item(tmp_path, "t5", ["late_market_report"], ["e2e_heavy"])
    assert heavy_tier_violations([item], tmp_path / "suite") == []  # type: ignore[list-item]


def test_items_outside_suite_are_ignored(tmp_path: Path) -> None:
    item = _item(tmp_path, "t6", ["report"], [], outside=True)
    assert heavy_tier_violations([item], tmp_path / "suite") == []  # type: ignore[list-item]


def test_heavy_fixture_set_excludes_light_oracles() -> None:
    assert "synthetic_market" not in HEAVY_DIAGNOSTIC_FIXTURES
    assert "fold_market" not in HEAVY_DIAGNOSTIC_FIXTURES
    assert "ohlcv_market" not in HEAVY_DIAGNOSTIC_FIXTURES
