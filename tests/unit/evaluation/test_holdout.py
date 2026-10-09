"""Spec 38 part 6: a holdout is evidence only the first time it is looked at."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.evaluation.holdout import consume_holdout_look


def _window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def test_second_look_at_same_window_refused(tmp_path) -> None:
    """A second evaluation of the same window raises; a later window is allowed."""
    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", _window("2026-07-01", "2026-10-01"), path=path)
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "digest-b", _window("2026-08-01", "2026-11-01"), path=path)
    consume_holdout_look("flow_mom_top20", "digest-c", _window("2026-10-01", "2027-01-01"), path=path)


def test_holdout_journal_guards(tmp_path) -> None:
    """Invalid windows and corrupt journals fail closed."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.evaluation.holdout import holdout_overlaps

    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("", "d", _window("2026-07-01", "2026-10-01"), path=path)
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "", _window("2026-07-01", "2026-10-01"), path=path)
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", ("2026-07-01", "2026-10-01"), path=path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", "2026-07-01", path=path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-10-01", "2026-07-01"), path=path)
    with pytest.raises(DataIntegrityError):
        consume_holdout_look(
            "s", "d",
            (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01", tz="UTC")),
            path=path,
        )
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", ("x", "y"), path=path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", "x", path=path)  # type: ignore[arg-type]

    path.write_text("broken\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-07-01", "2026-10-01"), path=path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-07-01", "2026-10-01"), path=path)
    path.write_text('{"strategy_id": "s"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", _window("2026-07-01", "2026-10-01"), path=path)


def test_holdout_overlaps_only_same_strategy(tmp_path) -> None:
    """Other strategies' looks and blank lines never block a fresh window."""
    from src.evaluation.holdout import holdout_overlaps

    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    consume_holdout_look("other", "d", _window("2026-07-01", "2026-10-01"), path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert holdout_overlaps("flow_mom_top20", _window("2026-07-01", "2026-10-01"), path=path) is False
    consume_holdout_look("flow_mom_top20", "d", _window("2026-07-01", "2026-10-01"), path=path)
    assert holdout_overlaps("flow_mom_top20", _window("2026-08-01", "2026-09-01"), path=path) is True
