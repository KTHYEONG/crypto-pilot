"""Spec 38 part 6: a holdout is evidence only the first time it is looked at."""

from __future__ import annotations

import json

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


def test_public_reader_returns_journal_rows_in_order(tmp_path) -> None:
    from src.evaluation.holdout import read_holdout_looks
    from src.strategy.release import releases_dir

    first = _window("2026-07-01", "2026-10-01")
    second = _window("2026-10-01", "2027-01-01")
    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", first, path=journal)
    consume_holdout_look("flow_mom_top20", "digest-b", second, path=journal)
    original = journal.read_bytes()
    rows = read_holdout_looks("flow_mom_top40_control", root=tmp_path)
    assert [row["spec_digest"] for row in rows] == ["digest-a", "digest-b"]
    assert rows[0]["window_start"] == first[0].isoformat()
    assert journal.read_bytes() == original


def test_look_recorded_matches_exact_window_and_digest(tmp_path) -> None:
    from src.evaluation.holdout import holdout_look_recorded
    from src.strategy.release import releases_dir

    window = _window("2026-07-01", "2026-10-01")
    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", window, path=journal)
    assert holdout_look_recorded("flow_mom_top20", "digest-a", window, root=tmp_path) is True
    assert holdout_look_recorded("flow_mom_top20", "other", window, root=tmp_path) is False
    shifted = _window("2026-07-02", "2026-10-01")
    assert holdout_look_recorded("flow_mom_top20", "digest-a", shifted, root=tmp_path) is False
    shifted_end = _window("2026-07-01", "2026-10-02")
    assert holdout_look_recorded("flow_mom_top20", "digest-a", shifted_end, root=tmp_path) is False


@pytest.mark.parametrize("malformed", [
    "broken", "[]", "{}", '{"strategy_id": "flow_mom_top20"}',
    json.dumps({"strategy_id": "s", "spec_digest": "", "window_start": "2026-07-01T00:00:00Z", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "invalid", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "NaT", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "2026-07-01", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "2026-10-01T00:00:00Z", "window_end": "2026-07-01T00:00:00Z"}),
])
def test_malformed_journal_row_fails_closed(tmp_path, malformed) -> None:
    from src.evaluation.holdout import holdout_look_recorded, read_holdout_looks
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", _window("2026-07-01", "2026-10-01"), path=journal)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(malformed + "\n")
    with pytest.raises(DataIntegrityError):
        read_holdout_looks("flow_mom_top20", root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded(
            "flow_mom_top20", "digest-a", _window("2026-07-01", "2026-10-01"), root=tmp_path
        )


def test_public_holdout_guards_fail_closed(tmp_path) -> None:
    from src.evaluation.holdout import holdout_look_recorded, read_holdout_looks

    window = _window("2026-07-01", "2026-10-01")
    with pytest.raises(DataIntegrityError):
        read_holdout_looks("", root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("", "digest-a", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "digest-a", "2026-07-01", root=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "digest-a", ("2026-07-01", "2026-10-01"), root=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [
    (pd.NaT, pd.Timestamp("2026-10-01", tz="UTC")),
    (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01", tz="UTC")),
    _window("2026-10-01", "2026-07-01"),
    _window("2026-07-01", "2026-07-01"),
])
def test_public_holdout_rejects_invalid_window(tmp_path, window) -> None:
    from src.evaluation.holdout import holdout_look_recorded

    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "digest-a", window, root=tmp_path)


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
