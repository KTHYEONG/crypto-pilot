"""Spec 49 part 3: the one-look journal is bound to the signal digest."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.evaluation.holdout import (
    consume_holdout_look,
    holdout_covered,
    holdout_look_recorded,
)


def _window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def test_second_look_at_same_window_refused(tmp_path) -> None:
    """A second evaluation of the same window raises; a later window is allowed."""
    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="sig-a")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "digest-b", _window("2026-08-01", "2026-11-01"), path=path, signal_digest="sig-a")
    consume_holdout_look("flow_mom_top20", "digest-c", _window("2026-10-01", "2027-01-01"), path=path, signal_digest="sig-a")


def test_identical_rerun_is_not_a_second_look(tmp_path) -> None:
    """Same window and signal digest re-materializes: False, journal unchanged byte-for-byte."""
    path = tmp_path / "flow_mom.holdout.jsonl"
    window = _window("2026-07-01", "2026-10-01")
    assert consume_holdout_look("flow_mom_top20", "digest-a", window, path=path, signal_digest="sig-a") is True
    before = path.read_bytes()
    assert consume_holdout_look("flow_mom_top20", "changed-sizing", window, path=path, signal_digest="sig-a") is False
    assert path.read_bytes() == before


def test_same_utc_instants_match_after_operator_backfill(tmp_path) -> None:
    from src.strategy.release import releases_dir

    path = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "strategy_id": "flow_mom_top20", "spec_digest": "old", "signal_digest": "sig",
        "window_start": "2026-07-01T00:00:00Z", "window_end": "2026-10-01T00:00:00Z",
    }) + "\n")
    before = path.read_bytes()
    window = _window("2026-07-01", "2026-10-01")
    assert holdout_look_recorded("flow_mom_top20", "sig", window, root=tmp_path)
    assert consume_holdout_look("flow_mom_top20", "new", window, path=path, signal_digest="sig") is False
    assert path.read_bytes() == before


def test_changed_signal_cannot_reuse_window(tmp_path) -> None:
    """Same window with another signal digest raises."""
    path = tmp_path / "flow_mom.holdout.jsonl"
    window = _window("2026-07-01", "2026-10-01")
    consume_holdout_look("flow_mom_top20", "digest-a", window, path=path, signal_digest="sig-a")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "digest-a", window, path=path, signal_digest="sig-b")


@pytest.mark.parametrize("trailing", ["{}", json.dumps({
    "strategy_id": "flow_mom_top20", "spec_digest": "old", "signal_digest": "other",
    "window_start": "2026-08-01T00:00:00+00:00", "window_end": "2026-11-01T00:00:00+00:00",
})])
def test_rematerialization_checks_entire_journal(tmp_path, trailing) -> None:
    path = tmp_path / "flow_mom.holdout.jsonl"
    window = _window("2026-07-01", "2026-10-01")
    consume_holdout_look("flow_mom_top20", "d", window, path=path, signal_digest="sig")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(trailing + "\n")
    original = path.read_bytes()
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "d", window, path=path, signal_digest="sig")
    assert path.read_bytes() == original


def test_widened_window_refused(tmp_path) -> None:
    """July-September look blocks a July-October consume with the same digest."""
    path = tmp_path / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "d", _window("2026-07-01", "2026-09-01"), path=path, signal_digest="sig")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="sig")


def test_legacy_row_blocks_but_does_not_match(tmp_path) -> None:
    """A row without signal_digest never matches but still blocks overlapping windows."""
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps({"strategy_id": "flow_mom_top20", "spec_digest": "d",
                    "window_start": "2026-07-01T00:00:00+00:00",
                    "window_end": "2026-10-01T00:00:00+00:00"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    window = _window("2026-07-01", "2026-10-01")
    assert holdout_look_recorded("flow_mom_top20", "sig", window, root=tmp_path) is False
    assert holdout_covered("flow_mom_top20", "sig", window, root=tmp_path) is False
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("flow_mom_top20", "d", window, path=journal, signal_digest="sig")


def test_holdout_covered_requires_containment_and_signal(tmp_path) -> None:
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "d", _window("2026-07-01", "2026-10-01"), path=journal, signal_digest="sig")
    assert holdout_covered("flow_mom_top20", "sig", _window("2026-08-01", "2026-09-01"), root=tmp_path) is True
    assert holdout_covered("flow_mom_top20", "other", _window("2026-08-01", "2026-09-01"), root=tmp_path) is False
    assert holdout_covered("flow_mom_top20", "sig", _window("2026-09-01", "2026-11-01"), root=tmp_path) is False


def test_foreign_family_row_cannot_authorize_holdout(tmp_path) -> None:
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    window = _window("2026-07-01", "2026-10-01")
    consume_holdout_look("other", "d", window, path=journal, signal_digest="sig")
    assert not holdout_look_recorded("flow_mom_top20", "sig", window, root=tmp_path)
    assert not holdout_covered("flow_mom_top20", "sig", window, root=tmp_path)


def test_holdout_journal_guards(tmp_path) -> None:
    """Invalid windows and corrupt journals fail closed."""
    from src.evaluation.holdout import holdout_overlaps

    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="s")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="s")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", ("2026-07-01", "2026-10-01"), path=path, signal_digest="s")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", "2026-07-01", path=path, signal_digest="s")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-10-01", "2026-07-01"), path=path, signal_digest="s")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look(
            "s", "d",
            (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01", tz="UTC")),
            path=path, signal_digest="s",
        )
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", ("x", "y"), path=path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", "x", path=path)  # type: ignore[arg-type]

    path.write_text("broken\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="s")
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        consume_holdout_look("s", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="s")
    path.write_text('{"strategy_id": "s"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        holdout_overlaps("s", _window("2026-07-01", "2026-10-01"), path=path)


def test_public_reader_returns_journal_rows_in_order(tmp_path) -> None:
    from src.evaluation.holdout import read_holdout_looks
    from src.strategy.release import releases_dir

    first = _window("2026-07-01", "2026-10-01")
    second = _window("2026-10-01", "2027-01-01")
    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", first, path=journal, signal_digest="sig")
    consume_holdout_look("flow_mom_top20", "digest-b", second, path=journal, signal_digest="sig")
    original = journal.read_bytes()
    rows = read_holdout_looks("flow_mom_top40_control", root=tmp_path)
    assert [row["spec_digest"] for row in rows] == ["digest-a", "digest-b"]
    assert all(row["signal_digest"] == "sig" for row in rows)
    assert rows[0]["window_start"] == first[0].isoformat()
    assert journal.read_bytes() == original


def test_look_recorded_matches_exact_window_and_signal(tmp_path) -> None:
    from src.strategy.release import releases_dir

    window = _window("2026-07-01", "2026-10-01")
    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", window, path=journal, signal_digest="sig-a")
    assert holdout_look_recorded("flow_mom_top20", "sig-a", window, root=tmp_path) is True
    assert holdout_look_recorded("flow_mom_top20", "other", window, root=tmp_path) is False
    assert holdout_look_recorded("flow_mom_top40_control", "sig-a", window, root=tmp_path) is True
    shifted = _window("2026-07-02", "2026-10-01")
    assert holdout_look_recorded("flow_mom_top20", "sig-a", shifted, root=tmp_path) is False
    shifted_end = _window("2026-07-01", "2026-10-02")
    assert holdout_look_recorded("flow_mom_top20", "sig-a", shifted_end, root=tmp_path) is False


@pytest.mark.parametrize("malformed", [
    "broken", "[]", "{}",
    '{"strategy_id": "flow_mom_top20"}',
    json.dumps({"strategy_id": "s", "spec_digest": "", "window_start": "2026-07-01T00:00:00Z", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "invalid", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "NaT", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "2026-07-01", "window_end": "2026-10-01T00:00:00Z"}),
    json.dumps({"strategy_id": "s", "spec_digest": "d", "window_start": "2026-10-01T00:00:00Z", "window_end": "2026-07-01T00:00:00Z"}),
])
def test_malformed_journal_row_fails_closed(tmp_path, malformed) -> None:
    from src.evaluation.holdout import read_holdout_looks
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    consume_holdout_look("flow_mom_top20", "digest-a", _window("2026-07-01", "2026-10-01"), path=journal, signal_digest="sig")
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(malformed + "\n")
    with pytest.raises(DataIntegrityError):
        read_holdout_looks("flow_mom_top20", root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded(
            "flow_mom_top20", "sig", _window("2026-07-01", "2026-10-01"), root=tmp_path
        )


def test_public_holdout_guards_fail_closed(tmp_path) -> None:
    from src.evaluation.holdout import read_holdout_looks

    window = _window("2026-07-01", "2026-10-01")
    with pytest.raises(DataIntegrityError):
        read_holdout_looks("", root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("", "sig", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "sig", "2026-07-01", root=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "sig", ("2026-07-01", "2026-10-01"), root=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_covered("flow_mom_top20", "", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_covered("flow_mom_top20", "sig", "2026-07-01", root=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        holdout_covered("", "sig", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_covered("flow_mom_top20", "sig", ("2026-07-01", "2026-10-01"), root=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [
    (pd.NaT, pd.Timestamp("2026-10-01", tz="UTC")),
    (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01", tz="UTC")),
    _window("2026-10-01", "2026-07-01"),
    _window("2026-07-01", "2026-07-01"),
])
def test_public_holdout_rejects_invalid_window(tmp_path, window) -> None:
    with pytest.raises(DataIntegrityError):
        holdout_look_recorded("flow_mom_top20", "sig", window, root=tmp_path)
    with pytest.raises(DataIntegrityError):
        holdout_covered("flow_mom_top20", "sig", window, root=tmp_path)


def test_holdout_covered_corrupt_matching_row_fails_closed(tmp_path) -> None:
    from src.strategy.release import releases_dir

    journal = releases_dir(tmp_path) / "flow_mom.holdout.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps({"strategy_id": "flow_mom_top20", "spec_digest": "d", "signal_digest": "sig",
                    "window_start": "not-a-time", "window_end": "2026-10-01T00:00:00+00:00"},
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match="corrupt"):
        holdout_covered("flow_mom_top20", "sig", _window("2026-07-01", "2026-10-01"), root=tmp_path)


def test_holdout_overlaps_only_same_strategy(tmp_path) -> None:
    """Other strategies' looks and blank lines never block a fresh window."""
    from src.evaluation.holdout import holdout_overlaps

    path = tmp_path / "flow_mom_top20.holdout.jsonl"
    consume_holdout_look("other", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="sig")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert holdout_overlaps("flow_mom_top20", _window("2026-07-01", "2026-10-01"), path=path) is False
    consume_holdout_look("flow_mom_top20", "d", _window("2026-07-01", "2026-10-01"), path=path, signal_digest="sig")
    assert holdout_overlaps("flow_mom_top20", _window("2026-08-01", "2026-09-01"), path=path) is True
