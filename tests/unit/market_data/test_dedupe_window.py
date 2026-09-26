"""Windowed dedupe state must stay bounded: expired grid outcomes leave the heartbeat window."""

from __future__ import annotations

from src.market_data.streams.dedupe_window import DedupeWindow

_S = 1_000_000_000


def test_dedupe_window_expires_rest_outcomes_outside_window() -> None:
    dedupe = DedupeWindow(rest_window_s=60.0, ws_window_s=60.0)
    dedupe.record_rest_outcome(0, "book_ticker", True, 10, 0, 0.0)
    dedupe.record_rest_outcome(50 * _S, "book_ticker", True, 10, 0, 0.0)
    dedupe.commit()
    assert [o[0] for o in dedupe.rest_outcomes] == [0, 50 * _S]

    dedupe.check_rest("book_ticker", "2026-09-27T00:02:00+00:00", 100 * _S, now_ns=100 * _S)

    assert [o[0] for o in dedupe.rest_outcomes] == [50 * _S]


def test_dedupe_window_discard_leaves_committed_state_untouched() -> None:
    dedupe = DedupeWindow(rest_window_s=60.0, ws_window_s=60.0)
    dedupe.mark_rest_success("book_ticker", "g1", 5 * _S)
    dedupe.record_rest_outcome(5 * _S, "book_ticker", True, 1, 0, 0.0)
    dedupe.discard()

    assert not dedupe.rest_seen("book_ticker", "g1")
    assert list(dedupe.rest_outcomes) == []
