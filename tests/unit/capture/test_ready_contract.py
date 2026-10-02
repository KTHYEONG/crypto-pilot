"""The capture writer, watchdog and deploy host agree on the READY contract."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.application.ops.capture_handover import evaluate_ready
from src.capture.heartbeat import build_capture_heartbeat
from src.capture.rest import RestStatus
from src.capture.ws import WsStatus
from src.market_data.streams.recorder_health import _slot_ready

STARTED_NS = 1_759_000_000_000_000_000


def _document(
    *, book_first_ns: int | None, frame_ns: int | None, flush_failures: int = 0, stopped_ns: int | None = None
) -> dict:
    return build_capture_heartbeat(
        slot="blue",
        pid=1,
        fingerprint="sha256:test",
        started_at_ns=STARTED_NS,
        stopped_at_ns=stopped_ns,
        rest={"book_ticker": RestStatus(first_ok_at_ns=book_first_ns)},
        ws=WsStatus(first_frame_at_ns=frame_ns),
        last_flush_at_ns=STARTED_NS + 2_000_000_000,
        flush_failures=flush_failures,
        now_ns=STARTED_NS + 3_000_000_000,
    )


@pytest.mark.parametrize(
    ("document", "ready"),
    [
        (_document(book_first_ns=STARTED_NS + 1_000_000_000, frame_ns=STARTED_NS + 1_000_000_000), True),
        (_document(book_first_ns=None, frame_ns=STARTED_NS + 1_000_000_000), False),
        (_document(book_first_ns=STARTED_NS - 1_000_000_000, frame_ns=STARTED_NS + 1_000_000_000), False),
        (_document(book_first_ns=STARTED_NS + 1_000_000_000, frame_ns=None), False),
        (_document(book_first_ns=STARTED_NS + 1_000_000_000, frame_ns=STARTED_NS - 1_000_000_000), False),
        (_document(book_first_ns=STARTED_NS + 1_000_000_000, frame_ns=STARTED_NS + 1_000_000_000, flush_failures=1), False),
    ],
)
def test_writer_ready_agrees_with_watchdog_and_deploy_gate(document: dict, ready: bool) -> None:
    started_at = datetime.fromisoformat(document["started_at"].replace("Z", "+00:00"))
    now = datetime.fromisoformat(document["ts"].replace("Z", "+00:00"))
    assert document["ready"] is ready
    assert _slot_ready(document) is ready
    assert (evaluate_ready(document, container_started_at=started_at, now=now, stale_s=3600) == "ready") is ready


def test_deploy_gate_rejects_stale_and_stopped_writer_documents() -> None:
    document = _document(book_first_ns=STARTED_NS + 1, frame_ns=STARTED_NS + 1)
    started_at = datetime.fromisoformat(document["started_at"].replace("Z", "+00:00"))
    now = datetime.fromisoformat(document["ts"].replace("Z", "+00:00"))
    assert evaluate_ready(document, container_started_at=started_at, now=now + timedelta(seconds=31), stale_s=30) == "waiting"

    stopped = _document(
        book_first_ns=STARTED_NS + 1,
        frame_ns=STARTED_NS + 1,
        stopped_ns=STARTED_NS + 4_000_000_000,
    )
    stopped_at = datetime.fromisoformat(stopped["started_at"].replace("Z", "+00:00"))
    stopped_now = datetime.fromisoformat(stopped["ts"].replace("Z", "+00:00"))
    assert evaluate_ready(stopped, container_started_at=stopped_at, now=stopped_now, stale_s=3600) == "waiting"
