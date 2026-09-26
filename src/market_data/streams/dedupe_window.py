"""In-memory cross-slot dedupe window of the live-capture normalizer (work saver, never persisted)."""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any


class DedupeWindow:
    """Bounded in-memory sets of recently derived REST grids and WS frame digests.

    Purely a work saver. Correctness under restart relies on the earliest-receipt merge keys, so the
    window is never persisted and is pruned by receipt time. It also carries the per-grid outcomes
    that heartbeat window counters are folded from (stats only, never correctness).

    Marks made during a cycle are staged and only become visible to later cycles through
    ``commit``: a cycle whose derived write fails calls ``discard`` so its records are derived again
    on retry instead of being dropped as duplicates of rows that never reached disk. A seen key only
    suppresses a later receipt, never an earlier one, so a twin that arrives in a later cycle with an
    earlier receipt still reaches the earliest-receipt merge.
    """

    def __init__(self, *, rest_window_s: float, ws_window_s: float) -> None:
        """Bound both horizons in seconds."""
        self._rest_window_ns = int(rest_window_s * 1_000_000_000)
        self._ws_window_ns = int(ws_window_s * 1_000_000_000)
        self._rest_seen: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._rest_outcomes: deque[tuple[int, str, bool, int, int, float]] = deque()
        self._fresh_outcomes: list[tuple[int, str, bool, int, int, float]] = []
        self._ws_seen: OrderedDict[str, int] = OrderedDict()
        self._ws_last_recv_ns: int | None = None
        self._staged_rest: dict[tuple[str, str], int] = {}
        self._staged_ws: dict[str, int] = {}
        self._staged_outcomes: list[tuple[int, str, bool, int, int, float]] = []
        self._staged_ws_last: int | None = None

    @staticmethod
    def _prune(seen: OrderedDict[Any, int], cutoff: int) -> None:
        # 삽입 순서가 대략 수신 순서이므로 왼쪽부터 만료분만 제거한다(분할 상환 O(1)).
        while seen:
            _key, recv = next(iter(seen.items()))
            if recv >= cutoff:
                break
            seen.popitem(last=False)

    def _prune_rest(self, now_ns: int) -> None:
        cutoff = now_ns - self._rest_window_ns
        self._prune(self._rest_seen, cutoff)
        while self._rest_outcomes and self._rest_outcomes[0][0] < cutoff:
            self._rest_outcomes.popleft()

    def _seen_rest(self, key: tuple[str, str]) -> int | None:
        staged = self._staged_rest.get(key)
        return staged if staged is not None else self._rest_seen.get(key)

    def check_rest(self, stream: str, grid: str, recv_ns: int, *, now_ns: int) -> bool:
        """Return True when ``(stream, grid)`` was already derived from a receipt no later than ``recv_ns``."""
        self._prune_rest(now_ns)
        seen = self._seen_rest((stream, grid))
        return seen is not None and seen <= recv_ns

    def rest_seen(self, stream: str, grid: str) -> bool:
        """Return True when ``(stream, grid)`` was derived by any receipt within the window."""
        return self._seen_rest((stream, grid)) is not None

    def mark_rest_success(self, stream: str, grid: str, recv_ns: int) -> None:
        """Stage a derived grid so its later slot twin is never parsed."""
        self._staged_rest[(stream, grid)] = recv_ns

    def record_rest_outcome(
        self, grid_ns: int, stream: str, ok: bool, rows: int, rejected_rows: int, rejected_fraction: float
    ) -> None:
        """Stage one grid outcome for heartbeat window counters."""
        self._staged_outcomes.append((grid_ns, stream, ok, rows, rejected_rows, rejected_fraction))

    def drain_fresh_outcomes(self) -> list[tuple[int, str, bool, int, int, float]]:
        """Return committed outcomes recorded since the last drain (heartbeat folding)."""
        fresh = list(self._fresh_outcomes)
        del self._fresh_outcomes[:]
        return fresh

    @property
    def rest_outcomes(self) -> deque[tuple[int, str, bool, int, int, float]]:
        """Windowed committed grid outcomes, oldest first."""
        return self._rest_outcomes

    def check_frame(self, digest: str, recv_ns: int, *, now_ns: int) -> bool:
        """Return True when the frame was already derived from a receipt no later than ``recv_ns``; otherwise stage it."""
        self._prune(self._ws_seen, now_ns - self._ws_window_ns)
        self._staged_ws_last = recv_ns if self._staged_ws_last is None else max(self._staged_ws_last, recv_ns)
        staged = self._staged_ws.get(digest)
        seen = staged if staged is not None else self._ws_seen.get(digest)
        if seen is not None and seen <= recv_ns:
            return True
        self._staged_ws[digest] = recv_ns
        return False

    def commit(self) -> None:
        """Make this cycle's staged marks and outcomes durable in the window (write succeeded)."""
        for key, recv in self._staged_rest.items():
            self._rest_seen.pop(key, None)
            self._rest_seen[key] = recv
        for digest, recv in self._staged_ws.items():
            self._ws_seen.pop(digest, None)
            self._ws_seen[digest] = recv
        self._rest_outcomes.extend(self._staged_outcomes)
        self._fresh_outcomes.extend(self._staged_outcomes)
        if self._staged_ws_last is not None:
            self._ws_last_recv_ns = (
                self._staged_ws_last if self._ws_last_recv_ns is None else max(self._ws_last_recv_ns, self._staged_ws_last)
            )
        self.discard()

    def discard(self) -> None:
        """Forget this cycle's staged marks (write failed; the records will be derived again)."""
        self._staged_rest = {}
        self._staged_ws = {}
        self._staged_outcomes = []
        self._staged_ws_last = None

    @property
    def ws_last_recv_ns(self) -> int | None:
        """Receipt instant of the latest committed WS frame, if any."""
        return self._ws_last_recv_ns
